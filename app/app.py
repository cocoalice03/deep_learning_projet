import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "data/huggingface_cache"))

from transformers import AutoTokenizer

sys.path.append(PROJECT_ROOT)

from src.models.autoencoder import ConvAutoEncoder, ConvVAE
from src.models.multimodal import MultiModalNet
from src.models.supervised import build_model
from src.utils.common import LABELS, denormalize_imagenet, get_device


st.set_page_config(page_title="Triage radiologique", layout="wide")
st.title("Système d'aide au tri radiologique")
st.caption("Démonstrateur pédagogique : ne pas utiliser pour un diagnostic médical.")

device = get_device()


def image_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def available_supervised_checkpoints():
    checkpoints = sorted(Path("checkpoints").glob("*_best.pt"))
    preferred = [
        Path("checkpoints/chestmnist_resnet18_best.pt"),
        Path("checkpoints/chestmnist_densenet121_best.pt"),
        Path("checkpoints/chestmnist_vit_best.pt"),
        Path("checkpoints/resnet18_best.pt"),
    ]
    ordered = [p for p in preferred if p.exists()]
    ordered += [p for p in checkpoints if p not in ordered]
    return ordered


def available_anomaly_checkpoints():
    candidates = [
        Path("checkpoints/autoencoder_chestmnist.pt"),
        Path("checkpoints/vae_chestmnist.pt"),
        Path("checkpoints/autoencoder_openi.pt"),
    ]
    ordered = [p for p in candidates if p.exists()]
    remaining = sorted(Path("checkpoints").glob("*autoencoder*.pt")) + sorted(Path("checkpoints").glob("*vae*.pt"))
    ordered += [p for p in remaining if p not in ordered]
    return ordered


def normalize_openi_uid(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def difference_hash(image, hash_size=8):
    grayscale = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


@st.cache_data
def load_openi_report_lookup():
    csv_path = Path(PROJECT_ROOT) / "data/openi_metadata.csv"
    if not csv_path.exists():
        return {}, {}
    metadata = pd.read_csv(csv_path)
    by_filename = {}
    by_uid = {}
    for _, row in metadata.iterrows():
        text = str(row.get("text", "")).strip()
        uid = normalize_openi_uid(row.get("source_uid"))
        if not text or text.lower() == "nan":
            continue
        filename = Path(str(row.get("image_path", ""))).name.lower()
        if filename:
            by_filename[filename] = (uid, text)
        if uid:
            by_uid[uid] = text
    return by_filename, by_uid


@st.cache_data
def load_openi_visual_lookup():
    csv_path = Path(PROJECT_ROOT) / "data/openi_metadata.csv"
    if not csv_path.exists():
        return []
    metadata = pd.read_csv(csv_path)
    visual_lookup = []
    for _, row in metadata.iterrows():
        text = str(row.get("text", "")).strip()
        uid = normalize_openi_uid(row.get("source_uid"))
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.is_absolute():
            image_path = Path(PROJECT_ROOT) / image_path
        if not uid or not text or text.lower() == "nan" or not image_path.exists():
            continue
        try:
            with Image.open(image_path) as image:
                visual_lookup.append((difference_hash(image), uid, text))
        except OSError:
            continue
    return visual_lookup


def find_openi_report(filename, image=None):
    by_filename, by_uid = load_openi_report_lookup()
    normalized_filename = Path(filename).name.lower()
    if normalized_filename in by_filename:
        return by_filename[normalized_filename]

    match = re.match(r"cxr(\d+)(?:_|\b)", normalized_filename, flags=re.IGNORECASE)
    if match and match.group(1) in by_uid:
        uid = match.group(1)
        return uid, by_uid[uid]

    match = re.match(r"openi_(\d+)_frontal", normalized_filename, flags=re.IGNORECASE)
    if match and match.group(1) in by_uid:
        uid = match.group(1)
        return uid, by_uid[uid]

    if image is not None:
        uploaded_hash = difference_hash(image)
        candidates = load_openi_visual_lookup()
        if candidates:
            best_hash, uid, text = min(candidates, key=lambda candidate: (uploaded_hash ^ candidate[0]).bit_count())
            if (uploaded_hash ^ best_hash).bit_count() <= 6:
                return uid, text
    return None


@st.cache_resource
def load_supervised(path):
    ck = torch.load(path, map_location=device)
    labels = ck.get("labels", LABELS)
    arch = ck.get("arch", "resnet18")
    model = build_model(arch, len(labels), pretrained=False, image_size=int(ck.get("image_size", 224))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, labels, ck


@st.cache_resource
def load_anomaly_model(path):
    ck = torch.load(path, map_location=device)
    model_type = ck.get("model_type", "vae" if "vae" in Path(path).name.lower() else "autoencoder")
    if model_type == "vae":
        model = ConvVAE(
            image_size=int(ck.get("image_size", 64)),
            latent_dim=int(ck.get("latent_dim", 128)),
        ).to(device)
    else:
        model = ConvAutoEncoder().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@st.cache_resource
def load_multimodal():
    path = Path("checkpoints/multimodal_openi.pt")
    if not path.exists():
        return None, None, None, "Checkpoint multimodal absent."
    ck = torch.load(path, map_location=device)
    tokenizer_path = Path("checkpoints/multimodal_tokenizer")
    try:
        tokenizer_source = tokenizer_path if tokenizer_path.exists() else ck.get("text_model", "distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
        model = MultiModalNet(
            num_labels=len(ck.get("labels", LABELS)),
            image_model=ck.get("image_model", "resnet18"),
            text_model_name=ck.get("text_model", "distilbert-base-uncased"),
            pretrained_image=False,
            pretrained_text=False,
            text_config=ck.get("text_config"),
        ).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        return model, tokenizer, ck, None
    except Exception as exc:
        return None, None, None, str(exc)


checkpoints = available_supervised_checkpoints()
anomaly_checkpoints = available_anomaly_checkpoints()
selected_checkpoint = None
selected_anomaly_checkpoint = None
with st.sidebar:
    st.header("Modèles")
    if checkpoints:
        selected_checkpoint = st.selectbox("Checkpoint supervisé", checkpoints, format_func=lambda p: p.name)
    else:
        st.warning("Aucun checkpoint supervisé trouvé.")
    if anomaly_checkpoints:
        selected_anomaly_checkpoint = st.selectbox(
            "Checkpoint anomalie",
            anomaly_checkpoints,
            format_func=lambda p: p.name,
        )

uploaded = st.file_uploader("Radiographie thoracique", type=["png", "jpg", "jpeg"])
uploaded_image = Image.open(uploaded).convert("RGB") if uploaded is not None else None
openi_match = find_openi_report(uploaded.name, uploaded_image) if uploaded is not None else None
default_report = openi_match[1] if openi_match else ""
report_label = "Compte-rendu OpenI associé" if openi_match else "Compte-rendu radiologique optionnel"
report_key = f"report_{uploaded.name}" if uploaded is not None else "report_empty"
report_text = st.text_area(report_label, default_report, height=140, key=report_key)
if openi_match:
    st.caption(f"OpenI - étude {openi_match[0]} - compte-rendu chargé automatiquement")

if uploaded:
    image = uploaded_image
    col_img, col_pred = st.columns([1, 2])
    with col_img:
        st.image(image, caption="Image chargée", width="stretch")

    if selected_checkpoint is None:
        st.warning("Lance un entraînement supervisé avant d'utiliser le démonstrateur.")
    else:
        model, labels, ck = load_supervised(str(selected_checkpoint))
        image_size = int(ck.get("image_size", 224))
        threshold = float(ck.get("threshold", 0.5))
        x = image_transform(image_size)(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0].cpu().numpy()
        pred_df = pd.DataFrame({"pathologie": labels, "probabilite": probs})
        pred_df["decision"] = pred_df["probabilite"] >= threshold
        pred_df = pred_df.sort_values("probabilite", ascending=False)

        with col_pred:
            st.subheader("Prédictions supervisées")
            st.write(f"Dataset modèle : {ck.get('dataset', 'inconnu')} | seuil : {threshold:.2f}")
            st.dataframe(
                pred_df.assign(probabilite=lambda d: (100 * d["probabilite"]).round(1)),
                hide_index=True,
                width="stretch",
            )
            st.bar_chart(pred_df.set_index("pathologie")["probabilite"])

        st.subheader("Détection d'anomalie")
        if selected_anomaly_checkpoint is None:
            st.info(
                "Aucun modèle d'anomalie trouvé. Lance "
                "`python src/training/train_autoencoder_chestmnist.py --epochs 5 --image_size 64` "
                "ou `python src/training/train_vae_chestmnist.py --epochs 5 --image_size 64`."
            )
        else:
            anomaly_model, anomaly_ck = load_anomaly_model(str(selected_anomaly_checkpoint))
            anomaly_image_size = int(anomaly_ck.get("image_size", image_size))
            anomaly_x = image_transform(anomaly_image_size)(image).unsqueeze(0).to(device)
            anomaly_threshold = anomaly_ck.get("threshold")
            with torch.no_grad():
                output = anomaly_model(anomaly_x)
                rec = output[0] if isinstance(output, tuple) else output
                target = denormalize_imagenet(anomaly_x)
                score = torch.mean((rec - target) ** 2).item()
            c1, c2 = st.columns(2)
            c1.metric("Erreur de reconstruction", f"{score:.6f}")
            if anomaly_threshold is not None:
                c2.metric("Seuil appris", f"{float(anomaly_threshold):.6f}")
                anomaly_name = "VAE" if anomaly_ck.get("model_type") == "vae" else "Autoencoder"
                st.caption(f"{anomaly_name} : {anomaly_ck.get('dataset', 'OpenI')}")
                st.write("Décision :", "Atypique" if score > float(anomaly_threshold) else "Non atypique")

        st.subheader("Fusion image + texte")
        if not Path("checkpoints/multimodal_openi.pt").exists():
            st.info("Modèle multimodal non entraîné : les données image + rapport OpenI sont nécessaires.")
        elif report_text.strip():
            multimodal, tokenizer, multi_ck, error = load_multimodal()
            if multimodal is None:
                st.info(f"Modèle multimodal indisponible : {error}")
            else:
                multimodal_image_size = int(multi_ck.get("image_size", 224))
                multimodal_x = image_transform(multimodal_image_size)(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    enc = tokenizer(
                        [report_text],
                        truncation=True,
                        padding=True,
                        max_length=int(multi_ck.get("max_len", 256)),
                        return_tensors="pt",
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    multimodal_probs = torch.sigmoid(
                        multimodal(multimodal_x, enc["input_ids"], enc["attention_mask"])
                    )[0].cpu().numpy()
                multi_df = pd.DataFrame({"pathologie": multi_ck.get("labels", LABELS), "probabilite": multimodal_probs})
                multi_df["decision"] = multi_df["probabilite"] >= float(multi_ck.get("threshold", 0.5))
                st.dataframe(
                    multi_df.sort_values("probabilite", ascending=False).assign(
                        probabilite=lambda d: (100 * d["probabilite"]).round(1)
                    ),
                    hide_index=True,
                    width="stretch",
                )
        else:
            st.info("Ajoute un compte-rendu pour tester la brique multimodale.")
else:
    st.info("Charge une radiographie pour lancer l'analyse.")
