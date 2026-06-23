import argparse, os, sys
import numpy as np, pandas as pd, torch, mlflow
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "data/huggingface_cache"))

from transformers import AutoTokenizer

sys.path.append(PROJECT_ROOT)

from src.data.dataset import OpenIXrayDataset
from src.models.multimodal import MultiModalNet
from src.utils.common import LABELS, configure_mlflow, seed_everything, get_device, ensure_dir
from src.evaluation.metrics import find_best_threshold, multilabel_metrics

def collate(batch):
    imgs, texts, ys = zip(*batch)
    return torch.stack(imgs), list(texts), torch.stack(ys)

def finite_log_metric(name, value, step=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(value):
        mlflow.log_metric(name, value, step=step)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/openi_metadata.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--image_model", default="resnet18", choices=["resnet18","densenet121"])
    ap.add_argument("--text_model", default="distilbert-base-uncased")
    ap.add_argument("--text_checkpoint", default="checkpoints/text_only_distilbert")
    ap.add_argument("--freeze_encoders", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num_workers", type=int, default=0)
    args=ap.parse_args()
    seed_everything(42); ensure_dir("checkpoints")
    df=pd.read_csv(args.csv); idx=np.arange(len(df)); tr,va=train_test_split(idx,test_size=.2,random_state=42)
    ds_tr=OpenIXrayDataset(args.csv,tr,image_size=args.image_size,train=True,with_text=True)
    ds_va=OpenIXrayDataset(args.csv,va,image_size=args.image_size,with_text=True)
    dl_tr=DataLoader(ds_tr,batch_size=args.batch_size,shuffle=True,collate_fn=collate,num_workers=args.num_workers); dl_va=DataLoader(ds_va,batch_size=args.batch_size,collate_fn=collate,num_workers=args.num_workers)
    text_source = args.text_checkpoint if Path(args.text_checkpoint).exists() else args.text_model
    tok=AutoTokenizer.from_pretrained(text_source)
    device=get_device()
    model=MultiModalNet(len(LABELS), image_model=args.image_model, text_model_name=text_source).to(device)
    if args.freeze_encoders:
        for parameter in model.image.parameters():
            parameter.requires_grad = False
        for parameter in model.text.parameters():
            parameter.requires_grad = False

    train_targets = ds_tr.df[LABELS].to_numpy(dtype="float32")
    positives = train_targets.sum(axis=0)
    negatives = len(train_targets) - positives
    pos_weight = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 20.0)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    opt=torch.optim.AdamW(trainable_parameters, lr=args.lr)
    configure_mlflow()
    mlflow.set_experiment("openi_multimodal")
    ckpt="checkpoints/multimodal_openi.pt"
    with mlflow.start_run(run_name="image_text_fusion"):
        mlflow.log_params(vars(args))
        mlflow.log_param("text_source", str(text_source))
        best_score = -1.0
        for e in range(args.epochs):
            model.train()
            if args.freeze_encoders:
                model.image.eval()
                model.text.eval()
            train_loss = 0.0
            for imgs,texts,y in dl_tr:
                enc=tok(texts,truncation=True,padding=True,max_length=args.max_len,return_tensors="pt")
                imgs=imgs.to(device); y=y.to(device); enc={k:v.to(device) for k,v in enc.items()}
                logits=model(imgs, enc["input_ids"], enc["attention_mask"])
                loss=loss_fn(logits,y); opt.zero_grad(); loss.backward(); opt.step()
                train_loss += loss.item() * len(y)
            model.eval(); probs=[]; ys=[]
            with torch.no_grad():
                for imgs,texts,y in dl_va:
                    enc=tok(texts,truncation=True,padding=True,max_length=args.max_len,return_tensors="pt")
                    imgs=imgs.to(device); enc={k:v.to(device) for k,v in enc.items()}
                    logits=model(imgs,enc["input_ids"],enc["attention_mask"])
                    probs.append(torch.sigmoid(logits).cpu().numpy()); ys.append(y.numpy())
            y_true=np.vstack(ys); y_prob=np.vstack(probs)
            threshold,_=find_best_threshold(y_true,y_prob)
            m=multilabel_metrics(y_true,y_prob,threshold=threshold)
            train_loss /= len(ds_tr)
            finite_log_metric("train_loss",train_loss,e)
            for k,v in m.items(): finite_log_metric("val_"+k,v,e)
            score=float(m.get("average_precision_macro", float("nan")))
            print(f"epoch={e+1} train_loss={train_loss:.4f} threshold={threshold:.2f} metrics={m}")
            if np.isfinite(score) and score > best_score:
                best_score = score
                torch.save({
                    "model":model.state_dict(),
                    "labels":LABELS,
                    "image_model":args.image_model,
                    "text_model":args.text_model,
                    "text_config":model.text.config.to_dict(),
                    "max_len":args.max_len,
                    "image_size":args.image_size,
                    "threshold":threshold,
                    "validation_metrics":m,
                    "dataset":"OpenI",
                },ckpt)
        tokenizer_dir=Path("checkpoints/multimodal_tokenizer")
        tok.save_pretrained(tokenizer_dir)
        mlflow.log_artifact(ckpt)
        mlflow.log_artifacts(str(tokenizer_dir), artifact_path="multimodal_tokenizer")
        print(f"checkpoint: {ckpt}")
if __name__=="__main__": main()
