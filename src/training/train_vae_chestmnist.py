import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.chestmnist import ChestMNISTMultiLabel
from src.models.autoencoder import ConvVAE
from src.utils.common import configure_mlflow, denormalize_imagenet, ensure_dir, get_device, seed_everything


def vae_loss(reconstruction, target, mu, logvar, beta):
    per_image_reconstruction = torch.mean((reconstruction - target) ** 2, dim=(1, 2, 3))
    per_image_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    loss = per_image_reconstruction + beta * per_image_kl
    return loss.mean(), per_image_reconstruction.mean(), per_image_kl.mean()


def train_epoch(model, loader, optimizer, device, beta, max_batches=None):
    model.train()
    total_losses, reconstruction_losses, kl_losses = [], [], []
    for batch_idx, (x, _) in enumerate(tqdm(loader, leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        target = denormalize_imagenet(x)
        reconstruction, mu, logvar = model(x)
        loss, reconstruction_loss, kl_loss = vae_loss(reconstruction, target, mu, logvar, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_losses.append(float(loss.item()))
        reconstruction_losses.append(float(reconstruction_loss.item()))
        kl_losses.append(float(kl_loss.item()))
    return {
        "total_loss": float(np.mean(total_losses)),
        "reconstruction_loss": float(np.mean(reconstruction_losses)),
        "kl_loss": float(np.mean(kl_losses)),
    }


def collect_scores(model, loader, device, max_batches=None):
    model.eval()
    scores, labels = [], []
    example_inputs, example_reconstructions = None, None
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(loader, leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device)
            target = denormalize_imagenet(x)
            reconstruction, _, _ = model(x)
            per_image = torch.mean((reconstruction - target) ** 2, dim=(1, 2, 3))
            scores.extend(per_image.cpu().numpy().tolist())
            labels.extend((y.sum(dim=1) > 0).int().numpy().tolist())
            if example_inputs is None:
                example_inputs = target.cpu()
                example_reconstructions = reconstruction.cpu()
    return np.asarray(scores), np.asarray(labels), example_inputs, example_reconstructions


def anomaly_metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def save_score_distribution(scores, labels, threshold, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[labels == 0], bins=40, alpha=0.65, label="Sans pathologie")
    ax.hist(scores[labels == 1], bins=40, alpha=0.65, label="Avec pathologie")
    ax.axvline(threshold, color="#d62828", linestyle="--", label=f"Seuil={threshold:.5f}")
    ax.set_xlabel("Erreur de reconstruction")
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution des scores VAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_reconstruction_grid(inputs, reconstructions, path, max_items=6):
    n = min(max_items, inputs.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(8, 2.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    errors = torch.abs(inputs - reconstructions).mean(dim=1)
    for i in range(n):
        axes[i, 0].imshow(inputs[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(reconstructions[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Reconstruction VAE")
        axes[i, 2].imshow(errors[i].numpy(), cmap="magma")
        axes[i, 2].set_title("Erreur")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def finite_log_metrics(prefix, metrics):
    for name, value in metrics.items():
        if np.isfinite(value):
            mlflow.log_metric(f"{prefix}_{name}", float(value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/medmnist")
    ap.add_argument("--image_size", type=int, default=64, choices=[64, 128, 224])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--beta", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold_percentile", type=float, default=95.0)
    ap.add_argument("--checkpoint", default="checkpoints/vae_chestmnist.pt")
    ap.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_train_batches", type=int, default=None)
    ap.add_argument("--max_eval_batches", type=int, default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir("checkpoints")
    artifacts_dir = Path("outputs") / "chestmnist_vae" / datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ChestMNISTMultiLabel("train", args.data_root, args.image_size, args.download, train=True)
    val_ds = ChestMNISTMultiLabel("val", args.data_root, args.image_size, args.download, train=False)
    test_ds = ChestMNISTMultiLabel("test", args.data_root, args.image_size, args.download, train=False)

    train_normal_idx = np.flatnonzero(train_ds.targets.sum(axis=1) == 0).tolist()
    val_normal_idx = np.flatnonzero(val_ds.targets.sum(axis=1) == 0).tolist()
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_loader = DataLoader(Subset(train_ds, train_normal_idx), shuffle=True, **loader_kwargs)
    val_normal_loader = DataLoader(Subset(val_ds, val_normal_idx), shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    device = get_device()
    model = ConvVAE(image_size=args.image_size, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    configure_mlflow()
    mlflow.set_experiment("chestmnist_anomaly")
    is_smoke_test = args.max_train_batches is not None or args.max_eval_batches is not None
    run_name = "smoke_vae" if is_smoke_test else "conv_vae_normal_only"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("smoke_test", str(is_smoke_test).lower())
        mlflow.log_params(vars(args))
        mlflow.log_param("model_type", "vae")
        mlflow.log_param("normal_train_images", len(train_normal_idx))
        mlflow.log_param("normal_val_images", len(val_normal_idx))

        for epoch in range(args.epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, device, args.beta, args.max_train_batches)
            normal_scores, _, _, _ = collect_scores(model, val_normal_loader, device, args.max_eval_batches)
            val_loss = float(normal_scores.mean())
            mlflow.log_metric("train_total_loss", train_metrics["total_loss"], step=epoch)
            mlflow.log_metric("train_reconstruction_loss", train_metrics["reconstruction_loss"], step=epoch)
            mlflow.log_metric("train_kl_loss", train_metrics["kl_loss"], step=epoch)
            mlflow.log_metric("val_normal_reconstruction_loss", val_loss, step=epoch)
            print(
                f"epoch={epoch + 1} train_loss={train_metrics['total_loss']:.6f} "
                f"recon={train_metrics['reconstruction_loss']:.6f} "
                f"kl={train_metrics['kl_loss']:.6f} val_normal_loss={val_loss:.6f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model": model.state_dict()}, checkpoint)

        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
        normal_scores, _, _, _ = collect_scores(model, val_normal_loader, device, args.max_eval_batches)
        threshold = float(np.percentile(normal_scores, args.threshold_percentile))

        val_scores, val_labels, val_images, val_reconstructions = collect_scores(
            model, val_loader, device, args.max_eval_batches
        )
        test_scores, test_labels, _, _ = collect_scores(model, test_loader, device, args.max_eval_batches)
        val_metrics = anomaly_metrics(val_labels, val_scores, threshold)
        test_metrics = anomaly_metrics(test_labels, test_scores, threshold)
        finite_log_metrics("val", val_metrics)
        finite_log_metrics("test", test_metrics)

        pd.DataFrame({"label_anomaly": val_labels, "score": val_scores}).to_csv(
            artifacts_dir / "val_anomaly_scores.csv", index=False
        )
        pd.DataFrame({"label_anomaly": test_labels, "score": test_scores}).to_csv(
            artifacts_dir / "test_anomaly_scores.csv", index=False
        )
        save_score_distribution(val_scores, val_labels, threshold, artifacts_dir / "val_score_distribution.png")
        save_score_distribution(test_scores, test_labels, threshold, artifacts_dir / "test_score_distribution.png")
        save_reconstruction_grid(val_images, val_reconstructions, artifacts_dir / "val_reconstructions.png")

        torch.save(
            {
                "model": model.state_dict(),
                "model_type": "vae",
                "dataset": "ChestMNIST",
                "image_size": args.image_size,
                "latent_dim": args.latent_dim,
                "beta": args.beta,
                "threshold": threshold,
                "threshold_percentile": args.threshold_percentile,
                "protocol": "normal_train_only",
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "args": vars(args),
            },
            checkpoint,
        )
        mlflow.log_artifact(str(checkpoint))
        mlflow.log_artifacts(str(artifacts_dir))

        print("VALIDATION:", val_metrics)
        print("TEST:", test_metrics)
        print(f"checkpoint: {checkpoint}")
        print(f"artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()
