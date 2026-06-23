import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.chestmnist import ChestMNISTMultiLabel, compute_pos_weight, label_distribution
from src.evaluation.metrics import find_best_threshold, multilabel_metrics, per_label_metrics
from src.evaluation.plots import (
    save_cooccurrence_matrix,
    save_label_distribution,
    save_per_class_metric,
    save_predictions_csv,
    save_roc_curves,
)
from src.models.supervised import build_model
from src.utils.common import LABELS, configure_mlflow, ensure_dir, get_device, seed_everything


def finite_log_metric(name, value, step=None):
    if value is None:
        return
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(value):
        mlflow.log_metric(name, value, step=step)


def log_metrics(prefix, metrics, step=None):
    for key, value in metrics.items():
        finite_log_metric(f"{prefix}_{key}", value, step=step)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu", max_batches=None):
    train = optimizer is not None
    model.train(train)
    losses, probs, ys = [], [], []

    for batch_idx, (x, y) in enumerate(tqdm(loader, leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())

    return float(np.mean(losses)), np.vstack(ys), np.vstack(probs)


def build_loaders(args):
    train_ds = ChestMNISTMultiLabel(
        split="train",
        root=args.data_root,
        image_size=args.image_size,
        download=args.download,
        train=True,
    )
    val_ds = ChestMNISTMultiLabel(
        split="val",
        root=args.data_root,
        image_size=args.image_size,
        download=args.download,
        train=False,
    )
    test_ds = ChestMNISTMultiLabel(
        split="test",
        root=args.data_root,
        image_size=args.image_size,
        download=args.download,
        train=False,
    )
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def save_distribution_artifacts(train_ds, artifacts_dir):
    rows = []
    distribution = label_distribution(train_ds)
    for label, stats in distribution.items():
        rows.append({"label": label, **stats})
    pd.DataFrame(rows).to_csv(artifacts_dir / "train_label_distribution.csv", index=False)
    save_label_distribution(train_ds.targets, LABELS, artifacts_dir / "train_label_distribution.png")
    save_cooccurrence_matrix(train_ds.targets, LABELS, artifacts_dir / "train_label_cooccurrences.png")


def save_eval_artifacts(split, y_true, y_prob, threshold, artifacts_dir):
    metrics = multilabel_metrics(y_true, y_prob, threshold=threshold)
    per_class = per_label_metrics(y_true, y_prob, LABELS, threshold=threshold)
    per_class.to_csv(artifacts_dir / f"{split}_per_class_metrics.csv", index=False)
    pd.DataFrame([metrics]).to_csv(artifacts_dir / f"{split}_global_metrics.csv", index=False)
    save_roc_curves(y_true, y_prob, LABELS, artifacts_dir / f"{split}_roc_curves.png")
    save_per_class_metric(per_class, "f1", artifacts_dir / f"{split}_per_class_f1.png")
    save_predictions_csv(
        [f"{split}_{i}" for i in range(len(y_true))],
        y_true,
        y_prob,
        LABELS,
        artifacts_dir / f"{split}_predictions.csv",
    )
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/medmnist")
    ap.add_argument("--model", default="resnet18", choices=["simple_cnn", "resnet18", "densenet121", "vit"])
    ap.add_argument("--image_size", type=int, default=224, choices=[28, 64, 128, 224])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_balance", choices=["none", "pos_weight"], default="pos_weight")
    ap.add_argument("--selection_metric", choices=["auc_macro", "average_precision_macro", "f1_macro"], default="average_precision_macro")
    ap.add_argument("--early_stopping_patience", type=int, default=5)
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--optimize_threshold", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_train_batches", type=int, default=None)
    ap.add_argument("--max_eval_batches", type=int, default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir("checkpoints")
    artifacts_dir = Path("outputs") / "chestmnist" / f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(args)
    save_distribution_artifacts(train_ds, artifacts_dir)

    device = get_device()
    model = build_model(args.model, len(LABELS), pretrained=args.pretrained, image_size=args.image_size).to(device)

    pos_weight = None
    if args.class_balance == "pos_weight":
        pos_weight = torch.tensor(compute_pos_weight(train_ds.targets), device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    ckpt = Path("checkpoints") / f"chestmnist_{args.model}_best.pt"
    best_score = -1.0
    epochs_without_improvement = 0

    config_path = artifacts_dir / "run_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    configure_mlflow()
    mlflow.set_experiment("chestmnist_supervised")
    with mlflow.start_run(run_name=f"chestmnist_{args.model}"):
        mlflow.log_params(vars(args))
        mlflow.log_param("labels", ",".join(LABELS))
        if pos_weight is not None:
            for label, value in zip(LABELS, pos_weight.detach().cpu().numpy()):
                finite_log_metric(f"pos_weight_{label}", value)

        for epoch in range(args.epochs):
            train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, args.max_train_batches)
            val_loss, y_val, p_val = run_epoch(model, val_loader, criterion, None, device, args.max_eval_batches)
            val_metrics = multilabel_metrics(y_val, p_val, threshold=0.5)

            finite_log_metric("train_loss", train_loss, step=epoch)
            finite_log_metric("val_loss", val_loss, step=epoch)
            log_metrics("val", val_metrics, step=epoch)

            selection_score = val_metrics.get(args.selection_metric, float("nan"))
            if not np.isfinite(selection_score):
                selection_score = val_metrics.get("f1_macro", 0.0)
            scheduler.step(float(selection_score))

            print(
                f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"{args.selection_metric}={selection_score:.4f}"
            )

            if selection_score > best_score:
                best_score = float(selection_score)
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "labels": LABELS,
                        "arch": args.model,
                        "dataset": "ChestMNIST",
                        "image_size": args.image_size,
                        "threshold": 0.5,
                        "pretrained": args.pretrained,
                        "selection_metric": args.selection_metric,
                        "best_score": best_score,
                    },
                    ckpt,
                )
            else:
                epochs_without_improvement += 1

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping apres {epoch + 1} epochs")
                break

        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        val_loss, y_val, p_val = run_epoch(model, val_loader, criterion, None, device, args.max_eval_batches)

        threshold = 0.5
        if args.optimize_threshold:
            threshold, sweep = find_best_threshold(y_val, p_val, metric="f1_macro")
            sweep.to_csv(artifacts_dir / "val_threshold_sweep.csv", index=False)
        finite_log_metric("best_threshold", threshold)

        val_metrics = save_eval_artifacts("val", y_val, p_val, threshold, artifacts_dir)
        test_loss, y_test, p_test = run_epoch(model, test_loader, criterion, None, device, args.max_eval_batches)
        test_metrics = save_eval_artifacts("test", y_test, p_test, threshold, artifacts_dir)

        finite_log_metric("best_val_loss", val_loss)
        finite_log_metric("test_loss", test_loss)
        log_metrics("final_val", val_metrics)
        log_metrics("test", test_metrics)

        torch.save(
            {
                "model": model.state_dict(),
                "labels": LABELS,
                "arch": args.model,
                "dataset": "ChestMNIST",
                "image_size": args.image_size,
                "threshold": threshold,
                "pretrained": args.pretrained,
                "selection_metric": args.selection_metric,
                "best_score": best_score,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "args": vars(args),
            },
            ckpt,
        )
        mlflow.log_artifact(str(ckpt))
        mlflow.log_artifacts(str(artifacts_dir))

        print("VALIDATION:", val_metrics)
        print("TEST:", test_metrics)
        print(f"checkpoint: {ckpt}")
        print(f"artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()
