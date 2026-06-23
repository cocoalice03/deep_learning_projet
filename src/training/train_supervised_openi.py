import argparse, os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, torch, mlflow
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import OpenIXrayDataset
from src.models.supervised import build_model
from src.evaluation.metrics import find_best_threshold, multilabel_metrics, per_label_metrics
from src.evaluation.plots import save_label_distribution, save_per_class_metric, save_predictions_csv, save_roc_curves
from src.utils.common import configure_mlflow, seed_everything, get_device, ensure_dir, LABELS

def finite_log_metric(name, value, step=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(value):
        mlflow.log_metric(name, value, step=step)

def log_metrics(prefix, metrics, step=None):
    for key, value in metrics.items():
        finite_log_metric(f"{prefix}_{key}", value, step=step)

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)
    losses, probs, ys = [], [], []
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
    return np.mean(losses), np.vstack(ys), np.vstack(probs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/openi_metadata.csv")
    ap.add_argument("--model", default="resnet18", choices=["simple_cnn","resnet18","densenet121","vit"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--optimize_threshold", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    seed_everything(42)
    ensure_dir("checkpoints")
    df = pd.read_csv(args.csv)
    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(idx, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_ds = OpenIXrayDataset(args.csv, train_idx, train=True)
    val_ds = OpenIXrayDataset(args.csv, val_idx)
    test_ds = OpenIXrayDataset(args.csv, test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    artifacts_dir = Path("outputs") / "openi_supervised" / f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_label_distribution(df[LABELS].values, LABELS, artifacts_dir / "label_distribution.png")
    df[LABELS].sum().rename("positives").to_csv(artifacts_dir / "label_distribution.csv")

    device = get_device()
    model = build_model(args.model, len(LABELS), pretrained=args.pretrained).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    configure_mlflow()
    mlflow.set_experiment("openi_supervised")
    best_auc = -1
    ckpt = f"checkpoints/{args.model}_best.pt"
    with mlflow.start_run(run_name=args.model):
        mlflow.log_params(vars(args))
        for epoch in range(args.epochs):
            tr_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, yv, pv = run_epoch(model, val_loader, criterion, None, device)
            mets = multilabel_metrics(yv, pv)
            finite_log_metric("train_loss", tr_loss, epoch)
            finite_log_metric("val_loss", va_loss, epoch)
            log_metrics("val", mets, epoch)
            print(epoch+1, "train", tr_loss, "val", va_loss, mets)
            auc = mets.get("auc_macro", 0)
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                torch.save({"model": model.state_dict(), "labels": LABELS, "arch": args.model, "dataset": "OpenI", "threshold": 0.5, "pretrained": args.pretrained}, ckpt)

        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        _, yv, pv = run_epoch(model, val_loader, criterion, None, device)
        threshold = 0.5
        if args.optimize_threshold:
            threshold, sweep = find_best_threshold(yv, pv, metric="f1_macro")
            sweep.to_csv(artifacts_dir / "val_threshold_sweep.csv", index=False)
        finite_log_metric("best_threshold", threshold)
        te_loss, yt, pt = run_epoch(model, test_loader, criterion, None, device)
        tm = multilabel_metrics(yt, pt, threshold=threshold)
        per_class = per_label_metrics(yt, pt, LABELS, threshold=threshold)
        per_class.to_csv(artifacts_dir / "test_per_class_metrics.csv", index=False)
        pd.DataFrame([tm]).to_csv(artifacts_dir / "test_global_metrics.csv", index=False)
        save_roc_curves(yt, pt, LABELS, artifacts_dir / "test_roc_curves.png")
        save_per_class_metric(per_class, "f1", artifacts_dir / "test_per_class_f1.png")
        save_predictions_csv([f"test_{i}" for i in range(len(yt))], yt, pt, LABELS, artifacts_dir / "test_predictions.csv")
        log_metrics("test", tm)
        torch.save({"model": model.state_dict(), "labels": LABELS, "arch": args.model, "dataset": "OpenI", "threshold": threshold, "pretrained": args.pretrained, "test_metrics": tm}, ckpt)
        mlflow.log_artifact(ckpt)
        mlflow.log_artifacts(str(artifacts_dir))
        print("TEST:", tm)

if __name__ == "__main__":
    main()
