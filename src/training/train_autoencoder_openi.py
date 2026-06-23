import argparse, os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, torch, mlflow
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import OpenIXrayDataset
from src.models.autoencoder import ConvAutoEncoder
from src.utils.common import configure_mlflow, denormalize_imagenet, seed_everything, get_device, ensure_dir

def denorm(x):
    return denormalize_imagenet(x)

def finite_log_metric(name, value, step=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(value):
        mlflow.log_metric(name, value, step=step)

def save_reconstruction_grid(inputs, reconstructions, path, max_items=6):
    n = min(max_items, inputs.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(8, 2.6 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    error = torch.abs(inputs - reconstructions).mean(dim=1, keepdim=True)
    for i in range(n):
        for ax in axes[i]:
            ax.axis("off")
        axes[i, 0].imshow(inputs[i].detach().cpu().permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(reconstructions[i].detach().cpu().permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Reconstruction")
        axes[i, 2].imshow(error[i, 0].detach().cpu().numpy(), cmap="magma")
        axes[i, 2].set_title("Erreur")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/openi_metadata.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    seed_everything(42); ensure_dir("checkpoints")
    artifacts_dir = Path("outputs") / "openi_anomaly" / datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    train_ds = OpenIXrayDataset(args.csv, train_idx, train=True)
    val_ds = OpenIXrayDataset(args.csv, val_idx)
    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    va = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = get_device()
    model = ConvAutoEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss(reduction="none")
    ckpt = "checkpoints/autoencoder_openi.pt"
    best = 1e9

    configure_mlflow()
    mlflow.set_experiment("openi_anomaly")
    with mlflow.start_run(run_name="conv_autoencoder"):
        mlflow.log_params(vars(args))
        for e in range(args.epochs):
            model.train(); losses=[]
            for x,y in tqdm(tr, leave=False):
                x = x.to(device); target = denorm(x)
                rec = model(x)
                loss = loss_fn(rec, target).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            model.eval(); vloss=[]; scores=[]
            with torch.no_grad():
                for x,y in va:
                    x=x.to(device); target=denorm(x); rec=model(x)
                    per = loss_fn(rec,target).mean(dim=[1,2,3])
                    vloss.append(per.mean().item()); scores += per.cpu().numpy().tolist()
            val = float(np.mean(vloss)); thr = float(np.percentile(scores, 95))
            finite_log_metric("train_loss", float(np.mean(losses)), e)
            finite_log_metric("val_reconstruction_loss", val, e)
            finite_log_metric("anomaly_threshold_p95", thr, e)
            print(e+1, np.mean(losses), val, "threshold", thr)
            if val < best:
                best = val
                torch.save({"model": model.state_dict(), "threshold": thr}, ckpt)
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(va))
            x = x.to(device)
            target = denorm(x)
            rec = model(x)
            save_reconstruction_grid(target, rec, artifacts_dir / "validation_reconstructions.png")
        mlflow.log_artifact(ckpt)
        mlflow.log_artifacts(str(artifacts_dir))

if __name__ == "__main__":
    main()
