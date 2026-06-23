import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.chestmnist import ChestMNISTMultiLabel
from src.evaluation.plots import save_cooccurrence_matrix, save_label_distribution
from src.utils.common import LABELS, seed_everything


def save_sample_grid(dataset, path, n=16):
    n = min(n, len(dataset))
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9, 9))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for i in range(n):
        image, y = dataset.dataset[i]
        positive = [LABELS[j] for j, value in enumerate(np.asarray(y).reshape(-1)) if value == 1]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            if image.ndim == 3:
                image = image.permute(1, 2, 0)
            image = image.numpy()
            image = np.clip((image - image.min()) / max(float(image.max() - image.min()), 1e-8), 0, 1)
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(", ".join(positive[:2]) if positive else "Aucun label", fontsize=8)
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/medmnist")
    ap.add_argument("--image_size", type=int, default=224, choices=[28, 64, 128, 224])
    ap.add_argument("--out_dir", default="outputs/eda_chestmnist")
    ap.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        split: ChestMNISTMultiLabel(
            split=split,
            root=args.data_root,
            image_size=args.image_size,
            download=args.download,
            train=False,
        )
        for split in ["train", "val", "test"]
    }

    summary_rows = []
    for split, dataset in splits.items():
        targets = dataset.targets.astype(int)
        summary_rows.append(
            {
                "split": split,
                "samples": len(dataset),
                "mean_positive_labels_per_image": float(targets.sum(axis=1).mean()),
                "images_without_positive_label": int((targets.sum(axis=1) == 0).sum()),
            }
        )
        label_rows = []
        for i, label in enumerate(LABELS):
            label_rows.append(
                {
                    "split": split,
                    "label": label,
                    "positives": int(targets[:, i].sum()),
                    "prevalence": float(targets[:, i].mean()),
                }
            )
        pd.DataFrame(label_rows).to_csv(out_dir / f"{split}_label_distribution.csv", index=False)
        save_label_distribution(targets, LABELS, out_dir / f"{split}_label_distribution.png")
        save_cooccurrence_matrix(targets, LABELS, out_dir / f"{split}_cooccurrences.png")

    save_sample_grid(splits["train"], out_dir / "train_examples.png")
    pd.DataFrame(summary_rows).to_csv(out_dir / "split_summary.csv", index=False)
    print(f"EDA ChestMNIST sauvegardee dans {out_dir}")


if __name__ == "__main__":
    main()
