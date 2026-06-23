import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay


def save_label_distribution(y, labels, path):
    y = np.asarray(y).astype(int)
    counts = y.sum(axis=0)
    order = np.argsort(counts)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.array(labels)[order], counts[order], color="#2a9d8f")
    ax.set_xlabel("Nombre de cas positifs")
    ax.set_title("Distribution des labels")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_roc_curves(y_true, y_prob, labels, path):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    fig, ax = plt.subplots(figsize=(8, 7))
    plotted = 0
    for i, label in enumerate(labels):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        RocCurveDisplay.from_predictions(
            y_true[:, i],
            y_prob[:, i],
            name=label,
            ax=ax,
        )
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "Aucune courbe ROC calculable", ha="center", va="center")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1)
    ax.set_title("Courbes ROC par pathologie")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_per_class_metric(per_class_df, metric, path):
    df = per_class_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["label"], df[metric], color="#457b9d")
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} par pathologie")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_cooccurrence_matrix(y, labels, path):
    y = np.asarray(y).astype(int)
    cooc = y.T @ y
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cooc, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=7)
    ax.set_title("Cooccurrences des pathologies")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_predictions_csv(image_ids, y_true, y_prob, labels, path):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    rows = {"sample_id": list(image_ids)}
    for i, label in enumerate(labels):
        rows[f"{label}_true"] = y_true[:, i]
        rows[f"{label}_prob"] = y_prob[:, i]
    pd.DataFrame(rows).to_csv(path, index=False)
