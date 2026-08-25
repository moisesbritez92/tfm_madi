#!/usr/bin/env python3
"""Generate thesis figures from the stored encoder experiment results."""

import base64
import gzip
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "logs_entrenamiento_2026-08-24" / "raw"
IMG_DIR = ROOT / "memoria" / "img"

VARIANTS = {
    "v0": ("V0: ResNet-18 desde cero", "#0072B2"),
    "v1": ("V1: ResNet-18 congelada", "#D55E00"),
    "v2": ("V2: ResNet-18 con ajuste fino", "#009E73"),
}


def load_records(variant):
    path = RAW_DIR / f"{variant}_logs_json.txt.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def completed_epochs(records):
    # Keeping the last aggregate for each epoch selects the resumed V2 branch.
    by_epoch = {}
    for record in records:
        if "val_loss" in record:
            by_epoch[int(record["epoch"])] = record
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def standard_error(record):
    """Standard error of the mean over the 50 individual episode scores."""
    scores = [
        value
        for key, value in record.items()
        if key.startswith("test/sim_max_reward_")
    ]
    assert len(scores) == 50
    return statistics.stdev(scores) / math.sqrt(len(scores))


def style_axes(axis):
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_test_scores(records_by_variant):
    fig, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)

    for variant, (label, color) in VARIANTS.items():
        evaluations = [
            row for row in records_by_variant[variant] if "test/mean_score" in row
        ]
        epochs = [row["epoch"] for row in evaluations]
        scores = [row["test/mean_score"] for row in evaluations]
        errors = [standard_error(row) for row in evaluations]
        axis.errorbar(
            epochs,
            scores,
            yerr=errors,
            color=color,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            elinewidth=0.9,
            capsize=2.5,
            label=label,
        )

        best = max(evaluations, key=lambda row: row["test/mean_score"])
        axis.scatter(
            [best["epoch"]],
            [best["test/mean_score"]],
            color=color,
            marker="*",
            s=95,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    axis.set_xlim(0, 500)
    axis.set_ylim(0, 1.0)
    axis.set_xlabel("\u00c9poca")
    axis.set_ylabel("Puntuaci\u00f3n media de evaluaci\u00f3n")
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(axis)
    fig.savefig(IMG_DIR / "evolucion_puntuacion_prueba.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_losses(records_by_variant):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    for variant, (label, color) in VARIANTS.items():
        epochs = completed_epochs(records_by_variant[variant])
        x = [row["epoch"] for row in epochs]
        axes[0].plot(
            x,
            [row["train_loss"] for row in epochs],
            color=color,
            linewidth=1.4,
            label=label,
        )
        axes[1].plot(
            x,
            [row["val_loss"] for row in epochs],
            color=color,
            linewidth=1.4,
            label=label,
        )

    axes[0].set_title("Entrenamiento")
    axes[1].set_title("Validaci\u00f3n")
    for axis in axes:
        axis.set_xlabel("\u00c9poca")
        axis.set_ylabel("P\u00e9rdida")
        axis.set_yscale("log")
        style_axes(axis)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(IMG_DIR / "evolucion_perdidas.pdf", bbox_inches="tight")
    plt.close(fig)


def extract_rollout_figures():
    for variant in ("v1", "v2"):
        path = ROOT / "diffuser" / f"inferencia_{variant}_pusht.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        image_data = None
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                payload = output.get("data", {}).get("image/png")
                if payload:
                    image_data = "".join(payload) if isinstance(payload, list) else payload
                    break
            if image_data:
                break
        if image_data is None:
            raise RuntimeError(f"No stored rollout figure found in {path}")
        (IMG_DIR / f"inferencia_{variant}.png").write_bytes(
            base64.b64decode(image_data)
        )


def validate(records_by_variant):
    expected = {
        "v0": (10, 350, 0.8645003451686891, 499),
        "v1": (10, 150, 0.6675816882533911, 499),
        "v2": (6, 150, 0.6477134479849783, 265),
    }
    for variant, (count, best_epoch, best_score, last_complete) in expected.items():
        evaluations = [
            row for row in records_by_variant[variant] if "test/mean_score" in row
        ]
        best = max(evaluations, key=lambda row: row["test/mean_score"])
        completed = completed_epochs(records_by_variant[variant])
        assert len(evaluations) == count
        assert best["epoch"] == best_epoch
        assert abs(best["test/mean_score"] - best_score) < 1e-12
        assert completed[-1]["epoch"] == last_complete


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    records_by_variant = {
        variant: load_records(variant) for variant in VARIANTS
    }
    validate(records_by_variant)
    plot_test_scores(records_by_variant)
    plot_losses(records_by_variant)
    extract_rollout_figures()
    print(f"Figures written to {IMG_DIR}")


if __name__ == "__main__":
    main()
