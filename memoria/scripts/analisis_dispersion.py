#!/usr/bin/env python3
"""Dispersion and paired tests over the per-episode scores of the encoder experiment.

The rollout logs store the individual coverage score of each of the 50 evaluation
episodes (``test/sim_max_reward_<seed>``), so the dispersion of the reported metric
can be characterised without running the simulator again.
"""

import csv
import gzip
import json
import math
import statistics
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "logs_entrenamiento" / "raw"
OUT_DIR = ROOT / "memoria" / "datos"

VARIANTS = {
    "v0": "V0: ResNet-18 desde cero",
    "v1": "V1: ResNet-18 congelada",
    "v2": "V2: ResNet-18 con ajuste fino",
    "v3": "V3: DINOv2 ViT-S/14 congelada",
    "v4": "V4: CLIP ViT-B/16 congelada",
}

# Selected checkpoint and published mean score of each variant.
SELECTED = {
    "v0": (350, 0.8645003451686891),
    "v1": (150, 0.6675816882533911),
    "v2": (150, 0.6477134479849783),
    "v3": (100, 0.6224012451194982),
    "v4": (100, 0.5351329876277153),
}

LAST_K = 3


def load_evaluations(variant):
    """Return {epoch: [s_j]} for every rollout evaluation of a variant."""
    path = RAW_DIR / f"{variant}_logs_json.txt.gz"
    evaluations = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "test/mean_score" not in record:
                continue
            keys = sorted(k for k in record if k.startswith("test/sim_max_reward_"))
            scores = [record[k] for k in keys]
            # The stored mean must be reproducible from the individual episodes.
            assert len(scores) == 50, f"{variant} epoch {record['epoch']}: {len(scores)}"
            assert abs(statistics.fmean(scores) - record["test/mean_score"]) < 1e-12
            evaluations[int(record["epoch"])] = scores
    return evaluations


def describe(scores):
    n = len(scores)
    mean = statistics.fmean(scores)
    sd = statistics.stdev(scores)
    se = sd / math.sqrt(n)
    return {
        "n": n,
        "media": mean,
        "desviacion": sd,
        "error_estandar": se,
        "ic95_inf": mean - 1.96 * se,
        "ic95_sup": mean + 1.96 * se,
    }


def validate(evaluations_by_variant):
    expected_counts = {"v0": 10, "v1": 10, "v2": 6, "v3": 4, "v4": 4}
    for variant, (epoch, score) in SELECTED.items():
        evaluations = evaluations_by_variant[variant]
        assert len(evaluations) == expected_counts[variant]
        assert max(evaluations, key=lambda e: statistics.fmean(evaluations[e])) == epoch
        assert abs(statistics.fmean(evaluations[epoch]) - score) < 1e-12


def main():
    evaluations_by_variant = {v: load_evaluations(v) for v in VARIANTS}
    validate(evaluations_by_variant)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, label in VARIANTS.items():
        evaluations = evaluations_by_variant[variant]
        epoch = SELECTED[variant][0]
        stats = describe(evaluations[epoch])
        epochs = sorted(evaluations)
        last_k = statistics.fmean(
            statistics.fmean(evaluations[e]) for e in epochs[-LAST_K:]
        )
        rows.append(
            {
                "variante": variant.upper(),
                "epoca": epoch,
                "evaluaciones": len(epochs),
                **{k: round(v, 6) for k, v in stats.items()},
                f"media_ultimas_{LAST_K}": round(last_k, 6),
                "ultima_evaluacion": round(statistics.fmean(evaluations[epochs[-1]]), 6),
            }
        )
        print(
            f"{label}: epoca {epoch} · media {stats['media']:.4f} "
            f"± {stats['error_estandar']:.4f} (e.e.) · s {stats['desviacion']:.4f} · "
            f"IC95 [{stats['ic95_inf']:.3f}; {stats['ic95_sup']:.3f}] · "
            f"media ultimas {LAST_K}: {last_k:.3f}"
        )

    with (OUT_DIR / "dispersion_puntuaciones.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print()
    comparisons = []
    pairs = (
        ("v0", "v1"),
        ("v0", "v2"),
        ("v1", "v2"),
        ("v0", "v3"),
        ("v0", "v4"),
        ("v1", "v3"),
        ("v3", "v4"),
    )
    for first, second in pairs:
        a = evaluations_by_variant[first][SELECTED[first][0]]
        b = evaluations_by_variant[second][SELECTED[second][0]]
        # Normal approximation: the paired differences contain zeros, so the
        # exact distribution is not applicable.
        statistic, pvalue = wilcoxon(a, b, zero_method="wilcox", mode="approx")
        paired = describe([x - y for x, y in zip(a, b)])
        non_zero = sum(1 for x, y in zip(a, b) if x != y)
        comparisons.append(
            {
                "comparacion": f"{first.upper()}-{second.upper()}",
                "diferencia_medias": round(paired["media"], 6),
                "error_estandar": round(paired["error_estandar"], 6),
                "ic95_inf": round(paired["ic95_inf"], 6),
                "ic95_sup": round(paired["ic95_sup"], 6),
                "pares_no_nulos": non_zero,
                "W": statistic,
                "p": round(pvalue, 6),
            }
        )
        print(
            f"{first.upper()} vs {second.upper()}: diferencia {paired['media']:+.4f} "
            f"± {paired['error_estandar']:.4f} · "
            f"IC95 [{paired['ic95_inf']:.3f}; {paired['ic95_sup']:.3f}] · "
            f"pares no nulos {non_zero} · W = {statistic:.0f} · p = {pvalue:.4g}"
        )

    with (OUT_DIR / "wilcoxon_puntuaciones.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    print(f"\nResultados escritos en {OUT_DIR}")


if __name__ == "__main__":
    main()
