#!/usr/bin/env python3
"""Cost of the three runs had the early-stopping criterion been applied uniformly.

The criterion described in the methodology stops a run when the evaluation score
fails to beat its running maximum twice in a row, never before epoch 200. It was
adopted after V0 and V1 had already exhausted the 500-epoch budget, so the reported
cost does not correspond to the declared protocol. This script recovers what the
cost would have been, from the stored logs alone.
"""

import csv
import gzip
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs_entrenamiento"

VARIANTS = ("v0", "v1", "v2", "v3", "v4")
PATIENCE = 2
MIN_EPOCH = 200

# Epochs completed and total wall-clock hours actually reported in the memoir.
REPORTED = {
    "v0": (500, 17.4),
    "v1": (500, 96.5),
    "v2": (266, 84.9),
    "v3": (154, 10.9),
    "v4": (200, 27.4),
}


def evaluation_scores(variant):
    path = LOG_DIR / "raw" / f"{variant}_logs_json.txt.gz"
    scores = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "test/mean_score" in record:
                scores[int(record["epoch"])] = record["test/mean_score"]
    return scores


def stopping_epoch(scores):
    """Return (stop epoch, epoch of the checkpoint kept) under the uniform criterion."""
    best_epoch, best_score, misses = None, float("-inf"), 0
    for epoch in sorted(scores):
        if scores[epoch] > best_score:
            best_epoch, best_score, misses = epoch, scores[epoch], 0
            continue
        misses += 1
        if misses >= PATIENCE and epoch >= MIN_EPOCH:
            return epoch, best_epoch, best_score
    return None, best_epoch, best_score


def loop_hours(variant, until_epoch):
    path = LOG_DIR / f"{variant}_seed42_epocas.csv"
    rows = [r for r in csv.DictReader(path.open()) if r["tipo"] == "entrenamiento"]
    timed = [(int(r["epoch"]), float(r["segundos"])) for r in rows]
    mean_seconds = statistics.fmean(s for _, s in timed)
    counted = [s for e, s in timed if e <= until_epoch]
    # V0 and V2 kept the stopwatch for part of their epochs only: the missing ones
    # are extrapolated from the mean, as in the reported table.
    missing = (until_epoch + 1) - len(counted)
    return (sum(counted) + max(missing, 0) * mean_seconds) / 3600, len(counted)


def main():
    print(f"Criterio: {PATIENCE} evaluaciones sin superar el maximo, no antes de la "
          f"epoca {MIN_EPOCH}\n")
    total_loop = total_wall = 0.0
    for variant in VARIANTS:
        scores = evaluation_scores(variant)
        stop, kept_epoch, kept_score = stopping_epoch(scores)
        epochs, reported_hours = REPORTED[variant]
        stop_epoch = epochs - 1 if stop is None else stop
        hours, timed = loop_hours(variant, stop_epoch)
        wall = reported_hours * (stop_epoch + 1) / epochs
        total_loop += hours
        total_wall += wall
        print(
            f"{variant.upper()}: parada en la epoca {stop_epoch} "
            f"(ejecutadas {epochs}) · punto de control conservado: epoca {kept_epoch} "
            f"con {kept_score:.4f}"
        )
        print(
            f"     bucle {hours:.1f} h ({timed} epocas cronometradas) · "
            f"tiempo total estimado {wall:.1f} h frente a {reported_hours} h\n"
        )
    reported_wall = sum(hours for _, hours in REPORTED.values())
    print(f"Total bajo criterio uniforme: bucle {total_loop:.1f} h · "
          f"total {total_wall:.1f} h (frente a {reported_wall:.1f} h realmente empleadas)")


if __name__ == "__main__":
    main()
