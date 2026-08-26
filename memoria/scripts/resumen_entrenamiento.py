#!/usr/bin/env python3
"""Rebuild the per-epoch timing CSV and the run summary of an encoder-experiment run.

V0, V1 and V2 were summarised by hand. This script reproduces that work from the
logs already exported to ``logs_entrenamiento/raw/`` so that new variants (and the
seeds of phase 2) do not depend on manual bookkeeping.

    python memoria/scripts/resumen_entrenamiento.py v3 v4

Writes ``logs_entrenamiento/<variant>_seed<seed>_epocas.csv`` and merges the run
into ``logs_entrenamiento/resumen.json`` without touching the other variants.
"""

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs_entrenamiento"
RAW_DIR = LOG_DIR / "raw"
MODELS_DIR = ROOT / "diffuser" / "models"

# tqdm writes "Training epoch 3:  98%|...| 328/336 [02:06<00:02,  2.91it/s, loss=...]"
EPOCH_BAR = re.compile(
    r"Training epoch (\d+):.*?\[(?:(\d+):)?(\d+):(\d+)<"
)
TIMESTAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def epoch_seconds(variant, seed):
    """Wall-clock seconds of each epoch, from the last tqdm update of that epoch."""
    path = RAW_DIR / f"{variant}_seed{seed}.log"
    if not path.exists():
        return {}
    # tqdm redraws with carriage returns, so one physical line holds many updates.
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    elapsed = {}
    for match in EPOCH_BAR.finditer(text):
        epoch = int(match.group(1))
        hours = int(match.group(2) or 0)
        seconds = hours * 3600 + int(match.group(3)) * 60 + int(match.group(4))
        elapsed[epoch] = max(elapsed.get(epoch, 0), seconds)
    return elapsed


def write_csv(variant, seed, elapsed):
    path = LOG_DIR / f"{variant}_seed{seed}_epocas.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["epoch", "tipo", "segundos"])
        for epoch in sorted(elapsed):
            writer.writerow([epoch, "entrenamiento", float(elapsed[epoch])])
    return path


def read_logs(variant):
    path = RAW_DIR / f"{variant}_logs_json.txt.gz"
    epochs, rollouts, last = set(), [], None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            last = record
            if "epoch" in record:
                epochs.add(record["epoch"])
            if "test/mean_score" in record:
                rollouts.append([record["epoch"], record["test/mean_score"]])
    return epochs, rollouts, last


def end_timestamp(variant):
    """The run ends when its JSON log was last written; the copy loses that mtime."""
    path = RAW_DIR / f"{variant}_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("logs_json_mtime")


def start_timestamp(variant):
    path = RAW_DIR / f"{variant}_train_hydra.log"
    if not path.exists():
        return None
    match = TIMESTAMP.search(path.read_text(encoding="utf-8", errors="replace"))
    return match.group(1) if match else None


def config_name(variant):
    path = RAW_DIR / f"{variant}_hydra_config.yaml"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return None


def num_epochs_cfg(variant):
    path = RAW_DIR / f"{variant}_hydra_overrides.yaml"
    if not path.exists():
        return None
    match = re.search(r"training\.num_epochs=(\d+)", path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def count_checkpoints(variant):
    folder = MODELS_DIR / variant.upper()
    return len(list(folder.glob("*.ckpt"))) if folder.is_dir() else None


def summarise(variant, seed):
    elapsed = epoch_seconds(variant, seed)
    csv_path = write_csv(variant, seed, elapsed)
    epochs, rollouts, last = read_logs(variant)
    total = float(sum(elapsed.values()))
    return csv_path, {
        "variant": config_name(variant),
        "num_epochs_cfg": num_epochs_cfg(variant),
        "max_epoch": max(epochs) + 1 if epochs else None,
        "epochs_logged": len(epochs),
        "last_step": last.get("global_step") if last else None,
        "train_epochs_timed": len(elapsed),
        "total_train_secs": total,
        "avg_epoch": total / len(elapsed) if elapsed else None,
        "last_train_loss": last.get("train_loss") if last else None,
        "last_val_loss": last.get("val_loss") if last else None,
        "n_rollouts": len(rollouts),
        "first_score": rollouts[0] if rollouts else None,
        "last_score": rollouts[-1] if rollouts else None,
        "best": max(rollouts, key=lambda item: item[1]) if rollouts else None,
        "n_ckpt": count_checkpoints(variant),
        "start": start_timestamp(variant),
        "end": end_timestamp(variant),
    }


def hms(seconds):
    seconds = int(seconds)
    hours, minutes, seconds = seconds // 3600, seconds % 3600 // 60, seconds % 60
    return f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds}s"


def report(variant, entry):
    lines = [f"--- {variant.upper()}: {entry['variant']} ---"]
    lines.append(f"  Run dir            : data/outputs/encoder_exp/{variant}_seed42")
    lines.append(f"  Inicio (train.log) : {entry['start']}")
    lines.append(f"  Fin (logs.json.txt): {entry['end']}")
    lines.append(
        f"  Epocas registradas : {entry['epochs_logged']} "
        f"(indice maximo {entry['max_epoch'] - 1})"
    )
    lines.append(f"  Presupuesto (cfg)  : {entry['num_epochs_cfg']} epocas")
    lines.append(f"  Global steps       : {entry['last_step']}")
    lines.append(f"  Epocas cronometradas (tqdm): {entry['train_epochs_timed']}")
    lines.append(f"  Tiempo acumulado entrenamiento: {hms(entry['total_train_secs'])}")
    lines.append(f"  Tiempo medio por epoca        : {hms(entry['avg_epoch'])}")
    lines.append(f"  Ultimo train_loss  : {entry['last_train_loss']:.5f}")
    val = entry["last_val_loss"]
    lines.append(f"  Ultimo val_loss    : {val:.5f}" if val else "  Ultimo val_loss    : -")
    lines.append(f"  Rollouts evaluados : {entry['n_rollouts']}")
    lines.append(
        f"  Mejor score        : epoch {entry['best'][0]} -> "
        f"test/mean_score={entry['best'][1]:.4f}"
    )
    lines.append(
        f"  Ultimo score       : epoch {entry['last_score'][0]} -> {entry['last_score'][1]:.4f}"
    )
    lines.append(f"  Checkpoints (.ckpt): {entry['n_ckpt']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+", help="v0 v1 v2 v3 v4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary_path = LOG_DIR / "resumen.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for variant in args.variants:
        csv_path, entry = summarise(variant, args.seed)
        previous = summary.get(variant, {})
        entry = {
            key: previous.get(key) if value is None else value
            for key, value in entry.items()
        }
        summary[variant] = entry
        print(f"{csv_path.name}: {entry['train_epochs_timed']} epocas cronometradas")
        print(report(variant, entry))
        print()

    summary = {key: summary[key] for key in sorted(summary)}
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"actualizado {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
