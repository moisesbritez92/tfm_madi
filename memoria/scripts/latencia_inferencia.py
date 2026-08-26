#!/usr/bin/env python3
"""Inference latency of the five encoder variants of the Push-T experiment.

Section 3.6 of the memoir promises four cost indicators. Two of them (parameter
counts and optimisation time per epoch) come from the training logs; the latency
of a policy call does not, because no rollout log stores it. This script measures
it directly on the selected checkpoints.

Three magnitudes are reported per variant, and the split is the point of the
script: ``predict_action`` runs the visual encoder once and the conditional UNet
``num_inference_steps`` times, so the total call is dominated by a network that
is identical across the five variants. Reporting only the total would hide the
difference between encoders, which is the object of the study.

  * ``llamada``      -- full ``policy.predict_action(obs_dict)``.
  * ``codificador``  -- ``policy.obs_encoder(this_nobs)`` alone.
  * ``por_accion``   -- full call divided by ``n_action_steps``, the quantity that
                       bounds the achievable closed-loop control rate.

The measurement is round-robin: every variant contributes a few repetitions per
round and the rounds alternate, so the thermal drift of a laptop GPU affects the
five variants alike instead of penalising whichever was measured last. Only one
policy sits on the GPU at a time; the other four wait on host memory.

Environment: this runs under the Windows inference environment
(``.venv_diffuser_infer``, torch 2.6.0+cu124), not under the WSL environment used
for training (torch 1.12.1+cu116). The CSV records which one was used, and the
memoir declares it in the table note.

Usage:
    .venv_diffuser_infer/Scripts/python.exe memoria/scripts/latencia_inferencia.py
"""

import argparse
import csv
import gc
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIFFUSER_DIR = ROOT / "diffuser"
OUT_DIR = ROOT / "memoria" / "datos"

sys.path.insert(0, str(DIFFUSER_DIR))

import torch  # noqa: E402  (the diffuser path must be set before importing)

from v0_inference_utils import load_policy_bundle  # noqa: E402
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv  # noqa: E402

# Selected checkpoint of each variant, the one whose score the memoir reports.
VARIANTS = {
    "V0": ("ResNet-18 desde cero", "V0", "epoch=0350-test_mean_score=0.865.ckpt"),
    "V1": ("ResNet-18 congelada", "V1", "epoch=0150-test_mean_score=0.668.ckpt"),
    "V2": ("ResNet-18 con ajuste fino", "V2", "epoch=0150-test_mean_score=0.648.ckpt"),
    "V3": ("DINOv2 ViT-S/14 congelada", "V3", "epoch=0100-test_mean_score=0.622.ckpt"),
    "V4": ("CLIP ViT-B/16 congelada", "V4", "epoch=0100-test_mean_score=0.535.ckpt"),
}

BATCH_SIZES = (1, 8)
ROUNDS = 10
REPS_PER_ROUND = 5
WARMUP = 20
OBS_SEED = 100000

# cuDNN autotuning is left off so that the first rounds are not systematically
# slower than the last ones; the value is reported in the CSV.
CUDNN_BENCHMARK = False


def build_observation(n_obs_steps):
    """Return one real Push-T observation window, on CPU.

    Synthetic tensors would run through the same kernels, but not necessarily
    over the same value range: the normaliser of each policy was fitted on the
    demonstrations, and a batch of zeros would leave that range. The frames come
    from the simulator itself, reset with an evaluation seed.
    """
    env = PushTImageEnv()
    env.seed(OBS_SEED)
    frames = [env.reset()]
    while len(frames) < n_obs_steps:
        frames.append(env.step(env.action_space.sample())[0])

    window = {}
    for key in frames[0]:
        stacked = np.stack([np.asarray(f[key], dtype=np.float32) for f in frames])
        window[key] = torch.from_numpy(stacked).unsqueeze(0)  # (1, To, ...)

    assert window["image"].shape[1:] == (n_obs_steps, 3, 96, 96), window["image"].shape
    image = window["image"]
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0, "imagen fuera de [0,1]"
    pos = window["agent_pos"]
    assert 0.0 <= float(pos.min()) and float(pos.max()) <= 512.0, "agent_pos fuera del espacio"
    return window


def to_batch(window, batch_size, device):
    """Replicate the observation window into a batch on the target device."""
    return {
        key: value.repeat(batch_size, *([1] * (value.ndim - 1))).to(device)
        for key, value in window.items()
    }


def encoder_input(policy, obs_dict):
    """Reproduce the tensor that ``predict_action`` feeds to the obs encoder."""
    nobs = policy.normalizer.normalize(obs_dict)
    to = policy.n_obs_steps
    return dict_apply(nobs, lambda x: x[:, :to, ...].reshape(-1, *x.shape[2:]))


def timed(fn, device):
    """One synchronised repetition, in seconds."""
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def summarise(samples_s, divisor=1):
    """Median and dispersion in milliseconds; the mean is not reported on purpose."""
    values = np.asarray(samples_s, dtype=np.float64) * 1e3 / divisor
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    p5, p95 = np.percentile(values, [5, 95])
    return {
        "mediana_ms": round(float(median), 3),
        "q1_ms": round(float(q1), 3),
        "q3_ms": round(float(q3), 3),
        "iqr_ms": round(float(q3 - q1), 3),
        "p5_ms": round(float(p5), 3),
        "p95_ms": round(float(p95), 3),
    }


def driver_version():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:  # the driver string is descriptive, not load-bearing
        return "desconocido"


def load_policies(device):
    """Load the five selected checkpoints onto host memory.

    ``load_policy_bundle`` also builds the non-EMA model, which is never used at
    inference time and would double the host footprint of five policies; it is
    dropped as soon as the bundle is built.
    """
    policies = {}
    for name, (_, folder, filename) in VARIANTS.items():
        path = ROOT / "diffuser" / "models" / folder / filename
        assert path.is_file(), f"falta el punto de control de {name}: {path}"
        started = time.perf_counter()
        bundle = load_policy_bundle(path, device="cpu")
        workspace = bundle["workspace"]
        policy = bundle["policy"]
        if workspace.model is not policy:
            workspace.model = None
        gc.collect()

        cfg = bundle["cfg"]
        encoder_cfg = cfg.policy.obs_encoder
        resize = encoder_cfg.get("resize_shape")
        crop = encoder_cfg.get("crop_shape")
        # The encoder resolution is what the backbone actually sees: the resize
        # if there is one, the crop if there is one, the native 96 px otherwise.
        if crop is not None:
            encoder_px = int(crop[0])
        elif resize is not None:
            encoder_px = int(resize[0])
        else:
            encoder_px = int(cfg.task.shape_meta.obs.image.shape[1])

        assert int(cfg.n_obs_steps) == 2, f"{name}: n_obs_steps inesperado"
        assert int(cfg.n_action_steps) == 8, f"{name}: n_action_steps inesperado"
        assert int(policy.num_inference_steps) == 100, f"{name}: pasos de difusion inesperados"

        policies[name] = {
            "policy": policy,
            "workspace": workspace,
            "checkpoint": filename,
            "lote_entrenamiento": int(cfg.dataloader.batch_size),
            "px_entrada": int(cfg.task.shape_meta.obs.image.shape[1]),
            "px_codificador": encoder_px,
            "pasos_difusion": int(policy.num_inference_steps),
            "n_obs_steps": int(cfg.n_obs_steps),
            "n_action_steps": int(policy.n_action_steps),
        }
        print(f"{name}: cargado en {time.perf_counter() - started:.1f} s "
              f"({encoder_px} px, lote de entrenamiento {policies[name]['lote_entrenamiento']})")
    del device
    return policies


def measure(policies, device, rounds, reps, warmup):
    """Round-robin measurement over the five variants and both batch sizes."""
    samples = {(name, batch, magnitude): []
               for name in policies
               for batch in BATCH_SIZES
               for magnitude in ("llamada", "codificador")}

    window = build_observation(next(iter(policies.values()))["n_obs_steps"])

    for batch in BATCH_SIZES:
        print(f"\n--- lote {batch} ---")
        for round_idx in range(rounds):
            for name, entry in policies.items():
                policy = entry["policy"]
                policy.to(device)
                policy.eval()
                obs = to_batch(window, batch, device)
                with torch.no_grad():
                    enc_in = encoder_input(policy, obs)
                    call = lambda: policy.predict_action(obs)  # noqa: E731
                    enc = lambda: policy.obs_encoder(enc_in)  # noqa: E731

                    # A full warm-up before the first round, a short one after
                    # every host-to-device transfer.
                    for _ in range(warmup if round_idx == 0 else 3):
                        call()
                        enc()
                    torch.cuda.synchronize(device)

                    for _ in range(reps):
                        samples[(name, batch, "llamada")].append(timed(call, device))
                        samples[(name, batch, "codificador")].append(timed(enc, device))

                policy.to("cpu")
                del obs, enc_in
                torch.cuda.empty_cache()
            print(f"  ronda {round_idx + 1}/{rounds} completada")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rondas", type=int, default=ROUNDS)
    parser.add_argument("--repeticiones", type=int, default=REPS_PER_ROUND)
    parser.add_argument("--calentamiento", type=int, default=WARMUP)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "la medida exige GPU; sin ella no se reporta cifra"
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK

    n_reps = args.rondas * args.repeticiones
    assert n_reps >= 50, f"se exigen al menos 50 repeticiones cronometradas, hay {n_reps}"

    contexto = {
        "entorno": ".venv_diffuser_infer (Windows)",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "controlador": driver_version(),
        "python": platform.python_version(),
        "precision": "float32",
        "cudnn_benchmark": str(CUDNN_BENCHMARK),
    }
    print(" · ".join(f"{k}={v}" for k, v in contexto.items()))

    policies = load_policies(device)
    samples = measure(policies, device, args.rondas, args.repeticiones, args.calentamiento)

    rows = []
    for name, entry in policies.items():
        for batch in BATCH_SIZES:
            total = samples[(name, batch, "llamada")]
            encoder = samples[(name, batch, "codificador")]
            assert len(total) == n_reps and len(encoder) == n_reps
            magnitudes = (
                ("llamada", summarise(total), n_reps),
                ("codificador", summarise(encoder), n_reps),
                ("por_accion", summarise(total, entry["n_action_steps"]), n_reps),
            )
            for magnitude, stats, reps in magnitudes:
                rows.append({
                    "variante": name,
                    "codificador": VARIANTS[name][0],
                    "punto_control": entry["checkpoint"],
                    "magnitud": magnitude,
                    "lote": batch,
                    "px_entrada": entry["px_entrada"],
                    "px_codificador": entry["px_codificador"],
                    "pasos_difusion": entry["pasos_difusion"],
                    "n_obs_steps": entry["n_obs_steps"],
                    "n_action_steps": entry["n_action_steps"],
                    "lote_entrenamiento": entry["lote_entrenamiento"],
                    **stats,
                    "n_repeticiones": reps,
                    "n_calentamiento": args.calentamiento,
                    "n_rondas": args.rondas,
                    **contexto,
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destino = OUT_DIR / "latencia_inferencia.csv"
    with destino.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print()
    for row in rows:
        if row["lote"] == 1:
            print(f"{row['variante']} lote 1 {row['magnitud']:>12}: "
                  f"{row['mediana_ms']:9.3f} ms  IQR {row['iqr_ms']:.3f}")
    print(f"\nResultados escritos en {destino}")


if __name__ == "__main__":
    main()
