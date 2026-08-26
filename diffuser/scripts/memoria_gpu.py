#!/usr/bin/env python3
"""Peak GPU memory of the five encoder variants, in training and in inference.

Section 3.6 of the memoir promises the peak *reserved* graphics memory during
training. That figure is not inside a checkpoint: it only exists while an
optimisation step is running, because the gradients, the AdamW moments and the
EMA copy are what fill the 8 GB of the card. This script therefore rebuilds the
training workspace from the effective Hydra config of each run, restores the
selected checkpoint including the optimiser state, and replays a handful of
complete steps with that variant's own batch size.

It must run in WSL under the ``robodiff`` environment (torch 1.12.1+cu116), the
one that produced the runs reported in ``tab:coste``. The caching allocator and
the cuDNN workspaces changed between torch 1.12 and the 2.6 of the Windows
inference environment, so a figure taken there would not describe those runs and
could not justify the decisions of sections 3.3 and 3.4.

Two magnitudes are reported and must not be confused:

  * ``max_memory_reserved``  -- what the caching allocator holds from the driver.
    This is the promised indicator and the one that decides whether a variant
    fits in the card.
  * ``max_memory_allocated`` -- what the live tensors occupy. Always smaller.

Two modes are reported and must not be confused either: ``entrenamiento`` (the
promised one) and ``inferencia`` (what a deployment would need). Each mode of
each variant runs in its own process, because the allocator of one measurement
would contaminate the peak of the next.

Usage (from WSL, one variant and mode at a time):
    conda activate robodiff
    python diffuser/scripts/memoria_gpu.py --variante v0 --modo entrenamiento
    python diffuser/scripts/memoria_gpu.py --variante v0 --modo inferencia --lote 1

The driver ``diffuser/scripts/medir_memoria_gpu.sh`` runs the fifteen
combinations in order and is the intended entry point. Output is appended to
``memoria/datos/memoria_gpu.csv``.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "memoria" / "datos"
DESTINO = OUT_DIR / "memoria_gpu.csv"

# Read by huggingface_hub when it is imported, so it has to be set before the
# encoder pulls timm in. See neutralizar_descarga for why.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# The working copy where training actually ran; it is not versioned, so its path
# is a parameter and not an assumption baked into the script.
REPO_POR_DEFECTO = Path.home() / "tfm" / "diffusion_policy"

# Selected checkpoint of each variant, the one whose score the memoir reports.
PUNTOS_CONTROL = {
    "v0": ("ResNet-18 desde cero", "epoch=0350-test_mean_score=0.865.ckpt"),
    "v1": ("ResNet-18 congelada", "epoch=0150-test_mean_score=0.668.ckpt"),
    "v2": ("ResNet-18 con ajuste fino", "epoch=0150-test_mean_score=0.648.ckpt"),
    "v3": ("DINOv2 ViT-S/14 congelada", "epoch=0100-test_mean_score=0.622.ckpt"),
    "v4": ("CLIP ViT-B/16 congelada", "epoch=0100-test_mean_score=0.535.ckpt"),
}

PASOS_ENTRENAMIENTO = 12  # at least ten, and a multiple of the largest accumulation
LOTES_INFERENCIA = (1, 8)
OBS_SEED = 100000

CAMPOS = [
    "variante", "codificador", "punto_control", "modo", "lote", "acumulacion_gradiente",
    "lote_efectivo", "px_codificador", "pasos_difusion", "codificador_congelado",
    "pico_reservado_gb", "pico_asignado_gb", "capacidad_gb", "ocupacion_pct",
    "n_pasos", "entorno", "torch", "cuda", "gpu", "controlador", "precision",
]

GIB = 2 ** 30


def controlador():
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return salida.stdout.strip().splitlines()[0]
    except Exception:  # the driver string is descriptive, not load-bearing
        return "desconocido"


def px_codificador(cfg):
    """Resolution the visual backbone actually sees."""
    encoder = cfg.policy.obs_encoder
    crop = encoder.get("crop_shape")
    resize = encoder.get("resize_shape")
    if crop is not None:
        return int(crop[0])
    if resize is not None:
        return int(resize[0])
    return int(cfg.task.shape_meta.obs.image.shape[1])


def observacion(n_obs_steps, device):
    """One real Push-T observation window, batched later.

    The same construction as ``memoria/scripts/latencia_inferencia.py``: values
    taken from the simulator, not synthesised, so they fall in the range the
    normaliser of each policy was fitted on. The helper is duplicated instead of
    imported because that script lives in the Windows inference environment.
    """
    import numpy as np
    import torch
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

    env = PushTImageEnv()
    env.seed(OBS_SEED)
    marcos = [env.reset()]
    while len(marcos) < n_obs_steps:
        marcos.append(env.step(env.action_space.sample())[0])

    ventana = {}
    for clave in marcos[0]:
        apilado = np.stack([np.asarray(m[clave], dtype=np.float32) for m in marcos])
        ventana[clave] = torch.from_numpy(apilado).unsqueeze(0).to(device)

    imagen = ventana["image"]
    assert imagen.shape[1:] == (n_obs_steps, 3, 96, 96), imagen.shape
    assert 0.0 <= float(imagen.min()) and float(imagen.max()) <= 1.0
    return ventana


def neutralizar_descarga(cfg):
    """Keep the encoder from fetching its initial weights over the network.

    Instantiating a pretrained backbone downloads its weights, and that request
    can block indefinitely. The download is pointless here: the checkpoint
    restores the trained weights immediately afterwards, and the values a tensor
    holds do not change how much memory it occupies.

    The two families need different treatment. The torchvision path takes the
    argument from the config, so its weights are simply requested as absent, the
    same treatment that ``diffuser/v0_inference_utils.py::load_policy_bundle``
    applies on Windows. The ``timm`` factories of the training copy fix
    ``pretrained=True`` in their own signature, unlike the Windows inference
    copy, so the flag cannot travel through the config; ``create_model`` is
    wrapped instead. The training tree is not modified: it is not versioned, and
    a measurement script has no business editing it.
    """
    import timm
    from omegaconf import open_dict

    if not getattr(timm.create_model, "_sin_descarga", False):
        original = timm.create_model

        def sin_descarga(*args, **kwargs):
            kwargs["pretrained"] = False
            return original(*args, **kwargs)

        sin_descarga._sin_descarga = True
        timm.create_model = sin_descarga

    rgb = cfg.policy.obs_encoder.get("rgb_model", None)
    if rgb is None:
        return
    if str(rgb.get("_target_", "")) == "diffusion_policy.model.vision.model_getter.get_resnet":
        with open_dict(rgb):
            rgb.weights = None


def construir(repo, variante, cargar_optimizador):
    """Rebuild the training workspace of a variant from its effective config."""
    import hydra
    import torch
    from omegaconf import OmegaConf

    cfg_path = repo / "data" / "outputs" / "encoder_exp" / f"{variante}_seed42" / ".hydra" / "config.yaml"
    assert cfg_path.is_file(), f"no existe la configuracion efectiva de {variante}: {cfg_path}"
    cfg = OmegaConf.load(cfg_path)
    neutralizar_descarga(cfg)

    ckpt = repo / "data" / "outputs" / "encoder_exp" / f"{variante}_seed42" / "checkpoints" / PUNTOS_CONTROL[variante][1]
    assert ckpt.is_file(), f"falta el punto de control seleccionado de {variante}: {ckpt}"

    clase = hydra.utils.get_class(cfg._target_)
    salida = Path(os.environ.get("TMPDIR", "/tmp")) / f"memoria_gpu_{variante}"
    salida.mkdir(parents=True, exist_ok=True)
    workspace = clase(cfg, output_dir=str(salida))

    excluir = None if cargar_optimizador else ("optimizer",)
    payload = workspace.load_checkpoint(path=ckpt, exclude_keys=excluir, map_location="cpu")
    if cargar_optimizador:
        assert "optimizer" in payload["state_dicts"], (
            f"{variante}: el punto de control no guarda el estado del optimizador; "
            "el pico medido no incluiria los momentos de AdamW"
        )
    del payload
    torch.cuda.empty_cache()
    return cfg, workspace, ckpt


def medir_entrenamiento(repo, variante, n_pasos):
    """Replay complete optimisation steps and read the peak they produce."""
    import hydra
    import torch
    from torch.utils.data import DataLoader
    from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
    from diffusion_policy.model.common.lr_scheduler import get_scheduler

    cfg, workspace, ckpt = construir(repo, variante, cargar_optimizador=True)
    device = torch.device(cfg.training.device)
    acumulacion = int(cfg.training.gradient_accumulate_every)
    assert n_pasos % acumulacion == 0, "los pasos deben cubrir acumulaciones completas"

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    # The real batch size of the variant, with a single worker: the number of
    # workers changes host memory, not the peak on the card.
    cargador = DataLoader(
        dataset, batch_size=int(cfg.dataloader.batch_size), shuffle=True, num_workers=0
    )
    normalizador = dataset.get_normalizer()
    workspace.model.set_normalizer(normalizador)
    if cfg.training.use_ema:
        workspace.ema_model.set_normalizer(normalizador)

    planificador = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=workspace.optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=(len(cargador) * cfg.training.num_epochs) // acumulacion,
        last_epoch=workspace.global_step - 1,
    )
    ema = hydra.utils.instantiate(cfg.ema, model=workspace.ema_model) if cfg.training.use_ema else None

    workspace.model.to(device)
    if workspace.ema_model is not None:
        workspace.ema_model.to(device)
    optimizer_to(workspace.optimizer, device)

    if cfg.training.freeze_encoder:
        workspace.model.obs_encoder.eval()
        workspace.model.obs_encoder.requires_grad_(False)

    # The batches stay on host memory and travel one at a time, as the training
    # loop does. Preloading them onto the card would inflate the peak with data
    # that a real epoch never holds at once.
    lotes = []
    for lote in cargador:
        lotes.append(lote)
        if len(lotes) == n_pasos:
            break
    assert len(lotes) == n_pasos, "el conjunto no aporta suficientes lotes"

    # The loop keeps the first batch resident for the periodic sampling, so it
    # is part of the real footprint and is transferred once, outside the timing.
    lote_muestreo = dict_apply(lotes[0], lambda x: x.to(device, non_blocking=True))

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # The sequence reproduces the training loop of the workspace, including the
    # gradient accumulation and the norm clipping where the run configured them.
    paso_global = int(workspace.global_step)
    lote_gpu = None
    for indice, lote in enumerate(lotes):
        lote_gpu = lote_muestreo if indice == 0 else dict_apply(
            lote, lambda x: x.to(device, non_blocking=True))
        perdida_bruta = workspace.model.compute_loss(lote_gpu)
        assert torch.isfinite(perdida_bruta), f"{variante}: perdida no finita"
        (perdida_bruta / acumulacion).backward()
        if cfg.training.get("grad_norm_clip", None) is not None:
            torch.nn.utils.clip_grad_norm_(
                workspace.model.parameters(), cfg.training.grad_norm_clip)
        if paso_global % acumulacion == 0:
            workspace.optimizer.step()
            workspace.optimizer.zero_grad()
            planificador.step()
        if ema is not None:
            ema.step(workspace.model)
        paso_global += 1
    torch.cuda.synchronize(device)

    return cfg, ckpt, [{
        "modo": "entrenamiento",
        "lote": int(cfg.dataloader.batch_size),
        "acumulacion_gradiente": acumulacion,
        "lote_efectivo": int(cfg.dataloader.batch_size) * acumulacion,
        "pico_reservado_gb": round(torch.cuda.max_memory_reserved(device) / GIB, 3),
        "pico_asignado_gb": round(torch.cuda.max_memory_allocated(device) / GIB, 3),
        "n_pasos": n_pasos,
    }]


def medir_inferencia(repo, variante, lote):
    """Peak of a policy call, with the EMA weights and no optimiser state.

    One batch size per process. The caching allocator does not return a pool it
    has already grown, so measuring batch 8 after batch 1 in the same process
    would report the first pool instead of the second one's demand.
    """
    import torch

    cfg, workspace, ckpt = construir(repo, variante, cargar_optimizador=False)
    device = torch.device(cfg.training.device)

    politica = workspace.ema_model if cfg.training.use_ema else workspace.model
    # A deployment loads one set of weights, not two: the unused copy and the
    # optimiser are dropped so the figure describes what a robot would need.
    workspace.optimizer = None
    if politica is not workspace.model:
        workspace.model = None
    politica.eval()
    politica.to(device)

    ventana = observacion(int(cfg.n_obs_steps), device)
    obs = {k: v.repeat(lote, *([1] * (v.ndim - 1))) for k, v in ventana.items()}
    with torch.no_grad():
        politica.predict_action(obs)  # warm-up: allocates the workspaces
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        resultado = politica.predict_action(obs)
        torch.cuda.synchronize(device)
    assert resultado["action"].shape[:2] == (lote, int(politica.n_action_steps))

    return cfg, ckpt, [{
        "modo": "inferencia",
        "lote": lote,
        "acumulacion_gradiente": "",
        "lote_efectivo": "",
        "pico_reservado_gb": round(torch.cuda.max_memory_reserved(device) / GIB, 3),
        "pico_asignado_gb": round(torch.cuda.max_memory_allocated(device) / GIB, 3),
        "n_pasos": 1,
    }]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variante", required=True, choices=sorted(PUNTOS_CONTROL))
    parser.add_argument("--modo", required=True, choices=("entrenamiento", "inferencia"))
    parser.add_argument("--lote", type=int, choices=LOTES_INFERENCIA, default=1,
                        help="tamano de lote del modo inferencia; el de entrenamiento "
                             "lo fija la configuracion de la variante")
    parser.add_argument("--repo", type=Path, default=REPO_POR_DEFECTO)
    parser.add_argument("--pasos", type=int, default=PASOS_ENTRENAMIENTO)
    parser.add_argument("--reiniciar", action="store_true",
                        help="borra el CSV antes de escribir la primera fila")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    assert (repo / "diffusion_policy").is_dir(), f"no parece la copia de trabajo: {repo}"
    # The dataset path in the config is relative to the repository root.
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import torch
    assert torch.cuda.is_available(), "la medida exige GPU; sin ella no se reporta cifra"
    assert args.pasos >= 10, "el protocolo exige al menos diez pasos completos"

    if args.modo == "entrenamiento":
        cfg, ckpt, filas = medir_entrenamiento(repo, args.variante, args.pasos)
    else:
        cfg, ckpt, filas = medir_inferencia(repo, args.variante, args.lote)

    capacidad = torch.cuda.get_device_properties(0).total_memory / GIB
    contexto = {
        "variante": args.variante.upper(),
        "codificador": PUNTOS_CONTROL[args.variante][0],
        "punto_control": ckpt.name,
        "px_codificador": px_codificador(cfg),
        "pasos_difusion": int(cfg.policy.num_inference_steps),
        "codificador_congelado": str(bool(cfg.training.freeze_encoder)),
        "capacidad_gb": round(capacidad, 3),
        "entorno": "robodiff (WSL2 Ubuntu 24.04)",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "controlador": controlador(),
        "precision": "float32",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.reiniciar and DESTINO.exists():
        DESTINO.unlink()
    nuevo = not DESTINO.exists()
    with DESTINO.open("a", newline="", encoding="utf-8") as handle:
        escritor = csv.DictWriter(handle, fieldnames=CAMPOS, lineterminator="\n")
        if nuevo:
            escritor.writeheader()
        for fila in filas:
            fila = {**contexto, **fila}
            fila["ocupacion_pct"] = round(100 * fila["pico_reservado_gb"] / capacidad, 1)
            escritor.writerow({campo: fila[campo] for campo in CAMPOS})
            print(f"{fila['variante']} {fila['modo']:<14} lote {fila['lote']:>2}: "
                  f"reservado {fila['pico_reservado_gb']:.3f} GiB "
                  f"({fila['ocupacion_pct']:.1f} % de {capacidad:.2f} GiB) · "
                  f"asignado {fila['pico_asignado_gb']:.3f} GiB")

    print(f"Fila(s) anadidas a {DESTINO}")


if __name__ == "__main__":
    main()
