#!/usr/bin/env python3
"""Final evaluation of a frozen checkpoint on a block of seeds never consulted before.

Finding C1 of the external review is that the 50 conditions 100000-100049 both
choose the checkpoint and produce the reported mean, its interval and its
p-value. There is no independent measurement. This script produces one: the
selected checkpoint of a variant, evaluated once, on the disjoint block
200000-200199, under the protocol written down beforehand in
``memoria/preregistro_prueba_final.md``.

Nothing is retrained and nothing is re-selected. The checkpoint comes in frozen
and the seeds are read from the preregistration, not from a result.

Two things it does beyond ``eval.py`` of the upstream repository:

  * It overrides the evaluation block of the effective Hydra config, so the
    training runs stay untouched and the seeds used here are explicit.
  * It synchronises the diffusion noise across variants (common random numbers).
    Every call to the policy is preceded by a seed derived from the chunk and
    step indices, which are identical for the five variants because the batch
    shape, the number of chunks and the number of calls per chunk are identical.
    Two policies facing the same initial condition then differ only by their
    weights, which is what the paired Wilcoxon of the memoir assumes. This is
    finding M4.

It must run in WSL under the ``robodiff`` environment, the one that produced the
runs: the checkpoints were written by torch 1.12.1 and the simulator behaviour
must match the one the training rollouts used.

Usage (from WSL, one variant per process):
    conda activate robodiff
    python diffuser/scripts/evaluar_bloque_test.py --variante v0
    python diffuser/scripts/evaluar_bloque_test.py --variante v0 \
        --n-test 50 --test-start-seed 100000 --etiqueta cordura

The driver ``diffuser/scripts/evaluar_bloque_test.sh`` runs the five variants in
order and is the intended entry point. Output goes to
``logs_entrenamiento/prueba_final/<etiqueta>_<variante>.json``.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "logs_entrenamiento" / "prueba_final"

# Read by huggingface_hub on import, so it has to be set before timm is pulled
# in by the encoder. See neutralizar_descarga in memoria_gpu.py.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# The working copy where training actually ran; it is not versioned, so its path
# is a parameter and not an assumption baked into the script.
REPO_POR_DEFECTO = Path.home() / "tfm" / "diffusion_policy"

# Selected checkpoint of each variant, frozen before this evaluation exists.
PUNTOS_CONTROL = {
    "v0": ("ResNet-18 desde cero", "epoch=0350-test_mean_score=0.865.ckpt"),
    "v1": ("ResNet-18 congelada", "epoch=0150-test_mean_score=0.668.ckpt"),
    "v2": ("ResNet-18 con ajuste fino", "epoch=0150-test_mean_score=0.648.ckpt"),
    "v3": ("DINOv2 ViT-S/14 congelada", "epoch=0100-test_mean_score=0.622.ckpt"),
    "v4": ("CLIP ViT-B/16 congelada", "epoch=0100-test_mean_score=0.535.ckpt"),
}

# Preregistered block. Disjoint from the demonstrations (seeds 0-205) and from
# the selection set (100000-100049).
N_TEST = 200
TEST_START_SEED = 200000
BASE_SEED_DIFUSION = 20260827
N_ENVS = 8

SEMILLAS_DEMOSTRACIONES = range(0, 206)
SEMILLAS_SELECCION = range(100000, 100050)


def controlador():
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return salida.stdout.strip().splitlines()[0]
    except Exception:  # the driver string is descriptive, not load-bearing
        return "desconocido"


def sha256(path, bloque=1 << 22):
    resumen = hashlib.sha256()
    with open(path, "rb") as handle:
        for trozo in iter(lambda: handle.read(bloque), b""):
            resumen.update(trozo)
    return resumen.hexdigest()


def neutralizar_descarga(cfg):
    """Keep the encoder from fetching its initial weights over the network.

    Identical in purpose to the helper of ``diffuser/scripts/memoria_gpu.py``:
    the timm factories of the training copy fix ``pretrained=True`` in their own
    signature, so instantiating one blocks the process on a download. The
    checkpoint restores the trained weights immediately afterwards, so the
    download is pointless. The training tree is not modified; it is not
    versioned, and an evaluation script has no business editing it.
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


def cargar_politica(repo, variante, salida):
    """Rebuild the workspace from the effective config and restore the checkpoint."""
    import hydra
    import torch
    from omegaconf import OmegaConf

    run_dir = repo / "data" / "outputs" / "encoder_exp" / f"{variante}_seed42"
    cfg_path = run_dir / ".hydra" / "config.yaml"
    assert cfg_path.is_file(), f"no existe la configuracion efectiva de {variante}: {cfg_path}"
    cfg = OmegaConf.load(cfg_path)
    neutralizar_descarga(cfg)

    ckpt = run_dir / "checkpoints" / PUNTOS_CONTROL[variante][1]
    assert ckpt.is_file(), f"falta el punto de control seleccionado de {variante}: {ckpt}"

    clase = hydra.utils.get_class(cfg._target_)
    workspace = clase(cfg, output_dir=str(salida))
    # The optimiser state is a third of the file and plays no part in a rollout.
    workspace.load_checkpoint(path=ckpt, exclude_keys=("optimizer",), map_location="cpu")

    politica = workspace.ema_model if cfg.training.use_ema else workspace.model
    assert politica is not None, f"{variante}: el checkpoint no trae pesos EMA"
    workspace.optimizer = None
    politica.eval()
    politica.to(torch.device(cfg.training.device))
    return cfg, politica, ckpt


def sincronizar_ruido(politica, base_seed):
    """Common random numbers: same diffusion noise for every variant.

    The runner calls ``policy.reset()`` once per chunk of environments and
    ``predict_action`` once per control step. Seeding from that pair makes the
    noise a function of the position in the rollout and nothing else, so the
    five variants draw the same numbers at the same place. The shapes agree:
    ``torch.randn`` is called with (n_envs, horizon, action_dim) = (8, 16, 2) in
    all five, and in eval mode the crop randomiser is a centre crop, so the
    encoder consumes no randomness of its own.

    Returns a counter dict so the caller can record how many calls happened.
    """
    import torch

    estado = {"tanda": -1, "paso": 0, "llamadas": 0}
    predict_original = politica.predict_action
    reset_original = politica.reset

    def reset(*args, **kwargs):
        estado["tanda"] += 1
        estado["paso"] = 0
        return reset_original(*args, **kwargs)

    def predict_action(*args, **kwargs):
        torch.manual_seed(base_seed * 1000003 + estado["tanda"] * 1000 + estado["paso"])
        estado["paso"] += 1
        estado["llamadas"] += 1
        return predict_original(*args, **kwargs)

    politica.reset = reset
    politica.predict_action = predict_action
    return estado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variante", required=True, choices=sorted(PUNTOS_CONTROL))
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--test-start-seed", type=int, default=TEST_START_SEED)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED_DIFUSION)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--etiqueta", default="prueba",
                        help="prefijo del fichero de salida; 'cordura' para la "
                             "comprobacion sobre el conjunto de seleccion")
    parser.add_argument("--repo", type=Path, default=REPO_POR_DEFECTO)
    parser.add_argument("--forzar", action="store_true",
                        help="permite sobrescribir un resultado ya existente")
    args = parser.parse_args()

    semillas = list(range(args.test_start_seed, args.test_start_seed + args.n_test))
    if args.etiqueta == "prueba":
        # The whole point of the block is that nothing has ever looked at it.
        assert not set(semillas) & set(SEMILLAS_DEMOSTRACIONES), "el bloque toca las demostraciones"
        assert not set(semillas) & set(SEMILLAS_SELECCION), "el bloque toca el conjunto de seleccion"
        assert args.n_test == N_TEST and args.test_start_seed == TEST_START_SEED, (
            "la pasada de prueba usa el bloque preregistrado; para otra cosa, "
            "cambia --etiqueta"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destino = OUT_DIR / f"{args.etiqueta}_{args.variante}.json"
    if destino.exists() and not args.forzar:
        raise SystemExit(
            f"{destino} ya existe. La prueba final se ejecuta una sola vez; "
            "repetirla exige --forzar y una razon escrita en el preregistro."
        )

    repo = args.repo.expanduser().resolve()
    assert (repo / "diffusion_policy").is_dir(), f"no parece la copia de trabajo: {repo}"
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import hydra
    import torch
    from omegaconf import OmegaConf, open_dict

    assert torch.cuda.is_available(), "la evaluacion exige GPU; sin ella no se reporta cifra"

    salida = Path(os.environ.get("TMPDIR", "/tmp")) / f"prueba_final_{args.variante}"
    salida.mkdir(parents=True, exist_ok=True)
    cfg, politica, ckpt = cargar_politica(repo, args.variante, salida)

    # The evaluation block, and only that. Everything else -- legacy_test,
    # max_steps, fps, n_obs_steps, n_action_steps -- is inherited from the run.
    runner_cfg = cfg.task.env_runner
    with open_dict(runner_cfg):
        runner_cfg.n_train = 0
        runner_cfg.n_train_vis = 0
        runner_cfg.n_test = args.n_test
        runner_cfg.n_test_vis = 0  # no video: it costs time and adds nothing here
        runner_cfg.test_start_seed = args.test_start_seed
        runner_cfg.n_envs = args.n_envs
    if args.etiqueta == "prueba":
        # The runner pads the last chunk by repeating the first condition and
        # then discards the padding, so the metric survives a remainder. What
        # does not survive it is the alignment of the common random numbers, so
        # the preregistered pass uses an exact multiple.
        assert args.n_test % args.n_envs == 0, (
            "el ultimo grupo se rellenaria repitiendo condiciones y el ruido "
            "comun dejaria de alinearse; usa un n_test multiplo de n_envs"
        )

    estado = sincronizar_ruido(politica, args.base_seed)
    runner = hydra.utils.instantiate(runner_cfg, output_dir=str(salida))

    inicio = time.time()
    log = runner.run(politica)
    duracion = time.time() - inicio

    puntuaciones = {}
    for clave, valor in log.items():
        if clave.startswith("test/sim_max_reward_"):
            puntuaciones[int(clave.rsplit("_", 1)[1])] = float(valor)
    assert sorted(puntuaciones) == semillas, "las semillas evaluadas no son las preregistradas"
    media = sum(puntuaciones.values()) / len(puntuaciones)
    assert abs(media - float(log["test/mean_score"])) < 1e-9

    resultado = {
        "variante": args.variante.upper(),
        "codificador": PUNTOS_CONTROL[args.variante][0],
        "etiqueta": args.etiqueta,
        "punto_control": ckpt.name,
        "sha256": sha256(ckpt),
        "n_test": args.n_test,
        "test_start_seed": args.test_start_seed,
        "base_seed_difusion": args.base_seed,
        "llamadas_politica": estado["llamadas"],
        "tandas": estado["tanda"] + 1,
        "media": media,
        "segundos": round(duracion, 1),
        "runner": OmegaConf.to_container(runner_cfg, resolve=True),
        "entorno": "robodiff (WSL2 Ubuntu 24.04)",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "controlador": controlador(),
        "precision": "float32",
        "puntuaciones": {str(s): puntuaciones[s] for s in semillas},
    }
    destino.write_text(json.dumps(resultado, indent=2), encoding="utf-8")

    exitos = sum(1 for v in puntuaciones.values() if v >= 0.999)
    print(
        f"{resultado['variante']} · {args.etiqueta} · {args.n_test} condiciones desde "
        f"{args.test_start_seed}: media {media:.4f} · exito {exitos}/{args.n_test} · "
        f"{estado['llamadas']} llamadas en {duracion / 60:.1f} min"
    )
    print(f"Escrito en {destino}")


if __name__ == "__main__":
    main()
