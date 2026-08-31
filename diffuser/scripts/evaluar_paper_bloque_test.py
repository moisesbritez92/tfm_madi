#!/usr/bin/env python3
"""Evaluation of the published Diffusion Policy checkpoint on the final block.

Sibling of ``evaluar_bloque_test.py``, which is left untouched on purpose: that
script produced the five frozen JSON files of the final test and editing it would
break the chain of custody. Everything that can be shared is imported from it --
the common-random-numbers hook, the SHA-256 helper, the driver string and the
preregistered constants -- so the two paths differ only where they must.

The protocol is written down beforehand in
``memoria/preregistro_comparacion_paper.md``. Four things differ from the sibling:

  * The config does not come from ``<run_dir>/.hydra/config.yaml``. V_Paper has no
    run directory: its config travels inside the checkpoint, in ``payload["cfg"]``.
  * ``neutralizar_descarga`` reads ``cfg.policy.obs_encoder`` without a guard, and
    the hybrid policy has no such key -- it builds its encoder inside robomimic.
    The guard used here is the one already written in
    ``diffuser/v0_inference_utils.py``.
  * The checkpoint is a path, not one of the five variant names.
  * Three protocol assertions the sibling did not need, because there the five
    variants shared one config: the evaluator block must match the one recorded in
    ``prueba_v0.json``, the horizons must match, and the diffusion noise must have
    the same shape.

Like the sibling it must run in WSL under ``robodiff`` -- torch 1.12.1, the same
environment that produced the five files it is going to be compared against. It
also needs robomimic, which that environment already has.

Usage (from WSL, one run per process):
    conda activate robodiff
    python diffuser/scripts/evaluar_paper_bloque_test.py --etiqueta prueba
    python diffuser/scripts/evaluar_paper_bloque_test.py --etiqueta ruido_b \
        --base-seed 20260831
    python diffuser/scripts/evaluar_paper_bloque_test.py --etiqueta cordura \
        --n-test 50 --test-start-seed 4300000
    python diffuser/scripts/evaluar_paper_bloque_test.py --comprobar-ruido

The driver ``diffuser/scripts/evaluar_paper_bloque_test.sh`` is the intended entry
point. Output goes to ``logs_entrenamiento/prueba_final/<etiqueta>_paper.json``.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluar_bloque_test import (  # noqa: E402  (sets HF_HUB_OFFLINE on import)
    BASE_SEED_DIFUSION,
    N_ENVS,
    N_TEST,
    OUT_DIR,
    REPO_POR_DEFECTO,
    ROOT,
    SEMILLAS_DEMOSTRACIONES,
    SEMILLAS_SELECCION,
    TEST_START_SEED,
    cargar_politica,
    controlador,
    sha256,
    sincronizar_ruido,
)

CHECKPOINT_POR_DEFECTO = (
    ROOT / "diffuser" / "models" / "V_Paper" / "epoch=0500-test_mean_score=0.884.ckpt"
)

# The block the authors used to select and report their own checkpoint. Disjoint
# from ours; it is the second gate of the preregistration, not a result.
SEMILLAS_SELECCION_PAPER = range(4300000, 4300050)
CORDURA_ESPERADA = 0.884
CORDURA_TOLERANCIA = 0.07

# Reference file for the evaluator block. The comparison is only a comparison if
# both arms ran the same evaluator.
REFERENCIA = OUT_DIR / "prueba_v0.json"
CLAVES_RUNNER = ("legacy_test", "max_steps", "n_obs_steps", "n_action_steps", "fps")


def commit():
    try:
        salida = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return salida.stdout.strip()
    except Exception:  # descriptive, not load-bearing
        return "desconocido"


def neutralizar_descarga(cfg):
    """Same purpose as the sibling helper, but tolerant of the hybrid policy.

    ``evaluar_bloque_test.neutralizar_descarga`` reaches into
    ``cfg.policy.obs_encoder`` directly. The policy of the paper has no such key:
    its encoder is built inside robomimic from a bc_rnn config, and nothing is
    downloaded there either -- the ResNet-18 of robomimic defaults to
    ``pretrained=False``. So the guard is the whole of the difference.
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

    obs_encoder = getattr(cfg.policy, "obs_encoder", None)
    if obs_encoder is None:
        return
    rgb = obs_encoder.get("rgb_model", None)
    if rgb is None:
        return
    if str(rgb.get("_target_", "")) == "diffusion_policy.model.vision.model_getter.get_resnet":
        with open_dict(rgb):
            rgb.weights = None


def cargar_politica_paper(checkpoint, salida):
    """Rebuild the workspace from the config carried inside the checkpoint."""
    import dill
    import hydra
    import torch

    assert checkpoint.is_file(), f"falta el punto de control del articulo: {checkpoint}"
    with open(checkpoint, "rb") as handle:
        payload = torch.load(handle, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    neutralizar_descarga(cfg)

    clase = hydra.utils.get_class(cfg._target_)
    workspace = clase(cfg, output_dir=str(salida))
    # The optimiser state is half the file and plays no part in a rollout.
    workspace.load_payload(payload, exclude_keys=("optimizer",), include_keys=None)

    politica = workspace.ema_model if cfg.training.use_ema else workspace.model
    assert politica is not None, "el checkpoint del articulo no trae pesos EMA"
    workspace.optimizer = None
    politica.eval()
    politica.to(torch.device(cfg.training.device))
    return cfg, politica


def comprobar_protocolo(cfg):
    """The evaluator and the horizons must be the ones the five variants used."""
    assert REFERENCIA.is_file(), f"falta la referencia del protocolo: {REFERENCIA}"
    referencia = json.loads(REFERENCIA.read_text(encoding="utf-8"))["runner"]
    runner = cfg.task.env_runner
    for clave in CLAVES_RUNNER:
        propio, esperado = runner.get(clave), referencia[clave]
        assert propio == esperado, (
            f"el evaluador del articulo difiere en {clave}: {propio!r} frente a "
            f"{esperado!r} en {REFERENCIA.name}. No seria el mismo experimento."
        )
    for clave, esperado in (("horizon", 16), ("n_obs_steps", 2), ("n_action_steps", 8)):
        assert int(cfg[clave]) == esperado, f"{clave} del articulo es {cfg[clave]}, no {esperado}"
    assert int(cfg.policy.num_inference_steps) == 100, (
        f"pasos de difusion del articulo: {cfg.policy.num_inference_steps}, no 100"
    )


class _Capturado(Exception):
    """Sentinel: the noise tensor is in hand, the rollout is no longer needed."""


def primer_ruido(politica, semilla):
    """First tensor ``conditional_sample`` draws, with the generator pinned.

    Used by the third gate. The common random numbers only pair the two arms if
    both consume the generator in the same place: the encoder must not draw
    before the diffusion. With a fixed crop in eval mode it should not, but the
    preregistration says this is checked and not assumed.
    """
    import torch

    original = torch.randn
    capturado = {}

    def espia(*args, **kwargs):
        capturado["ruido"] = original(*args, **kwargs).detach().clone().cpu()
        torch.randn = original
        # The rest of the hundred diffusion steps would tell us nothing more.
        raise _Capturado

    dispositivo = next(politica.parameters()).device
    obs = {
        "image": torch.zeros(1, 2, 3, 96, 96, device=dispositivo).uniform_(0.0, 1.0),
        "agent_pos": torch.zeros(1, 2, 2, device=dispositivo).uniform_(0.0, 512.0),
    }
    torch.randn = espia
    try:
        torch.manual_seed(semilla)
        with torch.no_grad():
            politica.predict_action(obs)
    except _Capturado:
        pass
    finally:
        torch.randn = original
    assert "ruido" in capturado, "la politica no llego a muestrear ruido de difusion"
    return capturado["ruido"]


def comprobar_ruido(args):
    """Third gate: V0 and V_Paper must draw the same first noise tensor."""
    import torch

    repo = args.repo.expanduser().resolve()
    assert (repo / "diffusion_policy").is_dir(), f"no parece la copia de trabajo: {repo}"
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    salida = Path(os.environ.get("TMPDIR", "/tmp")) / "comprobar_ruido"
    salida.mkdir(parents=True, exist_ok=True)
    semilla = args.base_seed * 1000003

    _, politica_v0, _ = cargar_politica(repo, "v0", salida)
    ruido_v0 = primer_ruido(politica_v0, semilla)
    del politica_v0
    torch.cuda.empty_cache()

    _, politica_paper = cargar_politica_paper(args.checkpoint.expanduser().resolve(), salida)
    ruido_paper = primer_ruido(politica_paper, semilla)

    iguales = ruido_v0.shape == ruido_paper.shape and torch.equal(ruido_v0, ruido_paper)
    print(f"V0      : forma {tuple(ruido_v0.shape)}")
    print(f"V_Paper : forma {tuple(ruido_paper.shape)}")
    if iguales:
        print("Ruido comun alineado: los dos brazos muestrean el mismo tensor.")
    else:
        # The preregistration declares this contingency: the pairing by initial
        # condition survives, only the variance reduction is lost.
        print(
            "Ruido comun NO alineado. El pareo por condicion inicial sigue siendo "
            "valido y el analisis no cambia; se pierde la reduccion de varianza de "
            "los numeros aleatorios comunes y hay que declararlo en el informe."
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_POR_DEFECTO)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--test-start-seed", type=int, default=TEST_START_SEED)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED_DIFUSION)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--etiqueta", default="prueba",
                        help="prefijo del fichero de salida: 'prueba' y 'ruido_b' "
                             "para las dos realizaciones, 'cordura' para el porton")
    parser.add_argument("--repo", type=Path, default=REPO_POR_DEFECTO)
    parser.add_argument("--forzar", action="store_true",
                        help="permite sobrescribir un resultado ya existente")
    parser.add_argument("--comprobar-ruido", action="store_true",
                        help="tercer porton: compara el primer tensor de ruido de "
                             "V0 y V_Paper y termina")
    args = parser.parse_args()

    if args.comprobar_ruido:
        raise SystemExit(comprobar_ruido(args))

    pasada = args.etiqueta in ("prueba", "ruido_b")
    semillas = list(range(args.test_start_seed, args.test_start_seed + args.n_test))
    if pasada:
        # The block must stay untouched by anything that chose either arm.
        assert not set(semillas) & set(SEMILLAS_DEMOSTRACIONES), "el bloque toca las demostraciones"
        assert not set(semillas) & set(SEMILLAS_SELECCION), "el bloque toca el conjunto de seleccion"
        assert not set(semillas) & set(SEMILLAS_SELECCION_PAPER), (
            "el bloque toca el conjunto con el que los autores eligieron su punto de control"
        )
        assert args.n_test == N_TEST and args.test_start_seed == TEST_START_SEED, (
            "las dos realizaciones usan el bloque preregistrado; para otra cosa, "
            "cambia --etiqueta"
        )
        assert args.n_test % args.n_envs == 0, (
            "el ultimo grupo se rellenaria repitiendo condiciones y el ruido "
            "comun dejaria de alinearse; usa un n_test multiplo de n_envs"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destino = OUT_DIR / f"{args.etiqueta}_paper.json"
    if destino.exists() and not args.forzar:
        raise SystemExit(
            f"{destino} ya existe. Cada corrida se ejecuta una sola vez; "
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

    checkpoint = args.checkpoint.expanduser().resolve()
    salida = Path(os.environ.get("TMPDIR", "/tmp")) / f"prueba_final_paper_{args.etiqueta}"
    salida.mkdir(parents=True, exist_ok=True)
    cfg, politica = cargar_politica_paper(checkpoint, salida)
    comprobar_protocolo(cfg)

    # The evaluation block, and only that. legacy_test, max_steps, fps,
    # n_obs_steps and n_action_steps stay as the checkpoint carries them, and
    # comprobar_protocolo has already checked they match prueba_v0.json.
    runner_cfg = cfg.task.env_runner
    with open_dict(runner_cfg):
        runner_cfg.n_train = 0
        runner_cfg.n_train_vis = 0
        runner_cfg.n_test = args.n_test
        runner_cfg.n_test_vis = 0
        runner_cfg.test_start_seed = args.test_start_seed
        runner_cfg.n_envs = args.n_envs

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
        "variante": "V_PAPER",
        "codificador": "ResNet-18 con spatial softmax de robomimic",
        "etiqueta": args.etiqueta,
        "punto_control": checkpoint.name,
        "sha256": sha256(checkpoint),
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
        # Absent from the five files of the final test; the external review asks
        # for them in finding m8.
        "fecha_iso": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": commit(),
        "puntuaciones": {str(s): puntuaciones[s] for s in semillas},
    }
    destino.write_text(json.dumps(resultado, indent=2), encoding="utf-8")

    exitos = sum(1 for v in puntuaciones.values() if v >= 0.999)
    print(
        f"V_PAPER · {args.etiqueta} · {args.n_test} condiciones desde "
        f"{args.test_start_seed}: media {media:.4f} · exito {exitos}/{args.n_test} · "
        f"{estado['llamadas']} llamadas en {duracion / 60:.1f} min"
    )
    print(f"Escrito en {destino}")

    if args.etiqueta == "cordura":
        desvio = abs(media - CORDURA_ESPERADA)
        veredicto = "PASA" if desvio <= CORDURA_TOLERANCIA else "NO PASA"
        print(
            f"Porton de cordura: |{media:.4f} - {CORDURA_ESPERADA}| = {desvio:.4f} "
            f"frente a la tolerancia {CORDURA_TOLERANCIA} -> {veredicto}"
        )
        if desvio > CORDURA_TOLERANCIA:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
