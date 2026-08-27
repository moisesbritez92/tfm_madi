#!/usr/bin/env python3
"""Characterisation of the Push-T demonstration set actually used for training.

Section 3.2 of the memoir described the ``zarr`` structurally -- 206 episodes,
25650 transitions, stored fields -- but did not characterise it. This script
produces the quantitative evidence that section now reports: the exact
train/validation/discard split, the distribution of episode lengths, the score
the human demonstrator reaches in each episode, and the initial conditions of
both the demonstrations and the 56 evaluation episodes.

Three decisions deserve an explanation.

  * **The split is not reimplemented.** ``get_val_mask`` and ``downsample_mask``
    are imported from the training tree, so the CSV describes the partition the
    runs really used and not a plausible reconstruction of it. The same applies
    to ``create_indices``, which counts the training windows.

  * **The state is reconstructed with** ``legacy=False``. The stored ``state``
    comes from ``PushTEnv._get_obs``, and only the non-legacy ordering of
    ``_set_state`` round-trips it exactly: with ``legacy=True`` the block is
    rotated about its centre of mass after being placed, which displaces it by
    up to 90 px. Both orderings give a plausible-looking coverage, so the script
    checks the reconstruction against the ``keypoint`` array of the ``zarr``
    instead of trusting it.

  * **The demonstrations turn out to be seeds 0 to 205 of the environment.**
    ``PushTEnv(legacy=True).seed(i).reset()`` reproduces the first transition of
    demonstration ``i`` exactly, for the 206 of them. The script asserts it,
    because it is what licenses two claims of section 3.2: demonstrations and
    evaluation draw their initial conditions from the same generator with
    disjoint seeds, and the six ``train/`` rollout conditions are the initial
    conditions of demonstrations 0 to 5.

  * **The image array is never read.** ``ReplayBuffer.copy_from_path`` would pull
    the 2.84 GB of ``float32`` images into RAM; only ``state`` and
    ``episode_ends`` are needed here, so the ``zarr`` is opened read-only.

It must run in WSL under the ``robodiff`` environment, which is where the
dataset and ``pymunk`` live:

    source ~/mambaforge/etc/profile.d/conda.sh && conda activate robodiff
    SDL_VIDEODRIVER=dummy python diffuser/scripts/caracterizar_dataset.py

Writes ``memoria/datos/demostraciones_episodios.csv`` and
``memoria/datos/condiciones_evaluacion.csv``. The figure is drawn on the Windows
side by ``memoria/scripts/figuras_dataset.py``, which reads only those two files.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "memoria" / "datos"

# The working copy where training actually ran; it is not versioned, so its path
# is a parameter and not an assumption baked into the script.
REPO_POR_DEFECTO = Path.home() / "tfm" / "diffusion_policy"
ZARR_RELATIVO = Path("data") / "pusht" / "pusht_cchi_v7_replay.zarr"

# task/pusht_image.yaml. dataset.seed is a literal 42, not ${seed}: the split
# does not follow the training seed and stays the same for seeds 43 and 44.
SEMILLA_PARTICION = 42
VAL_RATIO = 0.02
MAX_EPISODIOS_ENTRENAMIENTO = 90

# horizon 16, pad_before = n_obs_steps - 1, pad_after = n_action_steps - 1.
HORIZONTE = 16
PAD_ANTES = 1
PAD_DESPUES = 7

# pusht_image.yaml env_runner: 6 training conditions and 50 evaluation ones. The
# first six coincide with the initial conditions of demonstrations 0 to 5.
SEMILLAS_ENTRENAMIENTO_ROLLOUT = range(0, 6)
SEMILLAS_EVALUACION = range(100000, 100050)

UMBRAL_EXITO = 0.95  # PushTEnv.success_threshold
FRECUENCIA_CONTROL = 10  # Hz, PushTEnv.control_hz

# Largest displacement accepted between the stored state and the reconstructed
# one, over a board of 512 px. See the asserts at the end of main.
TOLERANCIA_PIXELES = 2.0

# What the split has to come out as. Asserted, not printed, because a silent
# change here would invalidate every figure of section 3.2.
REPARTO_ESPERADO = {"entrenamiento": 90, "validacion": 4, "descarte": 112}
VENTANAS_ESPERADAS = {"entrenamiento": 10726, "validacion": 404}


def condiciones_iniciales(semillas, PushTEnv):
    """Initial observation each seed produces, as the evaluation sees it.

    ``legacy=True`` is what the runner uses (``legacy_test`` in
    ``pusht_image.yaml``) and what the recorded demonstrations were collected
    with: placing the block before rotating it displaces it by up to 90 px, so
    the ordering is not cosmetic. Comparing these states with the first
    transition of each demonstration is therefore a like-for-like comparison.
    """
    entorno = PushTEnv(legacy=True)
    filas = []
    for semilla in semillas:
        entorno.seed(semilla)
        filas.append(entorno.reset())
    return np.array(filas, dtype=np.float64)


def _detener(entorno):
    """Zero the velocities before placing a state.

    ``_set_state`` ends with a physics step, and neither body has its velocity
    reset, so the momentum the previous placement left behind displaces the
    block by up to a pixel. Stopping both bodies makes each transition depend
    only on its own stored state.
    """
    entorno.block.velocity = (0.0, 0.0)
    entorno.block.angular_velocity = 0.0
    entorno.agent.velocity = (0.0, 0.0)


def cobertura_por_transicion(estados, PushTEnv, pymunk_to_shapely):
    """Fraction of the goal area covered by the block at each transition."""
    entorno = PushTEnv(legacy=False)
    entorno.seed(0)
    entorno.reset()
    objetivo = pymunk_to_shapely(
        entorno._get_goal_pose_body(entorno.goal_pose), entorno.block.shapes)
    area_objetivo = objetivo.area

    cobertura = np.empty(len(estados))
    error_ida_vuelta = 0.0
    for i, estado in enumerate(estados):
        _detener(entorno)
        entorno._set_state(estado)
        pieza = pymunk_to_shapely(entorno.block, entorno.block.shapes)
        cobertura[i] = pieza.intersection(objetivo).area / area_objetivo
        observacion = entorno._get_obs()
        error_ida_vuelta = max(
            error_ida_vuelta,
            float(np.abs(observacion[:4] - estado[:4]).max()))
    return cobertura, error_ida_vuelta


def error_de_reconstruccion(estados, puntos, PushTEnv, PymunkKeypointManager):
    """Largest displacement, in pixels, against the stored keypoints.

    Independent check of the ``legacy=False`` choice: the keypoints were written
    by the data collection itself. It uses its own environment because
    ``PymunkKeypointManager`` mutates the ``pymunk`` space it is built from, and
    a shared one would silently zero the coverage.
    """
    entorno = PushTEnv(legacy=False)
    entorno.seed(0)
    entorno.reset()
    gestor = PymunkKeypointManager.create_from_pusht_env(entorno)
    error = 0.0
    for i in range(0, len(estados), 37):
        _detener(entorno)
        entorno._set_state(estados[i])
        obtenidos = gestor.get_keypoints_global(
            pose_map={"block": entorno.block}, is_obj=True)["block"]
        error = max(error, float(np.abs(obtenidos - puntos[i]).max()))
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_POR_DEFECTO)
    argumentos = parser.parse_args()

    repo = argumentos.repo.expanduser()
    assert (repo / "diffusion_policy").is_dir(), f"no parece la copia de trabajo: {repo}"
    sys.path.insert(0, str(repo))
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import zarr
    from diffusion_policy.common.sampler import (
        create_indices, downsample_mask, get_val_mask)
    from diffusion_policy.env.pusht.pusht_env import PushTEnv, pymunk_to_shapely
    from diffusion_policy.env.pusht.pymunk_keypoint_manager import (
        PymunkKeypointManager)

    almacen = zarr.open(str(repo / ZARR_RELATIVO), mode="r")
    finales = almacen["meta/episode_ends"][:]
    estados = almacen["data/state"][:]
    puntos = almacen["data/keypoint"][:]
    n_episodios = len(finales)
    inicios = np.concatenate([[0], finales[:-1]])
    longitudes = finales - inicios

    mascara_validacion = get_val_mask(
        n_episodes=n_episodios, val_ratio=VAL_RATIO, seed=SEMILLA_PARTICION)
    mascara_entrenamiento = downsample_mask(
        mask=~mascara_validacion,
        max_n=MAX_EPISODIOS_ENTRENAMIENTO,
        seed=SEMILLA_PARTICION)
    mascara_descarte = ~mascara_validacion & ~mascara_entrenamiento

    particion = np.empty(n_episodios, dtype=object)
    particion[mascara_entrenamiento] = "entrenamiento"
    particion[mascara_validacion] = "validacion"
    particion[mascara_descarte] = "descarte"

    ventanas = {}
    for nombre, mascara in (("entrenamiento", mascara_entrenamiento),
                            ("validacion", mascara_validacion)):
        ventanas[nombre] = len(create_indices(
            finales, sequence_length=HORIZONTE, pad_before=PAD_ANTES,
            pad_after=PAD_DESPUES, episode_mask=mascara))

    error_puntos = error_de_reconstruccion(
        estados, puntos, PushTEnv, PymunkKeypointManager)
    cobertura, error_ida_vuelta = cobertura_por_transicion(
        estados, PushTEnv, pymunk_to_shapely)

    # Each demonstration starts exactly where seed <index> of the environment
    # places it, so the demonstrations and the evaluation draw their initial
    # conditions from the same generator with disjoint seeds.
    reset_demostraciones = condiciones_iniciales(range(n_episodios), PushTEnv)
    error_semillas = float(np.abs(
        reset_demostraciones[:, :4] - estados[inicios][:, :4]).max())

    cobertura_maxima = np.array(
        [cobertura[a:b].max() for a, b in zip(inicios, finales)])
    cobertura_final = np.array([cobertura[b - 1] for b in finales])
    puntuacion = np.clip(cobertura_maxima / UMBRAL_EXITO, 0.0, 1.0)

    for nombre, esperado in REPARTO_ESPERADO.items():
        obtenido = int((particion == nombre).sum())
        assert obtenido == esperado, f"{nombre}: {obtenido} episodios, no {esperado}"
    for nombre, esperado in VENTANAS_ESPERADAS.items():
        assert ventanas[nombre] == esperado, (
            f"ventanas de {nombre}: {ventanas[nombre]}, no {esperado}")
    # Both tolerances separate the right reconstruction from the wrong one by
    # two orders of magnitude: with legacy=True these errors reach 90 px. What
    # is left is contact resolution, since _set_state ends with a physics step
    # and the agent touches the block in part of the transitions.
    assert error_ida_vuelta < TOLERANCIA_PIXELES, (
        f"la reconstruccion no devuelve el estado guardado: {error_ida_vuelta:.4f} px")
    assert error_puntos < TOLERANCIA_PIXELES, (
        f"la reconstruccion no reproduce los keypoints: {error_puntos:.4f} px")
    assert error_semillas < TOLERANCIA_PIXELES, (
        "las demostraciones no proceden de las semillas 0..n del entorno: "
        f"{error_semillas:.4f} px")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destino_episodios = OUT_DIR / "demostraciones_episodios.csv"
    with open(destino_episodios, "w", newline="", encoding="utf-8") as fichero:
        escritor = csv.writer(fichero)
        escritor.writerow([
            "episodio", "semilla", "particion", "transiciones", "duracion_s",
            "cobertura_max", "cobertura_final", "puntuacion",
            "agente_x", "agente_y", "bloque_x", "bloque_y", "bloque_theta"])
        for i in range(n_episodios):
            inicial = estados[inicios[i]]
            escritor.writerow([
                i, i, particion[i], int(longitudes[i]),
                round(float(longitudes[i]) / FRECUENCIA_CONTROL, 2),
                round(float(cobertura_maxima[i]), 6),
                round(float(cobertura_final[i]), 6),
                round(float(puntuacion[i]), 6),
                round(float(inicial[0]), 3), round(float(inicial[1]), 3),
                round(float(inicial[2]), 3), round(float(inicial[3]), 3),
                round(float(inicial[4] % (2 * np.pi)), 6)])

    destino_condiciones = OUT_DIR / "condiciones_evaluacion.csv"
    semillas = list(SEMILLAS_ENTRENAMIENTO_ROLLOUT) + list(SEMILLAS_EVALUACION)
    iniciales = condiciones_iniciales(semillas, PushTEnv)
    with open(destino_condiciones, "w", newline="", encoding="utf-8") as fichero:
        escritor = csv.writer(fichero)
        escritor.writerow([
            "semilla", "conjunto",
            "agente_x", "agente_y", "bloque_x", "bloque_y", "bloque_theta"])
        for semilla, fila in zip(semillas, iniciales):
            conjunto = "entrenamiento" if semilla < 100000 else "evaluacion"
            escritor.writerow([
                semilla, conjunto,
                round(float(fila[0]), 3), round(float(fila[1]), 3),
                round(float(fila[2]), 3), round(float(fila[3]), 3),
                round(float(fila[4] % (2 * np.pi)), 6)])

    print(f"episodios              : {n_episodios}")
    for nombre in ("entrenamiento", "validacion", "descarte"):
        mascara = particion == nombre
        print(f"  {nombre:<14}: {mascara.sum():3d} episodios, "
              f"{longitudes[mascara].sum():5d} transiciones, "
              f"longitud media {longitudes[mascara].mean():.1f}, "
              f"puntuacion media {puntuacion[mascara].mean():.3f}")
    print(f"ventanas de horizonte {HORIZONTE}: {ventanas}")
    print(f"error de ida y vuelta  : {error_ida_vuelta:.4f} px")
    print(f"error contra keypoints : {error_puntos:.4f} px")
    print(f"error semillas 0..205  : {error_semillas:.4f} px")
    print(f"escritos {destino_episodios} y {destino_condiciones}")


if __name__ == "__main__":
    main()
