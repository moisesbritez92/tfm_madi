#!/usr/bin/env python3
"""Contrasta el port de Godot contra el entorno original.

Dos comprobaciones, las que el plan numera 3 y 4, y ninguna necesita GPU ni
servidor de politica.

``cobertura``  La geometria. Godot calcula el solape con
               ``Geometry2D.intersect_polygons`` y esto lo recalcula con
               shapely sobre las mismas poses. Una diferencia por encima de
               1e-3 delata un error de vertices o de convenio de angulo, y hay
               que mirarla antes que cualquier otra cosa.

``fisica``     El motor. Godot ejecuta un guion de acciones fijo y anota la pose
               de la pieza en cada paso de control; esto repite el mismo guion
               en pymunk desde la misma condicion inicial y mide como se
               separan las dos trayectorias. No van a coincidir: son
               solucionadores distintos, y esa diferencia es el objeto de la
               demostracion. Lo que si tiene que ser es gradual. Un salto en el
               primer contacto significa que falta anular la velocidad en cada
               subpaso o que el centro de masa no es (0, 45).

Uso:
    .venv_diffuser_infer\\Scripts\\python.exe diffuser/godot/servidor/verificar_port.py cobertura
    .venv_diffuser_infer\\Scripts\\python.exe diffuser/godot/servidor/verificar_port.py fisica
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

AQUI = Path(__file__).resolve().parent
GODOT = AQUI.parent
DIFFUSER = GODOT.parent
REPO_ROOT = DIFFUSER / "repo" / "diffusion_policy"
for ruta in (str(REPO_ROOT), str(AQUI)):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv  # noqa: E402
from rasterizador_pusht import RasterizadorPushT  # noqa: E402

TOLERANCIA_COBERTURA = 1e-3


def _cargar(ruta: Path) -> dict:
    if not ruta.is_file():
        raise SystemExit(
            f"falta {ruta}. Generalo antes con:\n"
            f"  godot --headless --path {GODOT} -- modo={ruta.stem.split('_')[0]}"
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def comprobar_cobertura(ruta: Path) -> int:
    datos = _cargar(ruta)
    rasterizador = RasterizadorPushT()
    print(f"{'pose':>22}  {'godot':>10}  {'shapely':>10}  {'delta':>10}")
    peor = 0.0
    for fila in datos["filas"]:
        x, y = fila["pos"]
        angulo = fila["ang"]
        # La cobertura no depende del agente; la posicion que se le pasa es
        # cualquiera que no estorbe.
        referencia = rasterizador.cobertura([256.0, 480.0, x, y, angulo])
        delta = abs(referencia - fila["cobertura_godot"])
        peor = max(peor, delta)
        print(f"({x:6.1f},{y:6.1f}) {angulo:6.3f}  "
              f"{fila['cobertura_godot']:10.6f}  {referencia:10.6f}  {delta:10.2e}")
    print(f"\npeor diferencia {peor:.2e} (tolerancia {TOLERANCIA_COBERTURA:.0e})")
    if peor <= TOLERANCIA_COBERTURA:
        print("el port geometrico coincide con shapely")
        return 0
    print("DIFIERE: revisa los vertices o el convenio de angulo")
    return 1


def comprobar_fisica(ruta: Path) -> int:
    datos = _cargar(ruta)
    traza = datos["traza"]
    estado0 = traza[0]["estado"]

    env = PushTImageEnv(legacy=True, render_size=96)
    env.reset()
    # Angulo antes que posicion, por la misma razon que en el rasterizador: el
    # cuerpo gira alrededor de (0, 45) y no de su origen.
    env.agent.position = (estado0[0], estado0[1])
    env.agent.velocity = (0.0, 0.0)
    env.block.angle = estado0[4]
    env.block.position = (estado0[2], estado0[3])
    env.space.reindex_shapes_for_body(env.block)

    filas = []
    for cuadro in traza[1:]:
        env.step(np.asarray(cuadro["objetivo"], dtype=np.float64))
        godot = np.asarray(cuadro["estado"][2:4], dtype=np.float64)
        pymunk = np.asarray([env.block.position[0], env.block.position[1]])
        d_ang = float(cuadro["estado"][4]) - float(env.block.angle)
        # A un circulo, para que un giro de 2 pi no cuente como divergencia.
        d_ang = (d_ang + np.pi) % (2.0 * np.pi) - np.pi
        filas.append({
            "paso": int(cuadro["paso"]),
            "d_pos": float(np.linalg.norm(godot - pymunk)),
            "d_ang": abs(d_ang),
            "godot": godot.tolist(),
            "pymunk": pymunk.tolist(),
        })

    print(f"{'paso':>6}  {'d_pos (px)':>11}  {'d_ang (rad)':>12}")
    for fila in filas:
        if fila["paso"] % 25 == 0 or fila["paso"] <= 3:
            print(f"{fila['paso']:6d}  {fila['d_pos']:11.3f}  {fila['d_ang']:12.4f}")

    saltos = []
    for anterior, actual in zip(filas, filas[1:]):
        saltos.append((actual["d_pos"] - anterior["d_pos"], actual["paso"]))
    salto_max, paso_salto = max(saltos) if saltos else (0.0, 0)

    print(f"\ndivergencia final {filas[-1]['d_pos']:.2f} px, "
          f"{filas[-1]['d_ang']:.4f} rad tras {len(filas)} pasos de control")
    print(f"mayor salto entre pasos vecinos: {salto_max:.2f} px en el paso {paso_salto}")
    print("\nLas dos trayectorias no tienen por que coincidir: Chipmunk y Godot "
          "Physics 2D\nresuelven los contactos de forma distinta. Lo que se "
          "vigila es que la separacion\nsea gradual y no un salto en el primer "
          "contacto.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("comprobacion", choices=("cobertura", "fisica"))
    parser.add_argument("--traza", type=Path, default=None,
                        help="por defecto, la que escribe el modo correspondiente de Godot")
    args = parser.parse_args()

    grabaciones = GODOT / "grabaciones"
    if args.comprobacion == "cobertura":
        return comprobar_cobertura(args.traza or grabaciones / "cobertura_godot.json")
    return comprobar_fisica(args.traza or grabaciones / "comparar_godot.json")


if __name__ == "__main__":
    raise SystemExit(main())
