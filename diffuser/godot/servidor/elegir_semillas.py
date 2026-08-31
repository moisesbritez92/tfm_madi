#!/usr/bin/env python3
"""Elige semillas utilizables para un experimento de perturbaciones.

Una perturbacion solo se puede observar si hay sitio para caer, asi que las
semillas se eligen entre las que las variantes implicadas ya resuelven. Ese
filtro sesga las cifras absolutas al alza y hay que declararlo, pero sin el no
hay nada que medir.

Hay un segundo filtro, menos evidente y que costo un barrido entero aprenderlo.
La puntuacion de Push-T es el **maximo** de la cobertura a lo largo del episodio,
y algunas condiciones iniciales ya arrancan con la pieza parcialmente sobre el
objetivo. En esas, si la politica empuja la pieza fuera y no la recupera, la
puntuacion final es la de la pose de partida: identica para todas las variantes,
identica para todas las perturbaciones, y sin ninguna informacion sobre nada.
Las semillas 200007 y 200019 arrancan a 0,665 y 0,373 de cobertura, y sus ocho
celdas dieron exactamente el mismo numero.

Este script cruza los dos filtros y ordena por cobertura inicial ascendente.

Uso:
    ... elegir_semillas.py --variantes v0 v3
    ... elegir_semillas.py --variantes v0 v3 --cobertura-maxima 0.05 --cuantas 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

AQUI = Path(__file__).resolve().parent
GODOT = AQUI.parent
DIFFUSER = GODOT.parent
RAIZ = DIFFUSER.parent
for ruta in (str(DIFFUSER / "repo" / "diffusion_policy"), str(AQUI)):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv  # noqa: E402
from rasterizador_pusht import RasterizadorPushT  # noqa: E402

PRUEBA_FINAL = RAIZ / "logs_entrenamiento" / "prueba_final"


def estado_inicial(env, seed: int) -> list:
    env.seed(seed)
    env.reset()
    return [
        float(env.agent.position[0]), float(env.agent.position[1]),
        float(env.block.position[0]), float(env.block.position[1]),
        float(env.block.angle),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variantes", nargs="+", default=["v0", "v3"],
                        help="todas deben resolver la semilla en el preregistro")
    parser.add_argument("--umbral-exito", type=float, default=0.999)
    parser.add_argument("--cobertura-maxima", type=float, default=0.30,
                        help="descarta las condiciones iniciales que ya empiezan resueltas")
    parser.add_argument("--cuantas", type=int, default=0,
                        help="0 para listarlas todas")
    args = parser.parse_args()

    puntuaciones = {}
    for variante in args.variantes:
        ruta = PRUEBA_FINAL / f"prueba_{variante}.json"
        if not ruta.is_file():
            raise SystemExit(f"falta {ruta}")
        puntuaciones[variante] = json.loads(ruta.read_text(encoding="utf-8"))["puntuaciones"]

    comunes = set.intersection(*[
        {s for s, v in puntuaciones[variante].items() if v >= args.umbral_exito}
        for variante in args.variantes
    ])

    env = PushTImageEnv(legacy=True, render_size=96)
    rasterizador = RasterizadorPushT()

    filas = []
    for seed in sorted(int(s) for s in comunes):
        inicial = rasterizador.cobertura(estado_inicial(env, seed))
        filas.append((inicial, seed))
    filas.sort()

    total = len(puntuaciones[args.variantes[0]])
    print(f"de {total} semillas del bloque disjunto, {len(comunes)} las resuelven "
          f"todas de {', '.join(v.upper() for v in args.variantes)}")
    print(f"\n{'seed':>8}  {'cob. inicial':>12}  estado")
    utilizables = []
    for inicial, seed in filas:
        if inicial <= args.cobertura_maxima:
            utilizables.append(seed)
            marca = "utilizable"
        else:
            marca = "DEGENERADA: arranca parcialmente resuelta"
        print(f"{seed:8d}  {inicial:12.4f}  {marca}")

    if args.cuantas:
        utilizables = utilizables[:args.cuantas]
    print(f"\n{len(utilizables)} utilizables: {' '.join(str(s) for s in utilizables)}")
    print("\nSiguen estando condicionadas al exito previo de todas las variantes,\n"
          "asi que sus lineas base salen mejores de lo que ninguna variante es en\n"
          "general. Lo unico legible sera la caida de cada una respecto de si misma.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
