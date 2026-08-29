#!/usr/bin/env python3
"""Pone la observacion de Godot al lado de la del renderizador original.

Es el paso 6 de verificacion. La condicion B solo tiene sentido si lo que sale
del SubViewport de Godot se parece a lo que la politica vio durante el
entrenamiento; si no se parece, un fallo de la politica no dice nada sobre la
politica, solo sobre el ajuste de la camara.

Escribe un PNG de tres paneles a 96 px: Godot, el rasterizador y el valor
absoluto de la diferencia, y ademas resume por color cuanto se desvia cada
elemento de la escena.

Uso:
    ... comparar_observacion.py --godot diffuser/godot/grabaciones/observacion_godot_seed10000.png
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
for ruta in (str(DIFFUSER / "repo" / "diffusion_policy"), str(AQUI)):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

from rasterizador_pusht import RasterizadorPushT  # noqa: E402

# Los colores que el renderizador original produce, para atribuir la diferencia
# a un elemento concreto de la escena en lugar de dar un solo numero global.
REFERENCIAS = {
    "fondo": (255, 255, 255),
    "objetivo": (144, 238, 144),
    "T relleno": (142, 163, 183),
    "T borde": (119, 136, 153),
    "agente relleno": (78, 126, 255),
    "agente borde": (65, 105, 225),
    "pared": (211, 211, 211),
}


def leer_png(ruta: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(ruta).convert("RGB"), dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--godot", type=Path,
                        default=GODOT / "grabaciones" / "observacion_godot_seed10000.png")
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()

    if not args.godot.is_file():
        raise SystemExit(
            f"falta {args.godot}. Generalo con:\n"
            f"  godot --path {GODOT} -- modo=observacion seed=10000"
        )
    estado_ruta = args.godot.with_suffix(".json")
    if not estado_ruta.is_file():
        raise SystemExit(f"falta el estado que acompana a la imagen: {estado_ruta}")

    estado = json.loads(estado_ruta.read_text(encoding="utf-8"))["estado"]
    de_godot = leer_png(args.godot)
    de_original = RasterizadorPushT().imagen(estado)

    if de_godot.shape != de_original.shape:
        raise SystemExit(f"formas distintas: godot {de_godot.shape}, "
                         f"original {de_original.shape}")

    diferencia = np.abs(de_godot.astype(np.int16) - de_original.astype(np.int16))
    distintos = int((diferencia.max(axis=-1) > 8).sum())
    total = de_godot.shape[0] * de_godot.shape[1]

    print(f"estado: {[round(v, 2) for v in estado]}")
    print(f"pixeles con diferencia mayor que 8: {distintos}/{total} "
          f"({100.0 * distintos / total:.1f} %)")
    print(f"diferencia media {diferencia.mean():.2f} | maxima {diferencia.max()}")

    # A que color se parece mas cada pixel, en una imagen y en la otra: si Godot
    # pinta la T con un tono que el original nunca produce, se ve aqui.
    print(f"\n{'elemento':>16}  {'original':>9}  {'godot':>9}")
    nombres = list(REFERENCIAS)
    paleta = np.asarray([REFERENCIAS[n] for n in nombres], dtype=np.int16)
    conteos = []
    for imagen in (de_original, de_godot):
        plano = imagen.reshape(-1, 1, 3).astype(np.int16)
        cercano = np.abs(plano - paleta[None, :, :]).sum(axis=-1).argmin(axis=1)
        conteos.append(np.bincount(cercano, minlength=len(nombres)))
    for i, nombre in enumerate(nombres):
        print(f"{nombre:>16}  {conteos[0][i]:9d}  {conteos[1][i]:9d}")

    salida = args.salida or args.godot.with_name(args.godot.stem + "_vs_original.png")
    from PIL import Image

    panel = np.concatenate([de_original, de_godot, diferencia.astype(np.uint8)], axis=1)
    Image.fromarray(panel).resize(
        (panel.shape[1] * 4, panel.shape[0] * 4), Image.NEAREST
    ).save(salida)
    print(f"\npaneles (original | godot | diferencia) en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
