#!/usr/bin/env python3
"""Convierte una grabacion en algo que se pueda mirar: un GIF o una tira.

Lanzar Godot para cada episodio es lento y no deja nada que comparar despues.
Esto lee el JSON de `grabaciones/` y dibuja la trayectoria con el rasterizador
del entrenamiento, que ya esta ahi y es exacto al pixel.

**La imagen es siempre la del renderizador original**, tambien cuando el episodio
se corrio con una perturbacion. Lo que se ve aqui es la trayectoria, no lo que la
politica veia: para eso estan los paneles de `comparar_observacion.py`. Se dice
en el propio nombre del fichero de salida.

Uso:
    ... ver_episodio.py --todos                        un GIF por grabacion
    ... ver_episodio.py --tira                         ademas, una tira de 8 poses
    ... ver_episodio.py --patron "*v3*t_roja*"         solo algunas
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

# El GIF se ve mejor grande; 96 px es lo que ve la politica, no lo que ve una
# persona.
ESCALA = 4
POSES_EN_LA_TIRA = 8


def fotogramas(rasterizador, traza, escala=ESCALA):
    salida = []
    for cuadro in traza:
        img = rasterizador.imagen(cuadro["estado"])
        salida.append(np.kron(img, np.ones((escala, escala, 1), dtype=np.uint8)))
    return salida


def etiquetar(datos: dict) -> str:
    return "%s | %s | %s | seed %s | recompensa %.4f" % (
        datos.get("variante", "v0").upper(),
        "A" if datos["obs"] == "estado" else "B",
        datos.get("perturbacion", "ninguna"),
        datos["seed"],
        datos["recompensa_max"],
    )


def escribir_gif(cuadros, destino: Path, fps: int) -> None:
    import imageio.v2 as imageio

    imageio.mimsave(destino, cuadros, duration=1.0 / fps, loop=0)


def escribir_tira(cuadros, destino: Path) -> None:
    """Una fila de poses igualmente espaciadas, de la primera a la ultima."""
    from PIL import Image

    indices = np.linspace(0, len(cuadros) - 1, POSES_EN_LA_TIRA).round().astype(int)
    tira = np.concatenate([cuadros[i] for i in indices], axis=1)
    Image.fromarray(tira).save(destino)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grabaciones", type=Path, default=GODOT / "grabaciones")
    parser.add_argument("--salida", type=Path, default=None,
                        help="por defecto, junto a las grabaciones, en video/")
    parser.add_argument("--patron", default="grabar_*.json")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--tira", action="store_true",
                        help="ademas del GIF, una tira de 8 poses")
    parser.add_argument("--solo-tira", action="store_true")
    args = parser.parse_args()

    rutas = sorted(args.grabaciones.glob(args.patron))
    if not rutas:
        raise SystemExit(f"nada que coincida con {args.patron} en {args.grabaciones}")

    destino = args.salida or args.grabaciones / "video"
    destino.mkdir(parents=True, exist_ok=True)
    rasterizador = RasterizadorPushT()

    for ruta in rutas:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        cuadros = fotogramas(rasterizador, datos["traza"])
        # El sufijo recuerda que la imagen es del renderizador original aunque el
        # episodio se corriera perturbado.
        base = destino / (ruta.stem.replace("grabar_", "") + "_render_original")
        if not args.solo_tira:
            escribir_gif(cuadros, base.with_suffix(".gif"), args.fps)
        if args.tira or args.solo_tira:
            escribir_tira(cuadros, base.with_name(base.name + "_tira").with_suffix(".png"))
        print(f"{etiquetar(datos):>62}  ->  {base.name}")

    print(f"\nEn {destino}")
    print("La imagen es la del renderizador original en todos los casos, tambien en\n"
          "los episodios perturbados: muestra la trayectoria, no lo que vio la politica.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
