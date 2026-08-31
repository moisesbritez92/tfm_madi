#!/usr/bin/env python3
"""Barre el color de la pieza y mira si la caida es gradual o hay un acantilado.

La pregunta que contesta: cuando la politica deja de resolver la tarea al pintar
la pieza de rojo, es porque el color se ha alejado *poco a poco* de lo que vio en
el entrenamiento, o porque cruza una frontera y de repente la pieza deja de ser
una pieza. Lo primero seria un problema de distancia en el espacio de color; lo
segundo, que la politica esta clasificando por tono.

El eje principal es una interpolacion lineal en RGB de 8 bits entre el gris
azulado del entrenamiento y el `firebrick` de la perturbacion `t_roja`. Lineal en
RGB y no en tono porque RGB es literalmente lo que entra a la red.

Ese eje mezcla dos cosas, y por eso lleva dos controles:

  luminancia   el gris azulado tiene luminancia 133,6 y el firebrick, 64,6. El
               rojo no solo es mas rojo, es mas oscuro.
  gris_oscuro  un gris neutro con la luminancia del firebrick. Si aqui tambien
               falla, parte del efecto es brillo y no tono.
  rojo_isolum  un rojo desaturado con la luminancia del gris azulado original.
               Si aqui falla igual que en el firebrick, el efecto es tono.

Necesita un servidor de politica ya levantado en modo `--obs godot`. No guarda
imagenes: escribe las grabaciones normales y va imprimiendo por pantalla.

Uso:
    # en otra consola
    .venv_diffuser_infer\\Scripts\\python.exe diffuser/godot/servidor/servidor_politica.py \\
        --variante v0 --obs godot --puerto 5563
    # aqui
    .venv_diffuser_infer\\Scripts\\python.exe diffuser/godot/servidor/barrido_color.py \\
        --puerto 5563 --semillas 200023 200024 200051
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
GODOT = AQUI.parent

# Los dos extremos del eje. El gris azulado es `LightSlateGray`, el color con el
# que se dibuja el borde de la pieza en el renderizador original; el rojo es el
# `firebrick` que usa la perturbacion `t_roja`.
GRIS = (119, 136, 153)
ROJO = (178, 34, 34)

# Coeficientes de luminancia de Rec. 709, los mismos que usa cualquier conversion
# a escala de grises sensata.
LUMA = (0.2126, 0.7152, 0.0722)

PATRON_FIN = re.compile(r"^fin \| (\d+) pasos \| cobertura maxima ([\d.]+) \| "
                        r"recompensa ([\d.]+)")


def luminancia(color) -> float:
    return sum(k * c for k, c in zip(LUMA, color))


def mezclar(a, b, t: float):
    return tuple(int(round(x + t * (y - x))) for x, y in zip(a, b))


def hacia_blanco(color, objetivo_luz: float):
    """Aclara hacia blanco hasta alcanzar una luminancia dada."""
    luz = luminancia(color)
    blanco = luminancia((255, 255, 255))
    if blanco <= luz:
        return color
    alfa = (objetivo_luz - luz) / (blanco - luz)
    alfa = max(0.0, min(1.0, alfa))
    return tuple(int(round(c + alfa * (255 - c))) for c in color)


def hexa(color) -> str:
    return "%02x%02x%02x" % color


def distancia(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def paleta(pasos: int) -> list:
    """El eje principal mas los dos controles, en orden de ejecucion."""
    filas = []
    for i in range(pasos):
        t = i / (pasos - 1)
        color = mezclar(GRIS, ROJO, t)
        filas.append({"etiqueta": f"mezcla {t:.2f}", "t": t, "color": color,
                      "control": False})
    filas.append({
        "etiqueta": "gris oscuro", "t": None,
        "color": tuple(int(round(luminancia(ROJO))) for _ in range(3)),
        "control": True,
    })
    filas.append({
        "etiqueta": "rojo isolum", "t": None,
        "color": hacia_blanco(ROJO, luminancia(GRIS)),
        "control": True,
    })
    return filas


def resolver_godot(dado: str) -> str:
    if dado:
        return dado
    import os

    local = Path(os.environ.get("LOCALAPPDATA", ""))
    paquetes = local / "Microsoft" / "WinGet" / "Packages"
    for ruta in sorted(paquetes.glob("GodotEngine*/Godot_v*_console.exe")):
        return str(ruta)
    enlace = local / "Microsoft" / "WinGet" / "Links" / "godot_console.exe"
    if enlace.is_file():
        return str(enlace)
    raise SystemExit("no se encuentra Godot; pasa --godot <ruta>")


def episodio(godot: str, perturbacion: str, seed: int, puerto: int, tiempo: int):
    orden = [
        godot, "--path", str(GODOT), "--",
        "modo=grabar", "obs=godot", f"seed={seed}",
        f"puerto={puerto}", f"perturbacion={perturbacion}",
    ]
    try:
        salida = subprocess.run(orden, capture_output=True, text=True, timeout=tiempo)
    except subprocess.TimeoutExpired:
        return None
    for linea in salida.stdout.splitlines():
        coincide = PATRON_FIN.match(linea.strip())
        if coincide:
            return {"pasos": int(coincide[1]), "cobertura": float(coincide[2]),
                    "recompensa": float(coincide[3])}
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--puerto", type=int, default=5563)
    parser.add_argument("--semillas", type=int, nargs="+",
                        default=[200023, 200024, 200051])
    parser.add_argument("--pasos-color", type=int, default=7,
                        help="puntos del eje gris -> rojo, extremos incluidos")
    parser.add_argument("--tiempo", type=int, default=400, help="segundos por episodio")
    parser.add_argument("--godot", default="")
    parser.add_argument("--salida", type=Path,
                        default=GODOT / "grabaciones" / "barrido_color.json")
    args = parser.parse_args()

    godot = resolver_godot(args.godot)
    colores = paleta(args.pasos_color)

    print(f"barrido de color sobre {len(args.semillas)} semillas "
          f"x {len(colores)} tonos = {len(colores) * len(args.semillas)} episodios")
    print(f"gris de entrenamiento #{hexa(GRIS)} (luminancia {luminancia(GRIS):.1f})  ->  "
          f"firebrick #{hexa(ROJO)} (luminancia {luminancia(ROJO):.1f})")
    cabecera = (f"\n{'tono':>12}  {'hex':>8}  {'dist RGB':>9}  {'luma':>6}  "
                + "  ".join(f"{s:>8}" for s in args.semillas) + f"  {'media':>7}")
    print(cabecera)
    print("-" * len(cabecera))

    filas = []
    inicio = time.time()
    for entrada in colores:
        color = entrada["color"]
        perturbacion = "t_color_" + hexa(color)
        recompensas = []
        celdas = []
        for seed in args.semillas:
            resultado = episodio(godot, perturbacion, seed, args.puerto, args.tiempo)
            if resultado is None:
                celdas.append(f"{'fallo':>8}")
            else:
                recompensas.append(resultado["recompensa"])
                celdas.append(f"{resultado['recompensa']:8.4f}")
            # Se imprime la linea entera cuando esta completa, pero el progreso se
            # nota: cada episodio son minuto y medio y esto corre media hora.
            print(f"\r{entrada['etiqueta']:>12}  #{hexa(color):>7}  "
                  f"{distancia(GRIS, color):9.1f}  {luminancia(color):6.1f}  "
                  + "  ".join(celdas), end="", flush=True)
        media = sum(recompensas) / len(recompensas) if recompensas else float("nan")
        print(f"  {media:7.4f}", flush=True)
        filas.append({
            "etiqueta": entrada["etiqueta"], "t": entrada["t"], "control": entrada["control"],
            "hex": hexa(color), "rgb": list(color),
            "distancia_rgb": round(distancia(GRIS, color), 2),
            "luminancia": round(luminancia(color), 1),
            "recompensas": recompensas, "media": media,
        })

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps({
        "semillas": args.semillas, "gris": list(GRIS), "rojo": list(ROJO),
        "filas": filas,
    }, indent=2), encoding="utf-8")
    print(f"\n{(time.time() - inicio) / 60:.0f} min | escrito en {args.salida}")

    print("\n--- para la bitacora ---\n")
    print("| tono | hex | distancia RGB | luminancia | " +
          " | ".join(str(s) for s in args.semillas) + " | media |")
    print("|" + "---|" * (5 + len(args.semillas)))
    for fila in filas:
        celdas = [f"{r:.4f}" for r in fila["recompensas"]]
        celdas += ["-"] * (len(args.semillas) - len(celdas))
        print(f"| {fila['etiqueta']} | #{fila['hex']} | {fila['distancia_rgb']:.1f} | "
              f"{fila['luminancia']:.1f} | " + " | ".join(celdas) +
              f" | **{fila['media']:.4f}** |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
