#!/usr/bin/env python3
"""Figuras del contraste V0 / V_Paper dentro de Godot, para el capitulo 4.

Dos figuras, una por condicion de observacion, con la misma condicion inicial y
la misma disposicion: dos filas (los dos artefactos) por tres columnas (inicio,
un instante intermedio y final del episodio).

Lo que se dibuja es **la vista en perspectiva del simulador**, con su luz y sus
sombras, y no la observacion de 96 por 96 que recibe la politica. Conviene tener
presente la consecuencia, que las leyendas declaran: esa vista **es la misma en
las dos condiciones**, porque lo que cambia entre ellas no es la escena sino que
imagen se le entrega a la politica. En la condicion A la entrada la dibuja el
rasterizador del entrenamiento; solo en la B procede de esta escena, capturada
por una camara ortografica cenital y reducida a 96 por 96.

Las imagenes las produce siempre Godot, en una sola ejecucion por celda, con
``modo=fotogramas`` y la clave ``vista`` puesta a ``demo``. El HUD no se monta:
rotula sus cifras como ilustrativas y estas figuras acompanan a un apartado de
resultados medidos.

Dos detalles que no son cosmeticos y que la leyenda declara:

  * **El fotograma inicial no se toma de la traza.** El cuadro cero que graba
    Godot lleva el agente en el origen, porque el cuerpo animado no adopta su
    posicion hasta el primer tic de fisica. La pose de partida se vuelve a
    muestrear aqui con el ``PushTEnv`` original, que es de donde salio.
  * **La columna central es el instante de cobertura maxima, que es el que
    puntua**, ya que la metrica toma el maximo a lo largo del episodio. Cuando el
    episodio termina por alcanzar el umbral ese instante es el ultimo, la columna
    repetiria imagen, y entonces pasa a mostrar el punto medio con otro rotulo.

    .venv_diffuser_infer\\Scripts\\python.exe memoria/scripts/figuras_godot_paper.py

Escribe ``memoria/img/godot_paper_condicion_{a,b}.pdf``.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GODOT_DIR = ROOT / "diffuser" / "godot"
EPISODIOS = GODOT_DIR / "grabaciones" / "godot_paper"
IMG_DIR = ROOT / "memoria" / "img"
TMP = GODOT_DIR / "grabaciones" / "figuras"

sys.path[:0] = [
    str(ROOT / "diffuser" / "repo" / "diffusion_policy"),
    str(GODOT_DIR / "servidor"),
]

# La unica condicion del bloque que resuelven los dos artefactos en las cuatro
# celdas, y ademas arranca en cobertura cero: no viene medio resuelta de casa.
SEMILLA = 200008
REALIZACION = "prueba"

BRAZOS = [("v0", "V0"), ("v_paper", r"V$_{\mathrm{pub}}$")]

# La condicion de observacion no se rotula dentro de la imagen: la nombra el pie
# de figura de la memoria, y repetirla aqui duplicaria el mismo texto dos veces
# en la misma pagina.


def estado_inicial(seed):
    """La pose de partida, del propio entorno y no de la traza."""
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

    env = PushTImageEnv(legacy=True, render_size=96)
    env.seed(seed)
    env.reset()
    return [
        float(env.agent.position[0]), float(env.agent.position[1]),
        float(env.block.position[0]), float(env.block.position[1]),
        float(env.block.angle),
    ]


def cuadros(brazo, condicion, seed=SEMILLA, realizacion=REALIZACION):
    """Las tres poses de una celda, con su cobertura, su paso y el rotulo central.

    La columna central es el instante de cobertura maxima, que es el que puntua.
    Cuando el episodio termina por alcanzar el umbral, ese instante **es** el
    ultimo y la columna repetiria imagen; en ese caso se muestra el punto medio
    del episodio y el rotulo cambia en consecuencia.
    """
    ruta = EPISODIOS / f"{realizacion}_{brazo}_{condicion}_seed{seed}.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    traza = datos["traza"]
    k = max(range(1, len(traza)), key=lambda i: traza[i].get("cobertura", 0.0))
    if traza[k]["paso"] == traza[-1]["paso"]:
        k, rotulo = len(traza) // 2, "Paso intermedio"
    else:
        rotulo = "Cobertura máxima"
    return datos, rotulo, [
        {"estado": estado_inicial(seed), "paso": 0, "cobertura": None},
        {"estado": traza[k]["estado"], "paso": traza[k]["paso"],
         "cobertura": traza[k].get("cobertura", 0.0)},
        {"estado": traza[-1]["estado"], "paso": traza[-1]["paso"],
         "cobertura": traza[-1].get("cobertura", 0.0)},
    ]


def resolver_godot(dado=""):
    from barrido_color import resolver_godot as _resolver

    return _resolver(dado)


def imagenes_godot(lista, etiqueta, godot, vista="demo"):
    """Le pide a Godot la vista de cada pose, en una sola ejecucion."""
    import imageio.v2 as imageio

    TMP.mkdir(parents=True, exist_ok=True)
    destinos = [TMP / f"{vista}_{etiqueta}_{i}.png" for i in range(len(lista))]
    plan = TMP / f"cuadros_{vista}_{etiqueta}.json"
    plan.write_text(json.dumps({"cuadros": [
        {"estado": c["estado"], "png": f"res://grabaciones/figuras/{d.name}", "vista": vista}
        for c, d in zip(lista, destinos)
    ]}), encoding="utf-8")

    orden = [godot, "--path", str(GODOT_DIR), "--", "modo=fotogramas",
             f"salida=res://grabaciones/figuras/{plan.name}"]
    salida = subprocess.run(orden, capture_output=True, text=True, timeout=300)
    faltan = [d for d in destinos if not d.is_file()]
    if faltan:
        raise SystemExit(
            f"Godot no escribio {len(faltan)} fotogramas de {etiqueta}.\n"
            f"{salida.stdout[-800:]}\n{salida.stderr[-800:]}"
        )
    return [np.asarray(imageio.imread(d)) for d in destinos]


def recorte_comun(imagenes, margen=18):
    """Caja que encierra la mesa en todas las capturas, con un margen.

    La captura ocupa \num{1280} por \num{720} y la escena solo su parte central,
    de modo que sin recortar la mayor parte de la figura seria fondo. El recorte
    se calcula sobre los seis paneles a la vez: uno por panel har\'ia que la
    camara pareciese moverse entre columnas.
    """
    fondo = np.array([235, 237, 242], dtype=int)   # el color de fondo del entorno
    filas, columnas = [], []
    for imagen in imagenes:
        util = np.abs(imagen[:, :, :3].astype(int) - fondo).max(axis=2) > 6
        if not util.any():
            continue
        ys, xs = np.where(util)
        filas += [ys.min(), ys.max()]
        columnas += [xs.min(), xs.max()]
    if not filas:
        return None
    alto, ancho = imagenes[0].shape[:2]
    return (max(min(filas) - margen, 0), min(max(filas) + margen, alto),
            max(min(columnas) - margen, 0), min(max(columnas) + margen, ancho))


def figura(condicion, godot):
    filas, rotulos = [], set()
    for brazo, etiqueta in BRAZOS:
        datos, rotulo, lista = cuadros(brazo, condicion)
        rotulos.add(rotulo)
        # La vista del simulador no depende de la condicion, solo de la pose,
        # pero las poses si dependen: cada celda tiene su propia trayectoria.
        imagenes = imagenes_godot(lista, f"{brazo}_{condicion}", godot)
        filas.append((etiqueta, datos, lista, imagenes))
    # Los dos artefactos deben coincidir en si el episodio acabo por umbral o por
    # agotar los pasos; si no, la columna central no seria la misma cosa en las
    # dos filas y la figura mentiria.
    assert len(rotulos) == 1, f"las dos filas piden rotulos distintos: {rotulos}"
    columnas = ["Inicio", rotulos.pop(), "Final del episodio"]

    caja = recorte_comun([im for _, _, _, ims in filas for im in ims])

    fig, ejes = plt.subplots(2, 3, figsize=(7.2, 4.0))
    for i, (etiqueta, datos, lista, imagenes) in enumerate(filas):
        for j, (cuadro, imagen) in enumerate(zip(lista, imagenes)):
            eje = ejes[i, j]
            if caja is not None:
                imagen = imagen[caja[0]:caja[1], caja[2]:caja[3]]
            eje.imshow(imagen, interpolation="bilinear")
            eje.set_xticks([])
            eje.set_yticks([])
            for lado in eje.spines.values():
                lado.set_color("#B0B0B0")
                lado.set_linewidth(0.8)
            if cuadro["cobertura"] is None:
                pie = f"paso {cuadro['paso']}"
            else:
                pie = f"paso {cuadro['paso']} · cobertura {cuadro['cobertura']:.3f}"
            eje.set_xlabel(pie, fontsize=8.5, labelpad=3)
            if i == 0:
                eje.set_title(columnas[j], fontsize=10, pad=6)
        ejes[i, 0].set_ylabel(etiqueta, fontsize=12, labelpad=8)
        ejes[i, 2].yaxis.set_label_position("right")
        ejes[i, 2].set_ylabel(
            f"{datos['pasos']} pasos\npuntuación {datos['recompensa_max']:.3f}",
            fontsize=9, labelpad=12, rotation=270, va="bottom",
        )
    fig.tight_layout()
    destino = IMG_DIR / f"godot_paper_condicion_{condicion}.pdf"
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)
    return destino, filas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", default="")
    parser.add_argument("--condiciones", nargs="+", default=["a", "b"])
    args = parser.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    godot = resolver_godot(args.godot)

    print(f"Semilla {SEMILLA}, realizacion {REALIZACION}.")
    for condicion in args.condiciones:
        destino, filas = figura(condicion, godot)
        for etiqueta, datos, lista, _ in filas:
            texto = "  ".join(
                f"[{c['paso']:3d}] " + ("inicio" if c["cobertura"] is None
                                        else f"{c['cobertura']:.4f}")
                for c in lista
            )
            print(f"  {condicion.upper()}  {etiqueta:22} {texto}   "
                  f"puntuacion {datos['recompensa_max']:.4f}")
        print(f"  -> {destino}")


if __name__ == "__main__":
    main()
