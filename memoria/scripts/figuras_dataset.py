#!/usr/bin/env python3
"""Figure and summary statistics of the Push-T demonstration set, for section 3.2.

Reads the two CSV that ``diffuser/scripts/caracterizar_dataset.py`` extracts in
WSL, so the figure can be redrawn on Windows without the ``zarr`` and without
the training environment. Everything the section states in numbers is printed
here as well, to keep the text and the data from drifting apart.

    .venv_diffuser_infer\\Scripts\\python.exe memoria/scripts/figuras_dataset.py

Writes ``memoria/img/caracterizacion_dataset.pdf``.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
DATOS_DIR = ROOT / "memoria" / "datos"
IMG_DIR = ROOT / "memoria" / "img"

COLORES = {
    "entrenamiento": "#0072B2",
    "descarte": "#D55E00",
    "validacion": "#009E73",
    "evaluacion": "#555555",
}
ETIQUETAS = {
    "entrenamiento": "Entrenamiento (90)",
    "descarte": "Descartados (112)",
    "validacion": "Validación (4)",
    "evaluacion": "Evaluación (50)",
}

# Best score of V0, the reference the memoir reports for the trained policy.
PUNTUACION_V0 = 0.8645003451686891


def leer_csv(nombre):
    with open(DATOS_DIR / nombre, encoding="utf-8") as fichero:
        return list(csv.DictReader(fichero))


def columna(filas, clave, tipo=float):
    return np.array([tipo(fila[clave]) for fila in filas])


def style_axes(axis):
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def histograma_comparado(axis, valores, particion, bordes):
    """Filled histogram for the training subset, outlined one for the discard.

    Two filled histograms would overlap into a third colour and hide which
    subset is which. Both are drawn as fractions of their own subset, since 90
    and 112 episodes are not directly comparable as counts.
    """
    entrenamiento = valores[particion == "entrenamiento"]
    descarte = valores[particion == "descarte"]
    axis.hist(
        entrenamiento,
        bins=bordes,
        weights=np.full(len(entrenamiento), 1 / len(entrenamiento)),
        histtype="stepfilled",
        alpha=0.5,
        color=COLORES["entrenamiento"],
        edgecolor=COLORES["entrenamiento"],
        linewidth=1.2,
        label=ETIQUETAS["entrenamiento"],
    )
    axis.hist(
        descarte,
        bins=bordes,
        weights=np.full(len(descarte), 1 / len(descarte)),
        histtype="step",
        color=COLORES["descarte"],
        linewidth=1.4,
        label=ETIQUETAS["descarte"],
    )
    axis.set_ylim(top=axis.get_ylim()[1] * 1.3)


def panel_longitudes(axis, episodios, particion):
    valores = columna(episodios, "transiciones")
    histograma_comparado(
        axis, valores, particion, np.histogram_bin_edges(valores, bins=16))
    axis.set_xlabel("Transiciones por episodio")
    axis.set_ylabel("Fracción de episodios")
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    style_axes(axis)


def panel_puntuaciones(axis, episodios, particion):
    valores = columna(episodios, "puntuacion")
    histograma_comparado(
        axis, valores, particion, np.arange(0.80, 1.0001, 0.0125))
    axis.set_xlim(0.80, 1.005)
    axis.axvline(PUNTUACION_V0, color="#000000", linestyle="--", linewidth=1.2,
                 zorder=5)
    axis.axvline(1.0, color="#000000", linestyle=":", linewidth=1.2, zorder=5)
    transformacion = axis.get_xaxis_transform()
    axis.text(PUNTUACION_V0 - 0.004, 0.96, "V0: 0,864", fontsize=7,
              ha="right", va="top", transform=transformacion)
    axis.text(0.996, 0.96, "Éxito", fontsize=7, rotation=90,
              ha="right", va="top", transform=transformacion)
    axis.set_xlabel("Puntuación del demostrador")
    axis.set_ylabel("Fracción de episodios")
    style_axes(axis)


def panel_posicion(axis, episodios, particion, evaluacion):
    for nombre in ("descarte", "entrenamiento"):
        seleccion = particion == nombre
        axis.scatter(
            columna(episodios, "bloque_x")[seleccion],
            columna(episodios, "bloque_y")[seleccion],
            s=14,
            color=COLORES[nombre],
            alpha=0.7,
            linewidth=0,
            label=ETIQUETAS[nombre],
        )
    conjunto = np.array([fila["conjunto"] for fila in evaluacion])
    seleccion = conjunto == "evaluacion"
    axis.scatter(
        columna(evaluacion, "bloque_x")[seleccion],
        columna(evaluacion, "bloque_y")[seleccion],
        s=24,
        marker="x",
        color=COLORES["evaluacion"],
        linewidth=1.0,
        label=ETIQUETAS["evaluacion"],
    )
    axis.set_xlim(0, 512)
    axis.set_ylim(0, 512)
    axis.set_aspect("equal")
    axis.set_xlabel("Posición inicial de la pieza, $x$ (píxeles)")
    axis.set_ylabel("Posición inicial de la pieza, $y$ (píxeles)")
    axis.legend(fontsize=6.5, loc="upper center", ncol=3,
                bbox_to_anchor=(0.5, -0.14), frameon=False,
                columnspacing=1.0, handletextpad=0.4)
    axis.grid(color="#D9D9D9", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def panel_angulo(axis, episodios, evaluacion):
    bordes = np.linspace(0, 2 * np.pi, 9)
    conjunto = np.array([fila["conjunto"] for fila in evaluacion])
    axis.hist(
        columna(episodios, "bloque_theta"),
        bins=bordes,
        density=True,
        histtype="stepfilled",
        alpha=0.45,
        color=COLORES["entrenamiento"],
        edgecolor=COLORES["entrenamiento"],
        linewidth=1.2,
        label="Demostraciones (206)",
    )
    axis.hist(
        columna(evaluacion, "bloque_theta")[conjunto == "evaluacion"],
        bins=bordes,
        density=True,
        histtype="step",
        color=COLORES["evaluacion"],
        linewidth=1.4,
        label=ETIQUETAS["evaluacion"],
    )
    axis.set_ylim(top=axis.get_ylim()[1] * 1.35)
    axis.set_xlim(0, 2 * np.pi)
    axis.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    axis.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axis.set_xlabel("Orientación inicial de la pieza (rad)")
    axis.set_ylabel("Densidad")
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    style_axes(axis)


def resumen(episodios, particion, evaluacion):
    longitud = columna(episodios, "transiciones")
    puntuacion = columna(episodios, "puntuacion")
    cobertura = columna(episodios, "cobertura_max")

    print("subconjunto      n  transiciones  long. media  duracion (min)  "
          "puntuacion media")
    for nombre in ("entrenamiento", "validacion", "descarte"):
        seleccion = particion == nombre
        print(f"{nombre:<14} {seleccion.sum():3d}  {int(longitud[seleccion].sum()):12d}"
              f"  {longitud[seleccion].mean():11.1f}"
              f"  {longitud[seleccion].sum() / 10 / 60:14.1f}"
              f"  {puntuacion[seleccion].mean():16.3f}")
    print(f"{'total':<14} {len(longitud):3d}  {int(longitud.sum()):12d}"
          f"  {longitud.mean():11.1f}  {longitud.sum() / 10 / 60:14.1f}"
          f"  {puntuacion.mean():16.3f}")

    print()
    print(f"puntuacion del demostrador: media {puntuacion.mean():.3f}, "
          f"mediana {np.median(puntuacion):.3f}, "
          f"recorrido {puntuacion.min():.3f}-{puntuacion.max():.3f}")
    print(f"cobertura maxima          : media {cobertura.mean():.3f}, "
          f"maxima {cobertura.max():.3f}; "
          f"episodios que alcanzan 0,95: {int((cobertura >= 0.95).sum())}")
    print(f"V0 (0,864) frente a la media del demostrador: "
          f"{100 * PUNTUACION_V0 / puntuacion.mean():.1f} %")

    print()
    entrenamiento = particion == "entrenamiento"
    descarte = particion == "descarte"
    for etiqueta, valores in (("longitud", longitud), ("puntuacion", puntuacion)):
        mwu = mannwhitneyu(valores[entrenamiento], valores[descarte]).pvalue
        ks = ks_2samp(valores[entrenamiento], valores[descarte]).pvalue
        print(f"entrenamiento vs descarte, {etiqueta:<10}: "
              f"Mann-Whitney p = {mwu:.3f}, Kolmogorov-Smirnov p = {ks:.3f}")

    conjunto = np.array([fila["conjunto"] for fila in evaluacion])
    seleccion = conjunto == "evaluacion"
    for clave in ("bloque_x", "bloque_y", "bloque_theta", "agente_x", "agente_y"):
        ks = ks_2samp(
            columna(episodios, clave)[entrenamiento],
            columna(evaluacion, clave)[seleccion]).pvalue
        print(f"entrenamiento vs evaluacion, {clave:<13}: "
              f"Kolmogorov-Smirnov p = {ks:.3f}")


def main():
    episodios = leer_csv("demostraciones_episodios.csv")
    evaluacion = leer_csv("condiciones_evaluacion.csv")
    particion = np.array([fila["particion"] for fila in episodios])

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4), constrained_layout=True)
    panel_longitudes(axes[0][0], episodios, particion)
    panel_puntuaciones(axes[0][1], episodios, particion)
    panel_posicion(axes[1][0], episodios, particion, evaluacion)
    panel_angulo(axes[1][1], episodios, evaluacion)
    for axis, letra in zip(axes.ravel(), "abcd"):
        axis.set_title(f"({letra})", loc="left", fontsize=9)
    destino = IMG_DIR / "caracterizacion_dataset.pdf"
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)

    resumen(episodios, particion, evaluacion)
    print(f"\nfigura escrita en {destino}")


if __name__ == "__main__":
    main()
