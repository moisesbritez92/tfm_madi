#!/usr/bin/env python3
"""Reanalisis del bloque de prueba final, declarado post hoc.

El preregistro (``memoria/preregistro_prueba_final.md``) fijo Wilcoxon pareado como
contraste del endpoint primario. La reevaluacion externa objeta, con razon, que el
estimando declarado es una *diferencia de medias* y que Wilcoxon no contrasta esa
media: opera sobre los rangos de las diferencias pareadas. Este script anade el
contraste que si apunta al estimando y homogeneiza los intervalos.

**Estado de estos analisis: post hoc.** No estaban en el preregistro. El punto 6 del
preregistro obliga a declararlo, y asi se hace. No sustituyen a los preregistrados:
se reportan junto a ellos.

Que anade, y a que hallazgo responde:

  * B1 (M6) prueba de permutacion sobre la diferencia media pareada, por inversion
    de signo (10 000 permutaciones, semilla 42). Es el contraste exacto para la
    hipotesis nula de simetria de las diferencias en torno a cero, y su estadistico
    es la media, no un rango. Se acompana de Wilcoxon, tal como se preregistro.
    Familia de las diez comparaciones, correccion de Holm.
  * B2 (M6) intervalo BCa por variante, con el mismo metodo y numero de remuestreos
    que los intervalos de las diferencias. La tabla anterior rotulaba un intervalo
    normal de media +- 1,96 EE sin declarar el metodo.
  * B3 (m5) intervalo de Wilson al 95 % para la tasa de exito. Endpoint secundario y
    descriptivo: no entra en la familia de Holm, y asi se declara en la memoria.

Entrada:  datos/prueba_final_episodios.csv  (200 puntuaciones x 5 variantes)
Salida:   datos/prueba_final_resumen_v2.csv
          datos/prueba_final_contrastes_v2.csv

    python memoria/scripts/analisis_prueba_final_v2.py
"""

import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, norm, wilcoxon

DATOS = Path(__file__).resolve().parents[1] / "datos"
ENTRADA = DATOS / "prueba_final_episodios.csv"

VARIANTES = ["V0", "V1", "V2", "V3", "V4"]
UMBRAL_EXITO = 0.999
N_REMUESTREOS = 10000
N_PERMUTACIONES = 10000
SEMILLA = 42
NIVEL = 0.95


def cargar():
    """Devuelve {variante: array de 200 puntuaciones ordenadas por semilla}."""
    por_variante = {v: {} for v in VARIANTES}
    with ENTRADA.open(encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            por_variante[fila["variante"]][int(fila["semilla"])] = float(
                fila["puntuacion"]
            )
    semillas = sorted(por_variante["V0"])
    assert semillas == list(range(200000, 200200)), "el bloque no es el preregistrado"
    salida = {}
    for v in VARIANTES:
        assert sorted(por_variante[v]) == semillas, f"{v}: semillas distintas"
        salida[v] = np.array([por_variante[v][s] for s in semillas])
    return semillas, salida


def ic_bca(muestra, rng):
    """Intervalo BCa de la media, mismo metodo que el de las diferencias."""
    res = bootstrap(
        (muestra,),
        np.mean,
        n_resamples=N_REMUESTREOS,
        confidence_level=NIVEL,
        method="BCa",
        random_state=rng,
    )
    return res.confidence_interval.low, res.confidence_interval.high


def ic_wilson(exitos, n, nivel=NIVEL):
    """Intervalo de Wilson para una proporcion binomial.

    Se elige Wilson y no el normal de Wald porque la tasa de V4 (0,08) esta lo
    bastante cerca de cero para que Wald produzca cobertura pobre.
    """
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - (1 - nivel) / 2)
    p = exitos / n
    denom = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denom
    radio = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centro - radio, centro + radio


def permutacion_media(dif, rng):
    """Prueba de permutacion por inversion de signo sobre la diferencia media.

    Bajo la nula de que las diferencias pareadas son simetricas en torno a cero,
    cambiar el signo de cualquier subconjunto deja la distribucion invariante. Se
    remuestrea el signo de las 200 diferencias y se compara la media observada con
    la distribucion nula. Bilateral, con el estimador de p con correccion +1 que
    evita informar p = 0 con un numero finito de permutaciones.
    """
    observado = float(np.mean(dif))
    n = dif.size
    signos = rng.choice([-1.0, 1.0], size=(N_PERMUTACIONES, n))
    nulas = (signos * dif).mean(axis=1)
    extremos = int(np.sum(np.abs(nulas) >= abs(observado) - 1e-15))
    return observado, (extremos + 1) / (N_PERMUTACIONES + 1)


def holm(pares_p):
    """Correccion secuencial de Holm. Devuelve {clave: p ajustado}."""
    ordenados = sorted(pares_p.items(), key=lambda kv: kv[1])
    m = len(ordenados)
    ajustados = {}
    previo = 0.0
    for i, (clave, p) in enumerate(ordenados):
        valor = min(1.0, (m - i) * p)
        valor = max(valor, previo)          # monotonia
        ajustados[clave] = valor
        previo = valor
    return ajustados


def main():
    semillas, datos = cargar()
    rng = np.random.default_rng(SEMILLA)

    # ---- B2 y B3: resumen por variante ----------------------------------
    filas_resumen = []
    for v in VARIANTES:
        x = datos[v]
        exitos = int(np.sum(x >= UMBRAL_EXITO))
        bca_inf, bca_sup = ic_bca(x, rng)
        wil_inf, wil_sup = ic_wilson(exitos, x.size)
        filas_resumen.append(
            {
                "variante": v,
                "n": x.size,
                "media": round(float(np.mean(x)), 6),
                "desviacion": round(float(np.std(x, ddof=1)), 6),
                "error_estandar": round(float(np.std(x, ddof=1) / np.sqrt(x.size)), 6),
                "ic95_bca_inf": round(bca_inf, 6),
                "ic95_bca_sup": round(bca_sup, 6),
                "exitos": exitos,
                "tasa_exito": round(exitos / x.size, 6),
                "wilson95_inf": round(wil_inf, 6),
                "wilson95_sup": round(wil_sup, 6),
            }
        )

    # ---- B1: contrastes ---------------------------------------------------
    filas_contrastes = []
    p_perm, p_wilc = {}, {}
    for a, b in itertools.combinations(VARIANTES, 2):
        clave = f"{a}-{b}"
        dif = datos[a] - datos[b]
        media, p = permutacion_media(dif, rng)
        no_nulos = int(np.sum(dif != 0))
        w, pw = wilcoxon(dif, zero_method="wilcox", correction=True, mode="approx")
        p_perm[clave], p_wilc[clave] = p, float(pw)
        filas_contrastes.append(
            {
                "comparacion": clave,
                "diferencia_medias": round(media, 6),
                "error_estandar": round(
                    float(np.std(dif, ddof=1) / np.sqrt(dif.size)), 6
                ),
                "pares_no_nulos": no_nulos,
                "p_permutacion": p,
                "W": float(w),
                "p_wilcoxon": float(pw),
            }
        )

    holm_perm = holm(p_perm)
    holm_wilc = holm(p_wilc)
    for fila in filas_contrastes:
        fila["p_perm_holm"] = holm_perm[fila["comparacion"]]
        fila["p_wilcoxon_holm"] = holm_wilc[fila["comparacion"]]

    escribir(DATOS / "prueba_final_resumen_v2.csv", filas_resumen)
    escribir(DATOS / "prueba_final_contrastes_v2.csv", filas_contrastes)

    # ---- informe en pantalla ---------------------------------------------
    print(f"n = {len(semillas)} condiciones, semillas "
          f"{semillas[0]}-{semillas[-1]}\n")
    print("Resumen por variante (IC BCa de la media, Wilson de la tasa)")
    print(f"{'':4} {'media':>7} {'IC95 BCa':>18} {'exito':>7} {'IC95 Wilson':>18}")
    for f in filas_resumen:
        print(
            f"{f['variante']:4} {f['media']:7.3f} "
            f"[{f['ic95_bca_inf']:.3f}, {f['ic95_bca_sup']:.3f}]".ljust(33)
            + f" {f['tasa_exito']:6.3f} "
            f"[{f['wilson95_inf']:.3f}, {f['wilson95_sup']:.3f}]"
        )

    print("\nContrastes de la diferencia media (permutacion) frente a Wilcoxon")
    print(f"{'par':7} {'dif':>7} {'p_perm':>10} {'Holm':>10} "
          f"{'p_wilc':>10} {'Holm':>10}")
    for f in filas_contrastes:
        print(
            f"{f['comparacion']:7} {f['diferencia_medias']:7.3f} "
            f"{f['p_permutacion']:10.3g} {f['p_perm_holm']:10.3g} "
            f"{f['p_wilcoxon']:10.3g} {f['p_wilcoxon_holm']:10.3g}"
        )

    print(f"\nEscritos:\n  {DATOS / 'prueba_final_resumen_v2.csv'}"
          f"\n  {DATOS / 'prueba_final_contrastes_v2.csv'}")


def escribir(path, filas):
    with path.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


if __name__ == "__main__":
    main()
