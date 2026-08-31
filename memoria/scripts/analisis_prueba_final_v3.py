#!/usr/bin/env python3
"""Prueba final con el ruido de difusion integrado en el estimando.

**Estado: post hoc respecto de `memoria/preregistro_prueba_final.md`**, que fijo una
sola trayectoria de difusion por condicion. Este script promedia **dos** realizaciones
del ruido dentro de cada condicion antes de agregar y contrastar. El punto 6 de aquel
preregistro obliga a declararlo, y asi se hace.

Lo que sostiene el cambio, y conviene no confundirlo con una eleccion oportunista:

  * El estimando que integra condicion **y** ruido no se inventa aqui. Estaba
    declarado en `memoria/preregistro_comparacion_paper.md`, cerrado y comprometido
    antes de observar dato alguno del brazo nuevo. Este script aplica ese mismo
    principio, ya registrado, a las cinco variantes.
  * Se aplica de forma **uniforme**: las cinco variantes y los diez contrastes, con
    la misma segunda semilla base. No se selecciona ninguna comparacion en funcion
    de su resultado.
  * El motivo es una medida, no una intuicion: cambiar solo la semilla del ruido
    movio la media de V0 de 0,8719 a 0,9019, magnitud comparable a las diferencias
    que separan a V1, V2 y V3 entre si.

Sigue siendo un cambio de estimando posterior a la observacion de los datos, y esa
limitacion se declara en la memoria junto a las cifras preregistradas, que quedan
intactas en `prueba_final_resumen.csv` y `prueba_final_contrastes.csv`.

Dos decisiones de metodo propias de promediar realizaciones:

  * **Unidad de observacion.** Las dos realizaciones se promedian dentro de cada
    condicion, de modo que el analisis opera sobre 200 diferencias independientes y
    no sobre 400 observaciones anidadas.
  * **Tasa de exito.** El exito es binario y el promedio de dos indicadores no es un
    indicador, asi que no se umbraliza la puntuacion promediada. Se estima la
    probabilidad de exito integrando el ruido, es decir la media sobre las 200
    condiciones de la fraccion de realizaciones con exito, y su intervalo procede de
    un bootstrap BCa sobre las condiciones. El intervalo de Wilson del analisis
    anterior no valdria aqui, porque las 400 observaciones no son independientes.

Entrada:  logs_entrenamiento/prueba_final/{prueba,ruido_b}_v{0..4}.json
Salida:   datos/prueba_final_episodios_v3.csv
          datos/prueba_final_resumen_v3.csv
          datos/prueba_final_contrastes_v3.csv

    python memoria/scripts/analisis_prueba_final_v3.py
"""

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analisis_prueba_final_v2 import holm, permutacion_media
from analisis_seleccion import UMBRAL_EXITO

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "logs_entrenamiento" / "prueba_final"
DATOS = ROOT / "memoria" / "datos"

VARIANTES = ["v0", "v1", "v2", "v3", "v4"]
ETIQUETAS = {"prueba": "A", "ruido_b": "B"}
SEMILLAS_BASE = {"prueba": 20260827, "ruido_b": 20260831}

N_ESPERADO = 200
SEMILLA_INICIAL = 200000
N_REMUESTREOS = 10000
SEMILLA = 42


def cargar(variante, etiqueta):
    path = RAW_DIR / f"{etiqueta}_{variante}.json"
    if not path.is_file():
        raise SystemExit(
            f"falta {path}. Ejecuta antes, desde WSL:\n"
            "  bash diffuser/scripts/evaluar_segunda_realizacion.sh"
        )
    datos = json.loads(path.read_text(encoding="utf-8"))
    semillas = [int(s) for s in datos["puntuaciones"]]
    esperadas = list(range(SEMILLA_INICIAL, SEMILLA_INICIAL + N_ESPERADO))
    assert semillas == esperadas, f"{path.name}: las semillas no son el bloque preregistrado"
    assert datos["n_test"] == N_ESPERADO, f"{path.name}: n_test es {datos['n_test']}"
    assert datos["base_seed_difusion"] == SEMILLAS_BASE[etiqueta], (
        f"{path.name}: semilla base {datos['base_seed_difusion']}, se esperaba "
        f"{SEMILLAS_BASE[etiqueta]}"
    )
    datos["scores"] = np.array([datos["puntuaciones"][str(s)] for s in semillas], dtype=float)
    datos["semillas"] = semillas
    return datos


def ic_bca(muestra, nivel=0.95):
    muestra = np.asarray(muestra, dtype=float)
    if np.allclose(muestra, muestra[0]):
        return float(muestra[0]), float(muestra[0])
    resultado = bootstrap(
        (muestra,),
        np.mean,
        n_resamples=N_REMUESTREOS,
        confidence_level=nivel,
        method="BCa",
        random_state=np.random.default_rng(SEMILLA),
    )
    return (
        float(resultado.confidence_interval.low),
        float(resultado.confidence_interval.high),
    )


def escribir(path, filas):
    with path.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def main():
    crudo = {
        (v, e): cargar(v, e) for v in VARIANTES for e in ETIQUETAS
    }
    semillas = crudo[("v0", "prueba")]["semillas"]

    # Comprobacion de protocolo: dentro de cada realizacion las cinco variantes
    # comparten evaluador y semilla base, que es lo que hace legitimo el pareo.
    claves = ("legacy_test", "max_steps", "n_obs_steps", "n_action_steps", "n_envs")
    referencia = crudo[("v0", "prueba")]["runner"]
    for (v, e), d in crudo.items():
        for clave in claves:
            assert d["runner"][clave] == referencia[clave], (
                f"{e}_{v}: el evaluador difiere en {clave}"
            )

    promedio, exito_medio = {}, {}
    filas_episodios, filas_resumen = [], []
    for v in VARIANTES:
        a = crudo[(v, "prueba")]["scores"]
        b = crudo[(v, "ruido_b")]["scores"]
        promedio[v] = (a + b) / 2.0
        # Fraccion de realizaciones con exito en cada condicion: 0, 0,5 o 1.
        exito_medio[v] = ((a >= UMBRAL_EXITO).astype(float)
                          + (b >= UMBRAL_EXITO).astype(float)) / 2.0

        for etiqueta, scores in (("prueba", a), ("ruido_b", b)):
            for semilla, valor in zip(semillas, scores):
                filas_episodios.append(
                    {
                        "variante": v.upper(),
                        "realizacion": ETIQUETAS[etiqueta],
                        "semilla_base": SEMILLAS_BASE[etiqueta],
                        "semilla": semilla,
                        "puntuacion": valor,
                    }
                )

        x = promedio[v]
        inf, sup = ic_bca(x)
        exito_inf, exito_sup = ic_bca(exito_medio[v])
        # Varianza intracondicion entre realizaciones, promediada sobre condiciones.
        intra = float(np.mean(np.var(np.stack([a, b]), axis=0, ddof=1)))
        filas_resumen.append(
            {
                "variante": v.upper(),
                "n": int(x.size),
                "media_a": round(float(np.mean(a)), 6),
                "media_b": round(float(np.mean(b)), 6),
                "media": round(float(np.mean(x)), 6),
                "desviacion": round(float(np.std(x, ddof=1)), 6),
                "error_estandar": round(float(np.std(x, ddof=1) / np.sqrt(x.size)), 6),
                "ic95_bca_inf": round(inf, 6),
                "ic95_bca_sup": round(sup, 6),
                "exitos_a": int(np.sum(a >= UMBRAL_EXITO)),
                "exitos_b": int(np.sum(b >= UMBRAL_EXITO)),
                "tasa_exito": round(float(np.mean(exito_medio[v])), 6),
                "exito_bca_inf": round(exito_inf, 6),
                "exito_bca_sup": round(exito_sup, 6),
                "varianza_ruido_intra": round(intra, 8),
                "media_preregistrada": round(float(np.mean(a)), 6),
            }
        )

    filas_contrastes = []
    p_perm, p_wilc = {}, {}
    for a, b in itertools.combinations(VARIANTES, 2):
        clave = f"{a.upper()}-{b.upper()}"
        dif = promedio[a] - promedio[b]
        media, p = permutacion_media(dif, np.random.default_rng(SEMILLA))
        inf, sup = ic_bca(dif)
        w, pw = wilcoxon(dif, zero_method="wilcox", correction=True, method="approx")
        p_perm[clave], p_wilc[clave] = p, float(pw)
        filas_contrastes.append(
            {
                "comparacion": clave,
                "diferencia_medias": round(media, 6),
                "error_estandar": round(float(np.std(dif, ddof=1) / np.sqrt(dif.size)), 6),
                "ic95_bca_inf": round(inf, 6),
                "ic95_bca_sup": round(sup, 6),
                "pares_no_nulos": int(np.sum(dif != 0)),
                "p_permutacion": p,
                "W": float(w),
                "p_wilcoxon": float(pw),
            }
        )

    holm_perm, holm_wilc = holm(p_perm), holm(p_wilc)
    for fila in filas_contrastes:
        fila["p_perm_holm"] = holm_perm[fila["comparacion"]]
        fila["p_wilcoxon_holm"] = holm_wilc[fila["comparacion"]]

    escribir(DATOS / "prueba_final_episodios_v3.csv", filas_episodios)
    escribir(DATOS / "prueba_final_resumen_v3.csv", filas_resumen)
    escribir(DATOS / "prueba_final_contrastes_v3.csv", filas_contrastes)

    print(f"n = {len(semillas)} condiciones, semillas {semillas[0]}-{semillas[-1]}, "
          f"dos realizaciones del ruido por variante.\n")
    print(f"{'':4} {'A':>7} {'B':>7} {'media':>7} {'IC95 BCa':>18} "
          f"{'exito':>7} {'var.ruido':>10}")
    for f in filas_resumen:
        print(
            f"{f['variante']:4} {f['media_a']:7.3f} {f['media_b']:7.3f} "
            f"{f['media']:7.3f} "
            + f"[{f['ic95_bca_inf']:.3f}, {f['ic95_bca_sup']:.3f}]".rjust(18)
            + f" {f['tasa_exito']:7.3f} {f['varianza_ruido_intra']:10.4f}"
        )

    print("\nContrastes sobre la media de las dos realizaciones")
    print(f"{'par':9} {'dif':>7} {'IC95 BCa':>20} {'p_perm':>9} {'Holm':>9} "
          f"{'p_wilc':>9} {'Holm':>9}")
    for f in filas_contrastes:
        print(
            f"{f['comparacion']:9} {f['diferencia_medias']:7.3f} "
            + f"[{f['ic95_bca_inf']:+.3f}, {f['ic95_bca_sup']:+.3f}]".rjust(20)
            + f" {f['p_permutacion']:9.3g} {f['p_perm_holm']:9.3g} "
            f"{f['p_wilcoxon']:9.3g} {f['p_wilcoxon_holm']:9.3g}"
        )

    print(f"\nEscritos en {DATOS}")


if __name__ == "__main__":
    main()
