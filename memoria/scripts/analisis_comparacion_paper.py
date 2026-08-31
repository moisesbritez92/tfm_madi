#!/usr/bin/env python3
"""Contraste preregistrado entre V0 y el punto de control publicado del articulo.

El protocolo esta fijado por adelantado en
``memoria/preregistro_comparacion_paper.md``. Este script no decide nada: ejecuta
lo que ese documento declara, y nada mas.

Un solo contraste primario, entre dos brazos, sobre un solo endpoint primario, de
modo que **no hay multiplicidad que corregir**. La familia de diez comparaciones de
la prueba final no se toca: este script no escribe ninguno de sus ficheros.

  * Endpoint primario: media de las 200 diferencias pareadas por condicion
    inicial, con las dos realizaciones de ruido promediadas dentro de cada
    condicion. El estimando integra la condicion y el ruido; esa es la respuesta al
    Hallazgo M5.
  * Contraste primario: permutacion por inversion de signo sobre la diferencia
    media, bilateral, 10 000 permutaciones, semilla 42. Se importa la funcion de
    ``analisis_prueba_final_v2``, no se copia.
  * Equivalencia: TOST con margen 0,05, resuelto como un intervalo BCa al 90 %
    contenido en (-0,05, +0,05). Responde al Hallazgo M9.
  * Secundarios, sin correccion y etiquetados como tales: tasa de exito por
    realizacion con McNemar exacto, y la componente de varianza del ruido.

Entrada:  logs_entrenamiento/prueba_final/{prueba,ruido_b}_{v0,paper}.json
Salida:   datos/comparacion_paper_episodios.csv
          datos/comparacion_paper_resumen.csv
          datos/comparacion_paper_contrastes.csv

    python memoria/scripts/analisis_comparacion_paper.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, bootstrap, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analisis_dispersion import describe
from analisis_prueba_final import ic_bca
from analisis_prueba_final_v2 import permutacion_media
from analisis_seleccion import UMBRAL_EXITO

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "logs_entrenamiento" / "prueba_final"
OUT_DIR = ROOT / "memoria" / "datos"

N_ESPERADO = 200
SEMILLA_INICIAL = 200000
N_REMUESTREOS = 10000
SEMILLA = 42

# Las dos realizaciones de ruido, con la semilla base declarada en el preregistro.
REALIZACIONES = {"prueba": 20260827, "ruido_b": 20260831}
BRAZOS = {"v0": "V0", "paper": "V_PAPER"}

# Margen de equivalencia, fijado en el preregistro antes de ver los datos.
DELTA = 0.05
NIVEL_TOST = 0.90

# Puntos de control congelados en el preregistro, por SHA-256.
SHA_ESPERADO = {
    "v0": "5310551ee71075d9efcf956c34670809741d84e06808809551e7675674e8ce63",
    "paper": "bac7221f7e34cd51162dc1972e1a39ffcddc87de1dc1780c44ffa61b88c4ff76",
}


def cargar(brazo, realizacion):
    path = RAW_DIR / f"{realizacion}_{brazo}.json"
    if not path.is_file():
        raise SystemExit(
            f"falta {path}. Ejecuta antes, desde WSL:\n"
            "  bash diffuser/scripts/evaluar_paper_bloque_test.sh --portones\n"
            "  bash diffuser/scripts/evaluar_paper_bloque_test.sh"
        )
    datos = json.loads(path.read_text(encoding="utf-8"))
    semillas = [int(s) for s in datos["puntuaciones"]]
    esperadas = list(range(SEMILLA_INICIAL, SEMILLA_INICIAL + N_ESPERADO))
    assert semillas == esperadas, f"{path.name}: las semillas no son el bloque preregistrado"
    assert datos["n_test"] == N_ESPERADO, f"{path.name}: n_test es {datos['n_test']}"
    assert datos["base_seed_difusion"] == REALIZACIONES[realizacion], (
        f"{path.name}: semilla base {datos['base_seed_difusion']}, se esperaba "
        f"{REALIZACIONES[realizacion]}"
    )
    assert datos["sha256"] == SHA_ESPERADO[brazo], (
        f"{path.name}: el punto de control no es el congelado en el preregistro"
    )
    datos["scores"] = np.array([datos["puntuaciones"][str(s)] for s in semillas], dtype=float)
    datos["semillas"] = semillas
    return datos


def ic_bca_nivel(diferencias, nivel):
    """BCa de la media al nivel pedido. ic_bca del preregistro anterior fija 95 %."""
    muestra = np.asarray(diferencias, dtype=float)
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


def mcnemar_exacto(a, b):
    """Prueba exacta de McNemar sobre dos vectores de exito/fracaso pareados.

    Condicionada a los pares discordantes, la nula es que un discordante cae de
    cada lado con probabilidad un medio. Es una binomial exacta, que con pocos
    discordantes es lo correcto y con muchos coincide con la chi cuadrado.
    """
    solo_a = int(np.sum(a & ~b))
    solo_b = int(np.sum(~a & b))
    discordantes = solo_a + solo_b
    if discordantes == 0:
        return solo_a, solo_b, 1.0
    p = binomtest(solo_a, discordantes, 0.5, alternative="two-sided").pvalue
    return solo_a, solo_b, float(p)


def decision(p, inf90, sup90):
    """Las cuatro casillas de la regla de decision, escritas en el preregistro."""
    dentro = inf90 > -DELTA and sup90 < DELTA
    if dentro and p >= 0.05:
        return "equivalencia practica"
    if dentro and p < 0.05:
        return "diferencia detectable pero practicamente irrelevante"
    if not dentro and p < 0.05:
        return "diferencia relevante"
    return "indeterminado, potencia insuficiente"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datos = {
        (brazo, realizacion): cargar(brazo, realizacion)
        for brazo in BRAZOS
        for realizacion in REALIZACIONES
    }
    semillas = datos[("v0", "prueba")]["semillas"]

    # Episodios en bruto, 800 filas: dos brazos por dos realizaciones por 200.
    with (OUT_DIR / "comparacion_paper_episodios.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["brazo", "realizacion", "semilla_base", "semilla", "puntuacion"])
        for (brazo, realizacion), d in sorted(datos.items()):
            for semilla, valor in zip(d["semillas"], d["scores"]):
                escritor.writerow(
                    [BRAZOS[brazo], realizacion, d["base_seed_difusion"], semilla, valor]
                )

    # Resumen por brazo y realizacion, mas la media de las dos realizaciones.
    filas_resumen = []
    promedio = {}
    for brazo in BRAZOS:
        por_realizacion = [datos[(brazo, r)]["scores"] for r in REALIZACIONES]
        promedio[brazo] = np.mean(por_realizacion, axis=0)
        for realizacion in REALIZACIONES:
            d = datos[(brazo, realizacion)]
            resumen = describe(list(d["scores"]))
            exitos = int(np.sum(d["scores"] >= UMBRAL_EXITO))
            filas_resumen.append(
                {
                    "brazo": BRAZOS[brazo],
                    "realizacion": realizacion,
                    "semilla_base": d["base_seed_difusion"],
                    "punto_control": d["punto_control"],
                    "n": resumen["n"],
                    "media": round(resumen["media"], 6),
                    "desviacion": round(resumen["desviacion"], 6),
                    "error_estandar": round(resumen["error_estandar"], 6),
                    "exitos": exitos,
                    "tasa_exito": round(exitos / N_ESPERADO, 6),
                    "segundos": d["segundos"],
                }
            )
        # Componente de varianza del ruido: varianza intra-condicion entre las dos
        # realizaciones, promediada sobre las condiciones. Endpoint secundario 2.
        intra = np.var(np.stack(por_realizacion, axis=0), axis=0, ddof=1)
        filas_resumen.append(
            {
                "brazo": BRAZOS[brazo],
                "realizacion": "media de las dos",
                "semilla_base": "",
                "punto_control": datos[(brazo, "prueba")]["punto_control"],
                "n": N_ESPERADO,
                "media": round(float(np.mean(promedio[brazo])), 6),
                "desviacion": round(float(np.std(promedio[brazo], ddof=1)), 6),
                "error_estandar": round(
                    float(np.std(promedio[brazo], ddof=1) / np.sqrt(N_ESPERADO)), 6
                ),
                "exitos": "",
                "tasa_exito": "",
                "segundos": "",
                "varianza_ruido_intra": round(float(np.mean(intra)), 8),
            }
        )

    campos = [
        "brazo", "realizacion", "semilla_base", "punto_control", "n", "media",
        "desviacion", "error_estandar", "exitos", "tasa_exito", "segundos",
        "varianza_ruido_intra",
    ]
    with (OUT_DIR / "comparacion_paper_resumen.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        escritor = csv.DictWriter(fh, fieldnames=campos, restval="")
        escritor.writeheader()
        escritor.writerows(filas_resumen)

    # --- Contraste primario ---------------------------------------------------
    diferencias = promedio["v0"] - promedio["paper"]
    assert diferencias.size == N_ESPERADO

    pareado = describe(list(diferencias))
    observado, p_perm = permutacion_media(diferencias, np.random.default_rng(SEMILLA))
    inf95, sup95 = ic_bca(diferencias)
    inf90, sup90 = ic_bca_nivel(diferencias, NIVEL_TOST)
    equivalencia = inf90 > -DELTA and sup90 < DELTA
    veredicto = decision(p_perm, inf90, sup90)

    no_nulos = int(np.sum(diferencias != 0))
    w, p_wilcoxon = wilcoxon(
        promedio["v0"], promedio["paper"],
        zero_method="wilcox", method="approx", correction=True,
    )

    fila = {
        "comparacion": "V0-V_PAPER",
        "n": N_ESPERADO,
        "realizaciones_ruido": len(REALIZACIONES),
        "diferencia_medias": round(observado, 6),
        "error_estandar": round(pareado["error_estandar"], 6),
        "ic95_bca_inf": round(inf95, 6),
        "ic95_bca_sup": round(sup95, 6),
        "ic90_bca_inf": round(inf90, 6),
        "ic90_bca_sup": round(sup90, 6),
        "delta_equivalencia": DELTA,
        "equivalencia": equivalencia,
        "p_permutacion": p_perm,
        "decision": veredicto,
        "pares_no_nulos": no_nulos,
        "W": w,
        "p_wilcoxon": float(p_wilcoxon),
        "gana_v0": int(np.sum(diferencias > 0)),
        "gana_paper": int(np.sum(diferencias < 0)),
    }

    # --- Secundario 1: tasa de exito por realizacion, McNemar exacto ----------
    for realizacion in REALIZACIONES:
        exito_v0 = datos[("v0", realizacion)]["scores"] >= UMBRAL_EXITO
        exito_paper = datos[("paper", realizacion)]["scores"] >= UMBRAL_EXITO
        solo_v0, solo_paper, p = mcnemar_exacto(exito_v0, exito_paper)
        fila[f"mcnemar_{realizacion}_solo_v0"] = solo_v0
        fila[f"mcnemar_{realizacion}_solo_paper"] = solo_paper
        fila[f"mcnemar_{realizacion}_p"] = p

    with (OUT_DIR / "comparacion_paper_contrastes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(fila))
        escritor.writeheader()
        escritor.writerow(fila)

    # --- Informe --------------------------------------------------------------
    print(f"Bloque {semillas[0]}-{semillas[-1]}, n = {N_ESPERADO}, "
          f"{len(REALIZACIONES)} realizaciones de ruido por brazo.\n")
    for f in filas_resumen:
        if f["realizacion"] == "media de las dos":
            print(f"  {f['brazo']:<8} media de las dos realizaciones: {f['media']:.4f} "
                  f"· varianza de ruido intra-condicion {f['varianza_ruido_intra']:.6f}")
        else:
            print(f"  {f['brazo']:<8} {f['realizacion']:<8} media {f['media']:.4f} "
                  f"· EE {f['error_estandar']:.4f} · exito {f['exitos']}/{N_ESPERADO}")

    print("\nContraste primario, V0 - V_PAPER:")
    print(f"  diferencia media   {observado:+.4f}  (EE {pareado['error_estandar']:.4f})")
    print(f"  IC95 BCa           [{inf95:+.4f}, {sup95:+.4f}]")
    print(f"  IC90 BCa (TOST)    [{inf90:+.4f}, {sup90:+.4f}]  frente a delta = {DELTA}")
    print(f"  permutacion        p = {p_perm:.6g}   ({fila['gana_v0']} condiciones a favor "
          f"de V0, {fila['gana_paper']} a favor de V_Paper)")
    print(f"  Wilcoxon (robustez) W = {w:g}, p = {p_wilcoxon:.6g}")
    print(f"  equivalencia practica: {'SI' if equivalencia else 'NO'}")
    print(f"\n  DECISION: {veredicto}")

    print("\nSecundario, tasa de exito (McNemar exacto, sin correccion de multiplicidad):")
    for realizacion in REALIZACIONES:
        print(f"  {realizacion:<8} solo V0 {fila[f'mcnemar_{realizacion}_solo_v0']:>3} · "
              f"solo V_Paper {fila[f'mcnemar_{realizacion}_solo_paper']:>3} · "
              f"p = {fila[f'mcnemar_{realizacion}_p']:.6g}")

    print(f"\nEscrito en {OUT_DIR}")


if __name__ == "__main__":
    main()
