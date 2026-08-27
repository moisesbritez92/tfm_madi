#!/usr/bin/env python3
"""Selection-free readings of the rollout logs, and how optimistic the reported maxima are.

Finding C1 of the external review says the reported score of each variant is the
maximum over ``K`` noisy evaluations of the *same* 50 episodes that later produce
the confidence interval and the p-value. Two defects hide under that label and
they need different answers:

  (i)  the maximum of K noisy estimates is biased upwards, and K was not equal
       across variants (10, 10, 6, 4, 4), so the bias is differential;
  (ii) there is no independent test block.

This script attacks (i) with the logs already on disk, no GPU involved. Defect
(ii) needs the disjoint seed block of ``diffuser/scripts/evaluar_bloque_test.py``.

Three readings, each written to ``memoria/datos/``:

  1. Fixed common epoch. Epoch 150 is the last one the five variants evaluated,
     so comparing them there is a rule that does not look at the scores: no
     maximum, no winner curse. The epoch is chosen by budget, never by result.
  2. Equal opportunities. The same argmax rule restricted to the first four
     evaluations of every variant, which is what V3 and V4 actually had.
  3. Empirical optimism by split-half. Selecting on half the episodes and
     reporting on the other half measures the optimism of the procedure instead
     of assuming a value for it.

    python memoria/scripts/analisis_seleccion.py
"""

import csv
import itertools
import random
import statistics
import sys
from pathlib import Path

from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analisis_dispersion import SELECTED, VARIANTS, describe, load_evaluations

OUT_DIR = Path(__file__).resolve().parents[2] / "memoria" / "datos"

# Last epoch evaluated by the five variants. V3 stopped at 154 and V4 exhausted
# its 200-epoch budget, so 150 is the largest common index. Chosen before
# looking at any score, which is the whole point of this reading.
EPOCA_COMUN = 150

# V3 and V4 only got four rollouts; restricting everybody to their first four
# removes the differential part of the selection bias.
K_IGUALADO = 4

N_PARTICIONES = 1000
SEED_PARTICION = 42
UMBRAL_EXITO = 0.999


def holm(pvalues):
    """Step-down Holm adjustment, returned in the order the p-values came in."""
    ajustados = [None] * len(pvalues)
    previo = 0.0
    for rango, indice in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        previo = max(previo, min(1.0, (len(pvalues) - rango) * pvalues[indice]))
        ajustados[indice] = previo
    return ajustados


def contraste(a, b):
    """Paired Wilcoxon plus the mean paired difference and its normal interval."""
    diferencias = [x - y for x, y in zip(a, b)]
    no_nulos = [d for d in diferencias if d != 0]
    empates = len(set(abs(d) for d in no_nulos)) != len(no_nulos)
    # The rule scipy itself applies: the exact null distribution needs no ties,
    # no zeros and a small sample. Recording which one ran is part of M6.
    exacto = len(no_nulos) <= 50 and not empates and len(no_nulos) == len(diferencias)
    estadistico, p = wilcoxon(
        a, b, zero_method="wilcox", method="exact" if exacto else "approx"
    )
    pareado = describe(diferencias)
    return {
        "diferencia_medias": round(pareado["media"], 6),
        "error_estandar": round(pareado["error_estandar"], 6),
        "ic95_inf": round(pareado["ic95_inf"], 6),
        "ic95_sup": round(pareado["ic95_sup"], 6),
        "pares_no_nulos": len(no_nulos),
        "metodo_wilcoxon": "exacto" if exacto else "aproximado",
        "W": estadistico,
        "p": round(p, 6),
    }


def escribir(nombre, filas):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destino = OUT_DIR / nombre
    with destino.open("w", newline="", encoding="utf-8") as handle:
        escritor = csv.DictWriter(handle, fieldnames=list(filas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    return destino


def tasa_exito(scores):
    return sum(1 for s in scores if s >= UMBRAL_EXITO) / len(scores)


def epoca_fija(evaluaciones):
    """Reading 1: the five variants at epoch 150, plus the ten paired contrasts."""
    filas = []
    for variante in VARIANTS:
        scores = evaluaciones[variante][EPOCA_COMUN]
        stats = describe(scores)
        epoca_pub, media_pub = SELECTED[variante]
        filas.append(
            {
                "variante": variante.upper(),
                "epoca": EPOCA_COMUN,
                **{k: round(v, 6) for k, v in stats.items()},
                "tasa_exito": round(tasa_exito(scores), 3),
                "epoca_publicada": epoca_pub,
                "media_publicada": round(media_pub, 6),
                "diferencia": round(stats["media"] - media_pub, 6),
            }
        )

    contrastes = []
    for primera, segunda in itertools.combinations(VARIANTS, 2):
        resultado = contraste(
            evaluaciones[primera][EPOCA_COMUN], evaluaciones[segunda][EPOCA_COMUN]
        )
        contrastes.append({"comparacion": f"{primera.upper()}-{segunda.upper()}", **resultado})
    for fila, ajustado in zip(contrastes, holm([c["p"] for c in contrastes])):
        fila["p_holm"] = round(ajustado, 6)

    return filas, contrastes


def oportunidades_igualadas(evaluaciones):
    """Reading 2: the argmax rule with the four evaluations V3 and V4 had."""
    filas = []
    for variante in VARIANTS:
        epocas = sorted(evaluaciones[variante])[:K_IGUALADO]
        medias = {e: statistics.fmean(evaluaciones[variante][e]) for e in epocas}
        elegida = max(medias, key=medias.get)
        epoca_pub, media_pub = SELECTED[variante]
        filas.append(
            {
                "variante": variante.upper(),
                "evaluaciones_totales": len(evaluaciones[variante]),
                "epocas_consideradas": " ".join(str(e) for e in epocas),
                "epoca_k4": elegida,
                "media_k4": round(medias[elegida], 6),
                "tasa_exito_k4": round(tasa_exito(evaluaciones[variante][elegida]), 3),
                "epoca_publicada": epoca_pub,
                "media_publicada": round(media_pub, 6),
                "diferencia": round(medias[elegida] - media_pub, 6),
            }
        )
    return filas


def optimismo(evaluaciones):
    """Reading 3: select on 25 episodes, report on the other 25, a thousand times.

    The gap against the published maximum is the optimism of the procedure,
    measured instead of assumed. It reads high rather than low: selecting on 25
    episodes is noisier than selecting on 50, so the optimism of the campaign is
    somewhat smaller than the figure this returns.
    """
    filas = []
    for variante in VARIANTS:
        epocas = sorted(evaluaciones[variante])
        matriz = [evaluaciones[variante][e] for e in epocas]
        n = len(matriz[0])
        epoca_pub, media_pub = SELECTED[variante]

        rng = random.Random(SEED_PARTICION)
        indices = list(range(n))
        honestas = []
        acuerdos = 0
        for _ in range(N_PARTICIONES):
            rng.shuffle(indices)
            mitad_a, mitad_b = indices[: n // 2], indices[n // 2 :]
            elegida = max(
                range(len(epocas)),
                key=lambda k: statistics.fmean(matriz[k][j] for j in mitad_a),
            )
            honestas.append(statistics.fmean(matriz[elegida][j] for j in mitad_b))
            acuerdos += epocas[elegida] == epoca_pub

        media_honesta = statistics.fmean(honestas)
        filas.append(
            {
                "variante": variante.upper(),
                "evaluaciones": len(epocas),
                "particiones": N_PARTICIONES,
                "media_publicada": round(media_pub, 6),
                "media_honesta": round(media_honesta, 6),
                "optimismo": round(media_pub - media_honesta, 6),
                "desviacion_honesta": round(statistics.stdev(honestas), 6),
                "acuerdo_con_epoca_publicada": round(acuerdos / N_PARTICIONES, 3),
            }
        )
    return filas


def main():
    evaluaciones = {v: load_evaluations(v) for v in VARIANTS}
    for variante in VARIANTS:
        assert EPOCA_COMUN in evaluaciones[variante], (
            f"{variante} no tiene evaluacion en la epoca {EPOCA_COMUN}"
        )
        assert len(evaluaciones[variante]) >= K_IGUALADO

    filas, contrastes = epoca_fija(evaluaciones)
    print(f"== Epoca comun {EPOCA_COMUN}, regla sin maximo ==")
    for fila in filas:
        print(
            f"{fila['variante']}: {fila['media']:.4f} +- {fila['error_estandar']:.4f} (e.e.) "
            f"· IC95 [{fila['ic95_inf']:.3f}; {fila['ic95_sup']:.3f}] "
            f"· exito {fila['tasa_exito']:.0%} "
            f"· publicada (epoca {fila['epoca_publicada']}) {fila['media_publicada']:.4f} "
            f"({fila['diferencia']:+.4f})"
        )
    print()
    for fila in contrastes:
        print(
            f"{fila['comparacion']}: {fila['diferencia_medias']:+.4f} "
            f"· IC95 [{fila['ic95_inf']:.3f}; {fila['ic95_sup']:.3f}] "
            f"· p = {fila['p']:.4g} · Holm = {fila['p_holm']:.4g} ({fila['metodo_wilcoxon']})"
        )
    escribir("seleccion_epoca_fija.csv", filas)
    escribir("seleccion_epoca_fija_contrastes.csv", contrastes)

    print(f"\n== Oportunidades igualadas a K = {K_IGUALADO} ==")
    filas = oportunidades_igualadas(evaluaciones)
    for fila in filas:
        print(
            f"{fila['variante']}: epoca {fila['epoca_k4']} · {fila['media_k4']:.4f} "
            f"· publicada (epoca {fila['epoca_publicada']}, K = "
            f"{fila['evaluaciones_totales']}) {fila['media_publicada']:.4f} "
            f"({fila['diferencia']:+.4f})"
        )
    escribir("seleccion_k4.csv", filas)

    print("\n== Optimismo empirico por particion mitad-mitad ==")
    filas = optimismo(evaluaciones)
    for fila in filas:
        print(
            f"{fila['variante']}: publicada {fila['media_publicada']:.4f} "
            f"· honesta {fila['media_honesta']:.4f} "
            f"· optimismo {fila['optimismo']:+.4f} "
            f"· acuerdo con la epoca publicada {fila['acuerdo_con_epoca_publicada']:.0%}"
        )
    escribir("seleccion_optimismo.csv", filas)

    print(f"\nResultados escritos en {OUT_DIR}")


if __name__ == "__main__":
    main()
