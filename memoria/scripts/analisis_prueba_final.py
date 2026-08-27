#!/usr/bin/env python3
"""Analysis of the final test block, exactly as preregistered.

Reads the raw rollouts that ``diffuser/scripts/evaluar_bloque_test.py`` wrote to
``logs_entrenamiento/prueba_final/`` and produces the three tables the memoir
reports as its final result. The protocol is not decided here: it is written in
``memoria/preregistro_prueba_final.md`` and committed before the evaluation ran.
This script only executes it.

What the preregistration fixed, and this script obeys:

  * primary endpoint, the mean coverage score over the 200 conditions;
  * secondary endpoint, the success rate at the 0.999 threshold;
  * 95 % intervals of the paired differences by BCa bootstrap, 10 000 resamples,
    seed 42 -- the memoir did not state a method before, which was finding M6;
  * paired Wilcoxon with zeros dropped, normal approximation with continuity
    correction, and the method actually used recorded in every row;
  * family of the ten pairwise comparisons, Holm step-down, threshold 0.05.

    python memoria/scripts/analisis_prueba_final.py
"""

import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analisis_dispersion import SELECTED, VARIANTS, describe
from analisis_seleccion import UMBRAL_EXITO, holm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "logs_entrenamiento" / "prueba_final"
OUT_DIR = ROOT / "memoria" / "datos"

N_REMUESTREOS = 10000
SEED_BOOTSTRAP = 42
N_ESPERADO = 200
SEMILLA_INICIAL = 200000


def cargar(variante):
    path = RAW_DIR / f"prueba_{variante}.json"
    if not path.is_file():
        raise SystemExit(
            f"falta {path}. Ejecuta antes, desde WSL:\n"
            "  bash diffuser/scripts/evaluar_bloque_test.sh"
        )
    datos = json.loads(path.read_text(encoding="utf-8"))
    semillas = [int(s) for s in datos["puntuaciones"]]
    assert semillas == list(range(SEMILLA_INICIAL, SEMILLA_INICIAL + N_ESPERADO)), (
        f"{variante}: las semillas no son el bloque preregistrado"
    )
    datos["scores"] = [datos["puntuaciones"][str(s)] for s in semillas]
    datos["semillas"] = semillas
    return datos


def ic_bca(diferencias):
    """BCa interval of the mean paired difference, as preregistered."""
    muestra = np.asarray(diferencias, dtype=float)
    if np.allclose(muestra, muestra[0]):
        # A constant sample has no bootstrap distribution; report the point.
        return float(muestra[0]), float(muestra[0])
    resultado = bootstrap(
        (muestra,),
        np.mean,
        n_resamples=N_REMUESTREOS,
        confidence_level=0.95,
        method="BCa",
        random_state=np.random.default_rng(SEED_BOOTSTRAP),
    )
    return float(resultado.confidence_interval.low), float(resultado.confidence_interval.high)


def contraste(a, b):
    diferencias = [x - y for x, y in zip(a, b)]
    no_nulos = [d for d in diferencias if d != 0]
    empates = len(set(abs(d) for d in no_nulos)) != len(no_nulos)
    # With 200 pairs the exact null distribution is out of reach; the condition
    # is checked rather than assumed, and the answer is recorded.
    exacto = len(no_nulos) <= 50 and not empates and len(no_nulos) == len(diferencias)
    estadistico, p = wilcoxon(
        a, b,
        zero_method="wilcox",
        method="exact" if exacto else "approx",
        correction=not exacto,
    )
    pareado = describe(diferencias)
    inf, sup = ic_bca(diferencias)
    return {
        "diferencia_medias": round(pareado["media"], 6),
        "error_estandar": round(pareado["error_estandar"], 6),
        "ic95_bca_inf": round(inf, 6),
        "ic95_bca_sup": round(sup, 6),
        "pares_no_nulos": len(no_nulos),
        "metodo_wilcoxon": "exacto" if exacto else "aproximado con correccion de continuidad",
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


def main():
    datos = {v: cargar(v) for v in VARIANTS}

    episodios = []
    resumen = []
    for variante, etiqueta in VARIANTS.items():
        registro = datos[variante]
        scores = registro["scores"]
        stats = describe(scores)
        exitos = sum(1 for s in scores if s >= UMBRAL_EXITO)
        epoca_sel, media_sel = SELECTED[variante]

        for semilla, score in zip(registro["semillas"], scores):
            episodios.append(
                {"variante": variante.upper(), "semilla": semilla, "puntuacion": score}
            )

        resumen.append(
            {
                "variante": variante.upper(),
                "codificador": registro["codificador"],
                "punto_control": registro["punto_control"],
                "epoca": epoca_sel,
                **{k: round(v, 6) for k, v in stats.items()},
                "exitos": exitos,
                "tasa_exito": round(exitos / len(scores), 3),
                "media_seleccion": round(media_sel, 6),
                "optimismo_realizado": round(media_sel - stats["media"], 6),
            }
        )
        print(
            f"{etiqueta}: {stats['media']:.4f} +- {stats['error_estandar']:.4f} (e.e.) "
            f"· IC95 [{stats['ic95_inf']:.3f}; {stats['ic95_sup']:.3f}] "
            f"· exito {exitos}/{len(scores)} "
            f"· seleccion {media_sel:.4f} ({media_sel - stats['media']:+.4f})"
        )

    contrastes = []
    for primera, segunda in itertools.combinations(VARIANTS, 2):
        contrastes.append(
            {
                "comparacion": f"{primera.upper()}-{segunda.upper()}",
                **contraste(datos[primera]["scores"], datos[segunda]["scores"]),
            }
        )
    for fila, ajustado in zip(contrastes, holm([c["p"] for c in contrastes])):
        fila["p_holm"] = round(ajustado, 6)

    print()
    for fila in contrastes:
        print(
            f"{fila['comparacion']}: {fila['diferencia_medias']:+.4f} "
            f"· IC95 BCa [{fila['ic95_bca_inf']:.3f}; {fila['ic95_bca_sup']:.3f}] "
            f"· W = {fila['W']:.0f} · p = {fila['p']:.4g} · Holm = {fila['p_holm']:.4g}"
        )

    # The summary must be reconstructible from the raw matrix, or one of the two
    # is wrong. Cheap check, and it is the one a reader would want run.
    assert len(episodios) == len(VARIANTS) * N_ESPERADO
    for fila in resumen:
        propios = [e["puntuacion"] for e in episodios if e["variante"] == fila["variante"]]
        assert abs(statistics.fmean(propios) - fila["media"]) < 1e-6

    escribir("prueba_final_episodios.csv", episodios)
    escribir("prueba_final_resumen.csv", resumen)
    escribir("prueba_final_contrastes.csv", contrastes)
    print(f"\nResultados escritos en {OUT_DIR}")


if __name__ == "__main__":
    main()
