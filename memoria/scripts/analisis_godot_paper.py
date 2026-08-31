#!/usr/bin/env python3
"""Analisis del contraste entre V0 y el punto de control publicado, dentro de Godot.

El protocolo esta fijado por adelantado en ``memoria/preregistro_godot_paper.md``.
Este guion no decide nada: ejecuta lo que ese documento declara.

Hermano de ``analisis_comparacion_paper.py``, que **no se toca**: aquel produjo los
CSV del contraste en el simulador original y editarlo romperia la cadena de
custodia. Todo lo que se puede compartir se importa de el y de sus vecinos, de modo
que los dos caminos difieren solo donde deben.

  * **Contraste primario, uno solo:** media de las 50 diferencias pareadas por
    condicion inicial **en condicion B**, con las dos realizaciones de ruido
    promediadas dentro de cada condicion. Permutacion por inversion de signo,
    bilateral, 10 000 permutaciones, semilla 42. **No hay multiplicidad que
    corregir.**
  * **Equivalencia:** TOST con margen 0,05, resuelto como intervalo BCa al 90 %.
    El preregistro ya declara que con n = 50 ese intervalo no cabe en el margen y
    que la casilla de equivalencia esta fuera de alcance por diseno.
  * **Cinco secundarios, sin correccion y etiquetados como tales:** el mismo
    contraste en condicion A; la caida de A a B por brazo; la deriva respecto de
    ``prueba_final/``; la tasa de exito con McNemar exacto; y la componente de
    varianza entre realizaciones.

La familia de diez comparaciones de la prueba final y el contraste de
``comparacion_paper`` no se tocan: este guion no escribe ninguno de sus ficheros.

Entrada:  logs_entrenamiento/godot_paper/{prueba,ruido_b}_{v0,v_paper}_{a,b}.json
Salida:   memoria/datos/godot_paper_episodios.csv
          memoria/datos/godot_paper_resumen.csv
          memoria/datos/godot_paper_contrastes.csv

    python memoria/scripts/analisis_godot_paper.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analisis_comparacion_paper import (
    DELTA,
    NIVEL_TOST,
    decision,
    ic_bca_nivel,
    mcnemar_exacto,
)
from analisis_dispersion import describe
from analisis_prueba_final import ic_bca
from analisis_prueba_final_v2 import permutacion_media
from analisis_seleccion import UMBRAL_EXITO

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "logs_entrenamiento" / "godot_paper"
PRUEBA_FINAL = ROOT / "logs_entrenamiento" / "prueba_final"
OUT_DIR = ROOT / "memoria" / "datos"

N_ESPERADO = 50
SEMILLA_INICIAL = 200000
SEMILLA = 42

REALIZACIONES = {"prueba": 20260827, "ruido_b": 20260831}
BRAZOS = {"v0": "V0", "v_paper": "V_PAPER"}
CONDICIONES = {"a": "estado (solo la fisica)", "b": "godot (fisica y pixeles)"}
CONDICION_PRIMARIA = "b"

# Cifras de la pasada preregistrada en el simulador original, para el secundario 3.
REFERENCIA_FINAL = {"v0": "prueba_v0.json", "v_paper": "prueba_paper.json"}

# Puntos de control congelados en el preregistro, por SHA-256.
SHA_ESPERADO = {
    "v0": "5310551ee71075d9efcf956c34670809741d84e06808809551e7675674e8ce63",
    "v_paper": "bac7221f7e34cd51162dc1972e1a39ffcddc87de1dc1780c44ffa61b88c4ff76",
}


def cargar(brazo, condicion, realizacion):
    path = RAW_DIR / f"{realizacion}_{brazo}_{condicion}.json"
    if not path.is_file():
        raise SystemExit(
            f"falta {path}. Ejecuta antes, desde Windows:\n"
            "  .venv_diffuser_infer\\Scripts\\python.exe "
            "diffuser/godot/servidor/comparar_godot_paper.py"
        )
    datos = json.loads(path.read_text(encoding="utf-8"))
    assert datos["base_seed_difusion"] == REALIZACIONES[realizacion], (
        f"{path.name}: semilla base {datos['base_seed_difusion']}, se esperaba "
        f"{REALIZACIONES[realizacion]}"
    )
    assert datos["sha256"] == SHA_ESPERADO[brazo], (
        f"{path.name}: el punto de control no es el congelado en el preregistro"
    )
    assert datos["condicion"] == condicion and datos["brazo"] == brazo
    assert datos["n_test"] == N_ESPERADO, f"{path.name}: n_test es {datos['n_test']}"
    assert datos["test_start_seed"] == SEMILLA_INICIAL
    assert datos["perturbacion"] == "ninguna", f"{path.name}: la escena esta perturbada"
    return datos


def semillas_comunes(datos):
    """Las condiciones que sobrevivieron en las ocho celdas.

    El preregistro permite perder un episodio tras un reintento. Un episodio
    perdido rompe el pareo, asi que su condicion se retira de **todas** las
    celdas, no solo de la suya, y la retirada se declara en el informe.
    """
    conjuntos = [set(d["puntuaciones"]) for d in datos.values()]
    comunes = sorted(int(s) for s in set.intersection(*conjuntos))
    return comunes


def vector(datos, semillas):
    return np.array([datos["puntuaciones"][str(s)] for s in semillas], dtype=float)


def contraste(nombre, diferencias, etiqueta_a, etiqueta_b, primario):
    """Una fila de la tabla de contrastes. Misma maquinaria para todos."""
    resumen = describe(list(diferencias))
    observado, p_perm = permutacion_media(diferencias, np.random.default_rng(SEMILLA))
    inf95, sup95 = ic_bca(diferencias)
    inf90, sup90 = ic_bca_nivel(diferencias, NIVEL_TOST)
    equivalencia = inf90 > -DELTA and sup90 < DELTA
    w, p_wilcoxon = wilcoxon(diferencias, zero_method="wilcox", method="approx",
                             correction=True)
    return {
        "contraste": nombre,
        "papel": "primario" if primario else "secundario (sin correccion)",
        "a": etiqueta_a,
        "b": etiqueta_b,
        "n": len(diferencias),
        "diferencia_medias": round(observado, 6),
        "error_estandar": round(resumen["error_estandar"], 6),
        "ic95_bca_inf": round(inf95, 6),
        "ic95_bca_sup": round(sup95, 6),
        "ic90_bca_inf": round(inf90, 6),
        "ic90_bca_sup": round(sup90, 6),
        "delta_equivalencia": DELTA,
        "equivalencia": equivalencia,
        "p_permutacion": p_perm,
        "decision": decision(p_perm, inf90, sup90) if primario else "",
        "W": w,
        "p_wilcoxon": float(p_wilcoxon),
        "gana_a": int(np.sum(diferencias > 0)),
        "gana_b": int(np.sum(diferencias < 0)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datos = {
        (brazo, condicion, realizacion): cargar(brazo, condicion, realizacion)
        for brazo in BRAZOS
        for condicion in CONDICIONES
        for realizacion in REALIZACIONES
    }
    semillas = semillas_comunes(datos)
    perdidas = sorted(set(range(SEMILLA_INICIAL, SEMILLA_INICIAL + N_ESPERADO))
                      - set(semillas))
    n = len(semillas)

    # Episodios en bruto: 8 celdas por las condiciones que sobrevivieron.
    with (OUT_DIR / "godot_paper_episodios.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["brazo", "condicion", "realizacion", "semilla_base",
                           "semilla", "puntuacion"])
        for (brazo, condicion, realizacion), d in sorted(datos.items()):
            for semilla in semillas:
                escritor.writerow([BRAZOS[brazo], condicion, realizacion,
                                   d["base_seed_difusion"], semilla,
                                   d["puntuaciones"][str(semilla)]])

    # Promedio de las dos realizaciones dentro de cada condicion inicial.
    promedio, filas_resumen = {}, []
    for brazo in BRAZOS:
        for condicion in CONDICIONES:
            por_realizacion = [
                vector(datos[(brazo, condicion, r)], semillas) for r in REALIZACIONES
            ]
            promedio[(brazo, condicion)] = np.mean(por_realizacion, axis=0)
            for realizacion in REALIZACIONES:
                d = datos[(brazo, condicion, realizacion)]
                valores = vector(d, semillas)
                resumen = describe(list(valores))
                exitos = int(np.sum(valores >= UMBRAL_EXITO))
                filas_resumen.append({
                    "brazo": BRAZOS[brazo],
                    "condicion": condicion,
                    "realizacion": realizacion,
                    "semilla_base": d["base_seed_difusion"],
                    "punto_control": d["punto_control"],
                    "n": resumen["n"],
                    "media": round(resumen["media"], 6),
                    "desviacion": round(resumen["desviacion"], 6),
                    "error_estandar": round(resumen["error_estandar"], 6),
                    "exitos": exitos,
                    "tasa_exito": round(exitos / n, 6),
                    "segundos": d["segundos"],
                })
            # Secundario 5: varianza intra-condicion entre las dos realizaciones.
            # Mezcla el cambio de semilla con el no determinismo de la GPU; el
            # preregistro lo declara y no se puede repartir entre los dos.
            intra = np.var(np.stack(por_realizacion, axis=0), axis=0, ddof=1)
            filas_resumen.append({
                "brazo": BRAZOS[brazo],
                "condicion": condicion,
                "realizacion": "media de las dos",
                "punto_control": datos[(brazo, condicion, "prueba")]["punto_control"],
                "n": n,
                "media": round(float(np.mean(promedio[(brazo, condicion)])), 6),
                "desviacion": round(float(np.std(promedio[(brazo, condicion)], ddof=1)), 6),
                "error_estandar": round(
                    float(np.std(promedio[(brazo, condicion)], ddof=1) / np.sqrt(n)), 6
                ),
                "varianza_ruido_intra": round(float(np.mean(intra)), 8),
                # Secundario 3: deriva respecto del simulador original.
                "media_prueba_final": round(referencia_final(brazo, semillas).mean(), 6),
            })

    campos = ["brazo", "condicion", "realizacion", "semilla_base", "punto_control",
              "n", "media", "desviacion", "error_estandar", "exitos", "tasa_exito",
              "segundos", "varianza_ruido_intra", "media_prueba_final"]
    with (OUT_DIR / "godot_paper_resumen.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        escritor = csv.DictWriter(fh, fieldnames=campos, restval="")
        escritor.writeheader()
        escritor.writerows(filas_resumen)

    # --- Contrastes -----------------------------------------------------------
    filas = []

    # Primario: V0 - V_Paper en condicion B.
    principal = contraste(
        "V0-V_PAPER en condicion B",
        promedio[("v0", CONDICION_PRIMARIA)] - promedio[("v_paper", CONDICION_PRIMARIA)],
        "V0", "V_PAPER", primario=True,
    )
    filas.append(principal)

    # Secundario 1: el mismo contraste en condicion A.
    filas.append(contraste(
        "V0-V_PAPER en condicion A",
        promedio[("v0", "a")] - promedio[("v_paper", "a")],
        "V0", "V_PAPER", primario=False,
    ))

    # Secundario 2: caida de A a B, por brazo. Es la cifra que responde a la
    # pregunta del preregistro: cuanto pierde cada artefacto al cambiar de dominio.
    for brazo in BRAZOS:
        filas.append(contraste(
            f"caida A-B de {BRAZOS[brazo]}",
            promedio[(brazo, "a")] - promedio[(brazo, "b")],
            f"{BRAZOS[brazo]} en A", f"{BRAZOS[brazo]} en B", primario=False,
        ))

    # Secundario 3: deriva respecto del simulador original, por brazo y condicion.
    for brazo in BRAZOS:
        referencia = referencia_final(brazo, semillas)
        for condicion in CONDICIONES:
            filas.append(contraste(
                f"deriva de {BRAZOS[brazo]}: pymunk - Godot {condicion.upper()}",
                referencia - promedio[(brazo, condicion)],
                f"{BRAZOS[brazo]} en prueba_final", f"{BRAZOS[brazo]} en Godot {condicion.upper()}",
                primario=False,
            ))

    # Secundario 4: tasa de exito por realizacion y condicion, McNemar exacto.
    for condicion in CONDICIONES:
        for realizacion in REALIZACIONES:
            exito_v0 = vector(datos[("v0", condicion, realizacion)], semillas) >= UMBRAL_EXITO
            exito_paper = vector(datos[("v_paper", condicion, realizacion)], semillas) >= UMBRAL_EXITO
            solo_v0, solo_paper, p = mcnemar_exacto(exito_v0, exito_paper)
            principal[f"mcnemar_{condicion}_{realizacion}_solo_v0"] = solo_v0
            principal[f"mcnemar_{condicion}_{realizacion}_solo_paper"] = solo_paper
            principal[f"mcnemar_{condicion}_{realizacion}_p"] = p

    principal["condiciones_perdidas"] = " ".join(str(s) for s in perdidas)

    campos = list(principal)
    with (OUT_DIR / "godot_paper_contrastes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        escritor = csv.DictWriter(fh, fieldnames=campos, restval="")
        escritor.writeheader()
        escritor.writerows(filas)

    # --- Informe --------------------------------------------------------------
    print(f"Bloque {semillas[0]}-{semillas[-1]}, n = {n} de {N_ESPERADO}, "
          f"{len(REALIZACIONES)} realizaciones de ruido por brazo y condicion.")
    if perdidas:
        print(f"ATENCION: {len(perdidas)} condiciones retiradas de todas las celdas "
              f"por episodio perdido: {perdidas}")
    print()
    for f in filas_resumen:
        if f["realizacion"] == "media de las dos":
            print(f"  {f['brazo']:<8} {f['condicion'].upper()}  media de las dos "
                  f"realizaciones: {f['media']:.4f} · varianza intra {f['varianza_ruido_intra']:.6f} "
                  f"· en pymunk {f['media_prueba_final']:.4f}")
        else:
            print(f"  {f['brazo']:<8} {f['condicion'].upper()}  {f['realizacion']:<8} "
                  f"media {f['media']:.4f} · EE {f['error_estandar']:.4f} "
                  f"· exito {f['exitos']}/{n}")

    print("\nContraste PRIMARIO, V0 - V_PAPER en condicion B "
          "(la fisica y los pixeles de Godot):")
    print(f"  diferencia media   {principal['diferencia_medias']:+.4f}  "
          f"(EE {principal['error_estandar']:.4f})")
    print(f"  IC95 BCa           [{principal['ic95_bca_inf']:+.4f}, "
          f"{principal['ic95_bca_sup']:+.4f}]")
    print(f"  IC90 BCa (TOST)    [{principal['ic90_bca_inf']:+.4f}, "
          f"{principal['ic90_bca_sup']:+.4f}]  frente a delta = {DELTA}")
    print(f"  permutacion        p = {principal['p_permutacion']:.6g}   "
          f"({principal['gana_a']} condiciones a favor de V0, "
          f"{principal['gana_b']} a favor de V_Paper)")
    print(f"  Wilcoxon (robustez) W = {principal['W']:g}, "
          f"p = {principal['p_wilcoxon']:.6g}")
    print(f"\n  DECISION: {principal['decision']}")
    print("  Recordatorio del preregistro: con n = 50 el IC90 no cabe en +/-0,05, "
          "asi que\n  la casilla de equivalencia esta fuera de alcance por diseno y no "
          "poder declararla\n  no es un resultado.")

    print("\nSecundarios, sin correccion de multiplicidad:")
    for fila in filas[1:]:
        print(f"  {fila['contraste']:<48} {fila['diferencia_medias']:+.4f}  "
              f"IC95 [{fila['ic95_bca_inf']:+.4f}, {fila['ic95_bca_sup']:+.4f}]  "
              f"p = {fila['p_permutacion']:.4g}")

    print("\nSecundario, tasa de exito (McNemar exacto):")
    for condicion in CONDICIONES:
        for realizacion in REALIZACIONES:
            print(f"  {condicion.upper()} {realizacion:<8} "
                  f"solo V0 {principal[f'mcnemar_{condicion}_{realizacion}_solo_v0']:>3} · "
                  f"solo V_Paper {principal[f'mcnemar_{condicion}_{realizacion}_solo_paper']:>3} · "
                  f"p = {principal[f'mcnemar_{condicion}_{realizacion}_p']:.6g}")

    print(f"\nEscrito en {OUT_DIR}")


def referencia_final(brazo, semillas):
    """Puntuaciones del simulador original para esas condiciones, del bloque final."""
    path = PRUEBA_FINAL / REFERENCIA_FINAL[brazo]
    puntuaciones = json.loads(path.read_text(encoding="utf-8"))["puntuaciones"]
    return np.array([puntuaciones[str(s)] for s in semillas], dtype=float)


if __name__ == "__main__":
    main()
