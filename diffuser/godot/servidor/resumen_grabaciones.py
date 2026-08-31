#!/usr/bin/env python3
"""Resume las grabaciones por variante, condicion y perturbacion.

Lee todos los `grabar_*.json` de `grabaciones/` y los agrupa, poniendo al lado la
puntuacion que ese mismo punto de control obtuvo en la pasada preregistrada
cuando la semilla pertenece al bloque 200000-200199.

**Esa columna de referencia es lo unico reportable de esta tabla.** Las otras dos
son de una demostracion: un episodio por celda, en un motor distinto, con un
muestreo de difusion que no se sincronizo con el de la prueba final, y en
episodios que ni siquiera se repiten iguales entre ejecuciones. Sirven para ver
efectos gruesos y para escoger que grabar, no para concluir nada.

La variante y la perturbacion se leen del cuerpo del JSON y no del nombre del
fichero: las grabaciones de la primera tanda son anteriores a que esos campos
existieran, y son por construccion V0 sin perturbar.

Uso:
    .venv_diffuser_infer\\Scripts\\python.exe diffuser/godot/servidor/resumen_grabaciones.py
    ... --markdown        para pegar en perturbaciones.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
GODOT = AQUI.parent
RAIZ = GODOT.parent.parent
PRUEBA_FINAL = RAIZ / "logs_entrenamiento" / "prueba_final"


# El brazo del articulo no sigue el patron `prueba_v<n>.json`: su fichero es
# `prueba_paper.json` y las grabaciones lo nombran `v_paper`. Sin esta entrada la
# columna de referencia saldria vacia justo para el brazo que mas la necesita.
FICHERO_REFERENCIA = {"v_paper": "prueba_paper.json"}


def referencia() -> dict:
    """Puntuaciones por semilla de la pasada preregistrada, por variante."""
    salida = {}
    for ruta in sorted(PRUEBA_FINAL.glob("prueba_v*.json")):
        variante = ruta.stem.split("_")[1]
        salida[variante] = json.loads(ruta.read_text(encoding="utf-8"))["puntuaciones"]
    for variante, nombre in FICHERO_REFERENCIA.items():
        ruta = PRUEBA_FINAL / nombre
        if ruta.is_file():
            salida[variante] = json.loads(
                ruta.read_text(encoding="utf-8")
            )["puntuaciones"]
    return salida


def leer(grabaciones: Path) -> list:
    filas = []
    for ruta in sorted(grabaciones.glob("grabar_*.json")):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        filas.append({
            "seed": int(datos["seed"]),
            "obs": datos["obs"],
            "variante": datos.get("variante", "v0"),
            "perturbacion": datos.get("perturbacion", "ninguna"),
            "pasos": datos["pasos"],
            "cobertura": datos["cobertura_max"],
            "recompensa": datos["recompensa_max"],
            "decisiones": datos["decisiones"],
        })
    return filas


def agrupar(filas: list) -> dict:
    grupos = defaultdict(list)
    for fila in filas:
        grupos[(fila["variante"], fila["perturbacion"], fila["obs"])].append(fila)
    for grupo in grupos.values():
        grupo.sort(key=lambda f: f["seed"])
    return grupos


def condicion(obs: str) -> str:
    return "A" if obs == "estado" else "B"


def imprimir_texto(grupos: dict, prerregistro: dict) -> None:
    for clave in sorted(grupos):
        variante, perturbacion, obs = clave
        grupo = grupos[clave]
        print(f"\n{variante.upper()} | condicion {condicion(obs)} ({obs}) | "
              f"perturbacion: {perturbacion}")
        print(f"{'seed':>8}  {'pasos':>6}  {'cobertura':>9}  "
              f"{'recomp.':>8}  {'preregistro':>11}")
        for fila in grupo:
            marca = prerregistro.get(variante, {}).get(str(fila["seed"]))
            texto = f"{marca:11.4f}" if marca is not None else f"{'-':>11}"
            print(f"{fila['seed']:8d}  {fila['pasos']:6d}  "
                  f"{fila['cobertura']:9.4f}  {fila['recompensa']:8.4f}  {texto}")
        media = sum(f["recompensa"] for f in grupo) / len(grupo)
        exitos = sum(1 for f in grupo if f["recompensa"] >= 0.999)
        print(f"{'media':>8}  {'':>6}  {'':>9}  {media:8.4f}  "
              f"({exitos}/{len(grupo)} con exito)")


def imprimir_markdown(grupos: dict, prerregistro: dict) -> None:
    """Una tabla por variante, con una columna por perturbacion."""
    variantes = sorted({clave[0] for clave in grupos})
    for variante in variantes:
        claves = [c for c in grupos if c[0] == variante and c[2] == "godot"]
        perturbaciones = sorted({c[1] for c in claves},
                                key=lambda n: (n != "ninguna", n))
        semillas = sorted({f["seed"] for c in claves for f in grupos[c]})
        if not semillas:
            continue
        print(f"\n### {variante.upper()}, condicion B\n")
        cabecera = ["semilla"] + perturbaciones + ["preregistro"]
        print("| " + " | ".join(cabecera) + " |")
        print("|" + "---|" * len(cabecera))
        for seed in semillas:
            celdas = [str(seed)]
            for perturbacion in perturbaciones:
                clave = (variante, perturbacion, "godot")
                valor = next(
                    (f["recompensa"] for f in grupos.get(clave, []) if f["seed"] == seed),
                    None,
                )
                celdas.append(f"{valor:.4f}" if valor is not None else "-")
            marca = prerregistro.get(variante, {}).get(str(seed))
            celdas.append(f"{marca:.4f}" if marca is not None else "-")
            print("| " + " | ".join(celdas) + " |")
        celdas = ["**media**"]
        for perturbacion in perturbaciones:
            grupo = grupos.get((variante, perturbacion, "godot"), [])
            if grupo:
                celdas.append(f"**{sum(f['recompensa'] for f in grupo) / len(grupo):.4f}**")
            else:
                celdas.append("-")
        celdas.append("")
        print("| " + " | ".join(celdas) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grabaciones", type=Path, default=GODOT / "grabaciones")
    parser.add_argument("--markdown", action="store_true",
                        help="tablas en markdown, para la bitacora")
    parser.add_argument("--semillas", type=int, nargs="+", default=None,
                        help="acota la tabla; sin esto salen todas las grabaciones, "
                             "incluidas las de tandas anteriores con otras semillas")
    args = parser.parse_args()

    filas = leer(args.grabaciones)
    if args.semillas:
        filas = [f for f in filas if f["seed"] in set(args.semillas)]
    if not filas:
        raise SystemExit(f"no hay grabaciones en {args.grabaciones}")
    grupos = agrupar(filas)
    prerregistro = referencia()

    if args.markdown:
        imprimir_markdown(grupos, prerregistro)
    else:
        imprimir_texto(grupos, prerregistro)

    aviso = (
        "Las columnas de cobertura y recompensa son de la demostracion en Godot: "
        "un episodio\npor celda, otro motor de fisica, ruido de difusion sin "
        "sincronizar entre celdas.\nNo son un resultado y no se reportan. La "
        "columna del preregistro si lo es, y esta\nen logs_entrenamiento/prueba_final/."
    )
    print("\n" + aviso)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
