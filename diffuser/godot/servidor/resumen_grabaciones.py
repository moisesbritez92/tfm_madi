#!/usr/bin/env python3
"""Resume las grabaciones y ayuda a elegir los episodios de la defensa.

Lee todos los `grabar_*.json` de `grabaciones/` y los pone en una tabla junto a
la puntuacion que ese mismo punto de control obtuvo en la pasada preregistrada,
cuando la semilla pertenece al bloque 200000-200199.

**Esa columna de referencia es lo unico reportable de esta tabla.** Las otras dos
son de una demostracion: un episodio por semilla, en un motor distinto, con un
muestreo de difusion que no se sincronizo con el de la prueba final. Sirven para
escoger que grabar, no para concluir nada. La comparacion columna a columna no
es un contraste: es una ayuda visual.

Uso:
    .venv_diffuser_infer\\Scripts\\python.exe diffuser/godot/servidor/resumen_grabaciones.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

AQUI = Path(__file__).resolve().parent
GODOT = AQUI.parent
RAIZ = GODOT.parent.parent
PRUEBA_FINAL = RAIZ / "logs_entrenamiento" / "prueba_final" / "prueba_v0.json"

PATRON = re.compile(r"^grabar_(?P<obs>estado|godot)_seed(?P<seed>\d+)\.json$")


def referencia() -> dict:
    """Puntuaciones por semilla de la pasada preregistrada de V0."""
    if not PRUEBA_FINAL.is_file():
        return {}
    return json.loads(PRUEBA_FINAL.read_text(encoding="utf-8"))["puntuaciones"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grabaciones", type=Path, default=GODOT / "grabaciones")
    args = parser.parse_args()

    filas = []
    for ruta in sorted(args.grabaciones.glob("grabar_*.json")):
        coincide = PATRON.match(ruta.name)
        if coincide is None:
            continue
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        filas.append({
            "seed": int(coincide["seed"]),
            "obs": coincide["obs"],
            "pasos": datos["pasos"],
            "cobertura": datos["cobertura_max"],
            "recompensa": datos["recompensa_max"],
            "decisiones": datos["decisiones"],
        })
    if not filas:
        raise SystemExit(f"no hay grabaciones en {args.grabaciones}")

    prerregistro = referencia()
    filas.sort(key=lambda f: (f["seed"], f["obs"]))

    print(f"{'seed':>8}  {'obs':>7}  {'pasos':>6}  {'cobertura':>9}  "
          f"{'recomp.':>8}  {'preregistro':>11}")
    for fila in filas:
        marca = prerregistro.get(str(fila["seed"]))
        texto = f"{marca:11.4f}" if marca is not None else f"{'-':>11}"
        print(f"{fila['seed']:8d}  {fila['obs']:>7}  {fila['pasos']:6d}  "
              f"{fila['cobertura']:9.4f}  {fila['recompensa']:8.4f}  {texto}")

    for obs in ("estado", "godot"):
        grupo = [f for f in filas if f["obs"] == obs]
        if not grupo:
            continue
        media = sum(f["recompensa"] for f in grupo) / len(grupo)
        exitos = sum(1 for f in grupo if f["recompensa"] >= 0.999)
        print(f"\ncondicion {'A' if obs == 'estado' else 'B'} ({obs}): "
              f"{len(grupo)} episodios, media {media:.4f}, {exitos} con exito")

    print("\nLas columnas 'cobertura' y 'recomp.' son de la demostracion en Godot: un\n"
          "episodio por semilla, otro motor, ruido de difusion sin sincronizar. No son\n"
          "un resultado y no se reportan. La columna 'preregistro' si lo es, y esta en\n"
          "logs_entrenamiento/prueba_final/prueba_v0.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
