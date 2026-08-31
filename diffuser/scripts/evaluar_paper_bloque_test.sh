#!/usr/bin/env bash
#
# evaluar_paper_bloque_test.sh - comparacion del punto de control publicado del
# articulo con V0, sobre el mismo bloque disjunto 200000-200199 y con dos
# realizaciones de ruido de difusion por brazo.
#
#   wsl -d Ubuntu -- bash /mnt/c/.../diffuser/scripts/evaluar_paper_bloque_test.sh --portones
#   wsl -d Ubuntu -- bash /mnt/c/.../diffuser/scripts/evaluar_paper_bloque_test.sh
#
# Los tres portones van primero y son obligatorios. Estan definidos con su
# criterio numerico en memoria/preregistro_comparacion_paper.md:
#
#   1. Deriva: V0 sobre las ocho primeras condiciones del bloque con la semilla
#      base original tiene que reproducir bit a bit las ocho primeras de
#      prueba_v0.json. Si no, no se puede reutilizar ese fichero como brazo de
#      la realizacion A.
#   2. Cordura de V_Paper: sobre SU bloque publicado 4300000-4300049, la media
#      tiene que caer a menos de 0,07 del 0,884 que los autores reportan. Es el
#      porton que valida que esta ruta de inferencia reproduce la de ellos.
#   3. Ruido comun: V0 y V_Paper tienen que muestrear el mismo primer tensor de
#      ruido con la misma semilla. Si no coinciden NO se aborta: el pareo por
#      condicion inicial sigue valiendo y solo se pierde la reduccion de
#      varianza. La contingencia esta declarada en el preregistro.
#
# Las tres corridas buenas se ejecutan UNA SOLA VEZ. El script de Python se
# niega a sobrescribir un resultado ya escrito; esa negativa es parte del
# protocolo, no una molestia.
#
# V0 en la realizacion A no se ejecuta: es prueba_v0.json, y el porton 1 lo
# respalda.
#
# Una corrida por proceso: el asignador de PyTorch no devuelve al driver lo que
# ya ha reservado. Requisito: nada mas en la GPU.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/tfm/diffusion_policy}"
TFM_DIR="${TFM_DIR:-/mnt/c/Users/moise/Documents/0001_MADI/TFM}"
SCRIPT="$TFM_DIR/diffuser/scripts/evaluar_paper_bloque_test.py"
SCRIPT_V0="$TFM_DIR/diffuser/scripts/evaluar_bloque_test.py"
PRUEBA_FINAL="$TFM_DIR/logs_entrenamiento/prueba_final"
LOCK_FILE="/tmp/encoder_exp.lock"

# Semillas base de las dos realizaciones de ruido, fijadas en el preregistro.
BASE_A=20260827
BASE_B=20260831

[[ -f "$SCRIPT" ]] || { echo "No existe el script: $SCRIPT" >&2; exit 1; }
[[ -f "$SCRIPT_V0" ]] || { echo "No existe el script hermano: $SCRIPT_V0" >&2; exit 1; }
[[ -d "$REPO_DIR" ]] || { echo "No existe la copia de trabajo: $REPO_DIR" >&2; exit 1; }

# shellcheck disable=SC1091
source "$HOME/mambaforge/etc/profile.d/conda.sh"
conda activate robodiff
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

if pgrep -f "train_diffusion_unet_image_workspace" > /dev/null; then
  echo "Hay un entrenamiento vivo; la evaluacion competiria por la tarjeta." >&2
  exit 1
fi

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Otro proceso tiene el lock del experimento." >&2; exit 1; }

python -c "import torch; assert torch.cuda.is_available()"
python -c "import robomimic; print('robomimic', robomimic.__version__)"
echo "entorno : $(python -c 'import torch,sys; print(sys.version.split()[0], torch.__version__, torch.version.cuda)')"

if [[ "${1:-}" == "--portones" ]]; then
  echo
  echo "== porton 1: deriva de la ruta de codigo, V0 sobre 200000-200007 =="
  python "$SCRIPT_V0" --variante v0 --n-test 8 --test-start-seed 200000 \
    --base-seed "$BASE_A" --etiqueta deriva --repo "$REPO_DIR" --forzar
  python - "$PRUEBA_FINAL" <<'PY'
import json, sys
from pathlib import Path

carpeta = Path(sys.argv[1])
nuevo = json.loads((carpeta / "deriva_v0.json").read_text())["puntuaciones"]
viejo = json.loads((carpeta / "prueba_v0.json").read_text())["puntuaciones"]
distintas = [s for s in nuevo if nuevo[s] != viejo[s]]
for s in sorted(nuevo):
    marca = "!=" if s in distintas else "=="
    print(f"  {s}: {nuevo[s]!r} {marca} {viejo[s]!r}")
if distintas:
    print(f"DERIVA en {len(distintas)}/{len(nuevo)} condiciones. prueba_v0.json no se "
          "puede reutilizar como realizacion A; hay que reejecutar tambien ese brazo.")
    sys.exit(1)
print("sin deriva: la ruta de codigo reproduce prueba_v0.json bit a bit")
PY

  echo
  echo "== porton 2: cordura de V_Paper sobre su bloque publicado 4300000-4300049 =="
  python "$SCRIPT" --n-test 50 --test-start-seed 4300000 \
    --etiqueta cordura --repo "$REPO_DIR" --forzar

  echo
  echo "== porton 3: alineacion del ruido comun entre V0 y V_Paper =="
  python "$SCRIPT" --comprobar-ruido --base-seed "$BASE_A" --repo "$REPO_DIR"

  echo
  echo "Portones superados. Ejecuta el script sin --portones para las tres corridas."
  exit 0
fi

echo
echo "== V_Paper, realizacion A (semilla base $BASE_A) =="
python "$SCRIPT" --base-seed "$BASE_A" --etiqueta prueba --repo "$REPO_DIR"

echo
echo "== V_Paper, realizacion B (semilla base $BASE_B) =="
python "$SCRIPT" --base-seed "$BASE_B" --etiqueta ruido_b --repo "$REPO_DIR"

echo
echo "== V0, realizacion B (semilla base $BASE_B) =="
python "$SCRIPT_V0" --variante v0 --base-seed "$BASE_B" --etiqueta ruido_b --repo "$REPO_DIR"

echo
echo "Resultados en $PRUEBA_FINAL/"
echo "Siguiente paso, en Windows: python memoria/scripts/analisis_comparacion_paper.py"
