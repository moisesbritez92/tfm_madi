#!/usr/bin/env bash
#
# evaluar_bloque_test.sh - prueba final de las cinco variantes sobre el bloque de
# semillas disjunto 200000-200199, dentro del entorno de WSL en el que se entreno.
#
#   wsl -d Ubuntu -- bash /mnt/c/.../diffuser/scripts/evaluar_bloque_test.sh --cordura
#   wsl -d Ubuntu -- bash /mnt/c/.../diffuser/scripts/evaluar_bloque_test.sh
#
# La comprobacion de cordura va primero y es obligatoria: reevalua V0 sobre el
# conjunto de seleccion 100000-100049 y comprueba que la media cae cerca de los
# 0,8645 registrados durante el entrenamiento. Si no cae ahi, la ruta de
# inferencia no es la del entrenamiento y no tiene sentido gastar las horas de
# la pasada buena.
#
# La pasada buena se ejecuta UNA SOLA VEZ. El script de Python se niega a
# sobrescribir un resultado ya escrito; esa negativa es parte del protocolo
# preregistrado en memoria/preregistro_prueba_final.md, no una molestia.
#
# Cada variante corre en su propio proceso: el asignador de PyTorch no devuelve
# al driver lo que ya ha reservado y V2 llega a rozar la capacidad de la tarjeta.
#
# Requisito: nada mas en la GPU. Vale la regla de instancia unica de
# run_encoder_exp.sh.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/tfm/diffusion_policy}"
TFM_DIR="${TFM_DIR:-/mnt/c/Users/moise/Documents/0001_MADI/TFM}"
SCRIPT="$TFM_DIR/diffuser/scripts/evaluar_bloque_test.py"
LOCK_FILE="/tmp/encoder_exp.lock"
VARIANTES="${VARIANTES:-v0 v1 v2 v3 v4}"

# Media registrada de V0 sobre el conjunto de seleccion y dos errores estandar.
CORDURA_ESPERADA=0.8645
CORDURA_TOLERANCIA=0.0793

[[ -f "$SCRIPT" ]] || { echo "No existe el script: $SCRIPT" >&2; exit 1; }
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
echo "entorno : $(python -c 'import torch,sys; print(sys.version.split()[0], torch.__version__, torch.version.cuda)')"

if [[ "${1:-}" == "--cordura" ]]; then
  echo
  echo "== comprobacion de cordura: V0 sobre el conjunto de seleccion =="
  python "$SCRIPT" --variante v0 --n-test 50 --test-start-seed 100000 \
    --etiqueta cordura --repo "$REPO_DIR" --forzar
  python - "$TFM_DIR" "$CORDURA_ESPERADA" "$CORDURA_TOLERANCIA" <<'PY'
import json, sys
from pathlib import Path

tfm, esperada, tolerancia = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
media = json.loads((tfm / "logs_entrenamiento" / "prueba_final" / "cordura_v0.json").read_text())["media"]
desvio = abs(media - esperada)
print(f"media {media:.4f} frente a {esperada:.4f} registrada: desvio {desvio:.4f}")
if desvio > tolerancia:
    print(f"FUERA DE TOLERANCIA ({tolerancia:.4f}). La ruta de inferencia no reproduce "
          "la del entrenamiento; no se ejecuta la prueba final.")
    sys.exit(1)
print("dentro de tolerancia: la ruta de inferencia reproduce la del entrenamiento")
PY
  echo
  echo "Cordura superada. Ejecuta el script sin --cordura para la pasada final."
  exit 0
fi

echo
echo "== prueba final: bloque 200000-200199, una sola pasada =="
for VARIANTE in $VARIANTES; do
  echo
  echo "== $VARIANTE =="
  python "$SCRIPT" --variante "$VARIANTE" --repo "$REPO_DIR"
done

echo
echo "Resultados en $TFM_DIR/logs_entrenamiento/prueba_final/"
echo "Siguiente paso, en Windows: python memoria/scripts/analisis_prueba_final.py"
