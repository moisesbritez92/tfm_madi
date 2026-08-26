#!/usr/bin/env bash
#
# medir_memoria_gpu.sh - pico de memoria grafica de las cinco variantes, en
# entrenamiento y en inferencia, dentro del entorno de WSL en el que se entreno.
#
#   wsl -d Ubuntu -- bash /mnt/c/.../diffuser/scripts/medir_memoria_gpu.sh
#
# Cada combinacion de variante y modo se ejecuta en su propio proceso: el
# asignador de PyTorch no devuelve al driver lo que ya ha reservado, de modo que
# medir dos variantes seguidas en el mismo proceso contamina el pico de la
# segunda. El resultado se acumula en memoria/datos/memoria_gpu.csv.
#
# Requisito: nada mas en la GPU. Vale la regla de instancia unica de
# run_encoder_exp.sh; no debe haber un entrenamiento vivo ni una medida de
# latencia corriendo en el lado de Windows.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/tfm/diffusion_policy}"
TFM_DIR="${TFM_DIR:-/mnt/c/Users/moise/Documents/0001_MADI/TFM}"
SCRIPT="$TFM_DIR/diffuser/scripts/memoria_gpu.py"

[[ -f "$SCRIPT" ]] || { echo "No existe el script: $SCRIPT" >&2; exit 1; }
[[ -d "$REPO_DIR" ]] || { echo "No existe la copia de trabajo: $REPO_DIR" >&2; exit 1; }

# shellcheck disable=SC1091
source "$HOME/mambaforge/etc/profile.d/conda.sh"
conda activate robodiff
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

if pgrep -f "train_diffusion_unet_image_workspace" > /dev/null; then
  echo "Hay un entrenamiento vivo; la medida no seria valida." >&2
  exit 1
fi

python -c "import torch; assert torch.cuda.is_available()"
echo "entorno : $(python -c 'import torch,sys; print(sys.version.split()[0], torch.__version__, torch.version.cuda)')"

PRIMERA=1
for VARIANTE in v0 v1 v2 v3 v4; do
  # entrenamiento (lote propio de la variante) + inferencia con lote 1 y 8
  for TAREA in "entrenamiento 0" "inferencia 1" "inferencia 8"; do
    set -- $TAREA
    MODO="$1"; LOTE="$2"
    echo
    echo "== $VARIANTE / $MODO ${LOTE/0/} =="
    ARGS=(--variante "$VARIANTE" --modo "$MODO" --repo "$REPO_DIR")
    [[ "$MODO" == "inferencia" ]] && ARGS+=(--lote "$LOTE")
    if [[ $PRIMERA -eq 1 ]]; then
      ARGS+=(--reiniciar)
      PRIMERA=0
    fi
    python "$SCRIPT" "${ARGS[@]}"
  done
done

echo
echo "Medidas completas en $TFM_DIR/memoria/datos/memoria_gpu.csv"
