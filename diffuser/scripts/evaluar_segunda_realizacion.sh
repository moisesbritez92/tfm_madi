#!/usr/bin/env bash
#
# evaluar_segunda_realizacion.sh - segunda realizacion del ruido de difusion para
# las cinco variantes, sobre el mismo bloque disjunto 200000-200199.
#
#   wsl -d Ubuntu -- bash /mnt/c/.../diffuser/scripts/evaluar_segunda_realizacion.sh
#
# Por que existe. La prueba final resolvio cada condicion con UNA trayectoria de
# difusion, de modo que sus medias quedan condicionadas a esa realizacion del
# ruido (hallazgo M5). La comparacion con el punto de control publicado midio esa
# sensibilidad y resulto ser grande: cambiar solo la semilla base movio la media
# de V0 de 0,8719 a 0,9019. Este script repite la medida de las cinco variantes
# con la segunda semilla base, 20260831, la misma que ya uso V0, para que las
# tablas puedan promediar dos realizaciones por condicion.
#
# ATENCION, y es lo importante: promediar dos realizaciones **cambia el estimando
# declarado** en memoria/preregistro_prueba_final.md, que fijo una sola
# trayectoria. Todo analisis que use estos ficheros es POST HOC y debe declararse
# asi, junto a las cifras preregistradas, nunca en su lugar. El precedente de
# como hacerlo bien esta en la cabecera de memoria/scripts/analisis_prueba_final_v2.py.
#
# No se toca evaluar_bloque_test.py: es el artefacto que produjo los cinco JSON
# congelados. Este driver solo lo invoca con otra semilla base y otra etiqueta.
# V0 no se ejecuta: ruido_b_v0.json ya existe, escrito por
# evaluar_paper_bloque_test.sh.
#
# Una corrida por proceso: el asignador de PyTorch no devuelve al driver lo que ya
# ha reservado. Requisito: nada mas en la GPU.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/tfm/diffusion_policy}"
TFM_DIR="${TFM_DIR:-/mnt/c/Users/moise/Documents/0001_MADI/TFM}"
SCRIPT="$TFM_DIR/diffuser/scripts/evaluar_bloque_test.py"
PRUEBA_FINAL="$TFM_DIR/logs_entrenamiento/prueba_final"
LOCK_FILE="/tmp/encoder_exp.lock"

# Segunda semilla base, declarada en memoria/preregistro_comparacion_paper.md.
BASE_B=20260831
VARIANTES="${VARIANTES:-v1 v2 v3 v4}"

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

for VARIANTE in $VARIANTES; do
  echo
  echo "== $VARIANTE, realizacion B (semilla base $BASE_B) =="
  python "$SCRIPT" --variante "$VARIANTE" --base-seed "$BASE_B" \
    --etiqueta ruido_b --repo "$REPO_DIR"
done

echo
echo "Resultados en $PRUEBA_FINAL/"
echo "Siguiente paso, en Windows: python memoria/scripts/analisis_prueba_final_v3.py"
