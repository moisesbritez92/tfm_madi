#!/usr/bin/env bash
#
# exportar_logs_wsl.sh - vuelca los logs de un run del experimento de encoders
# desde el arbol de trabajo de WSL al repo de Windows, con la convencion de
# logs_entrenamiento/raw/ que ya usan V0, V1 y V2.
#
#   ./exportar_logs_wsl.sh v3 42
#   ./exportar_logs_wsl.sh v3 42 /ruta/alternativa/logs_entrenamiento
#
# Los checkpoints NO se copian aqui: pesan varios GB por fichero y van aparte.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/tfm/diffusion_policy}"
OUT_ROOT="data/outputs/encoder_exp"
DEST_DEFAULT="/mnt/c/Users/moise/Documents/0001_MADI/TFM/logs_entrenamiento"

VARIANT="${1:-}"
SEED="${2:-42}"
DEST="${3:-$DEST_DEFAULT}"

[[ -n "$VARIANT" ]] || { echo "uso: $0 <variante> [seed] [destino]" >&2; exit 1; }

RUN_DIR="$REPO_DIR/$OUT_ROOT/${VARIANT}_seed${SEED}"
RUN_LOG="$REPO_DIR/$OUT_ROOT/${VARIANT}_seed${SEED}.log"
RAW="$DEST/raw"

[[ -d "$RUN_DIR" ]] || { echo "No existe el run_dir: $RUN_DIR" >&2; exit 1; }
mkdir -p "$RAW"

echo "run_dir : $RUN_DIR"
echo "destino : $RAW"

gzip -c "$RUN_DIR/logs.json.txt" > "$RAW/${VARIANT}_logs_json.txt.gz"
echo "  ${VARIANT}_logs_json.txt.gz"

if [[ -f "$RUN_LOG" ]]; then
  cp "$RUN_LOG" "$RAW/${VARIANT}_seed${SEED}.log"
  echo "  ${VARIANT}_seed${SEED}.log"
else
  echo "  AVISO: no hay stdout del run en $RUN_LOG (sin tiempos por epoca)" >&2
fi

cp "$RUN_DIR/train.log" "$RAW/${VARIANT}_train_hydra.log"
echo "  ${VARIANT}_train_hydra.log"

# V0-V2 no guardaron la config efectiva y la memoria la ha necesitado despues.
cp "$RUN_DIR/.hydra/config.yaml" "$RAW/${VARIANT}_hydra_config.yaml"
cp "$RUN_DIR/.hydra/overrides.yaml" "$RAW/${VARIANT}_hydra_overrides.yaml"
echo "  ${VARIANT}_hydra_config.yaml"
echo "  ${VARIANT}_hydra_overrides.yaml"

# La copia no conserva las mtime y el fin del run se deduce de la del log JSON.
cat > "$RAW/${VARIANT}_meta.json" <<EOF
{
  "variant": "${VARIANT}",
  "seed": ${SEED},
  "run_dir": "${OUT_ROOT}/${VARIANT}_seed${SEED}",
  "logs_json_mtime": "$(date -r "$RUN_DIR/logs.json.txt" '+%Y-%m-%d %H:%M:%S.%6N')",
  "exported_at": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
echo "  ${VARIANT}_meta.json"

echo "hecho."
