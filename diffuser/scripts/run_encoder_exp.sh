#!/usr/bin/env bash
#
# run_encoder_exp.sh - lanzador robusto para el experimento de encoders (TFM Push-T)
#
# Sustituye al `python train.py` suelto. Existe porque el 27/07/2026 el run
# v1_seed42 murio por tres causas que este script hace imposibles:
#   1. Se lanzaron DOS procesos sobre el mismo hydra.run.dir  -> flock de instancia unica
#   2. Se perdieron los overrides de Hydra que si uso V0      -> van fijos aqui, no a mano
#   3. Al cerrarse la sesion de WSL murio el proceso          -> setsid nohup
#
# Uso:
#   ./run_encoder_exp.sh v1 42            # run nuevo (falla si el run_dir ya existe)
#   ./run_encoder_exp.sh v1 42 --resume   # reanuda desde latest.ckpt
#   ./run_encoder_exp.sh --status
#   ./run_encoder_exp.sh --stop
#
set -euo pipefail

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$HOME/tfm/diffusion_policy}"
OUT_ROOT="data/outputs/encoder_exp"
LOCK_FILE="/tmp/encoder_exp.lock"
PID_FILE="/tmp/encoder_exp.pid"
CONDA_SH="$HOME/mambaforge/etc/profile.d/conda.sh"
CONDA_ENV="robodiff"

NUM_EPOCHS="${NUM_EPOCHS:-500}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-7000}"   # el UNet de 277M + AdamW + EMA no cabe en menos
MIN_RAM_MIB="${MIN_RAM_MIB:-8192}"     # replay buffer float32 = 2.84 GB + torch + envs
CALIB_WAIT="${CALIB_WAIT:-180}"        # segundos antes de reportar el s/it real

# it/s de referencia medidos en el run V0 completo (500 epocas, 2.11 it/s).
# V1-V4 resizean a 224 (5.4x pixeles) pero el backbone va en no_grad y el UNet
# -que domina el coste- es identico, asi que el rango sano es ~1.2-1.9 it/s.
CALIB_MIN_IT_S="1.0"

declare -A CONFIGS=(
  [v0]="pusht_v0_resnet18_scratch"
  [v1]="pusht_v1_resnet18_imagenet_frozen"
  [v2]="pusht_v2_resnet18_imagenet_ft"
  [v3]="pusht_v3_dinov2_vits14_frozen"
  [v4]="pusht_v4_clip_vitb16_frozen"
)

# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; BLD=$'\033[1m'; RST=$'\033[0m'
info() { printf '%s==>%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '%s[!]%s %s\n' "$YEL" "$RST" "$*"; }
die()  { printf '%s[X]%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

usage() {
  cat <<EOF
${BLD}Uso:${RST}
  $(basename "$0") <variante> <seed> [--resume] [--no-wait]
  $(basename "$0") --status
  $(basename "$0") --stop

  variante : ${!CONFIGS[*]}
  seed     : entero (42 para la fase pilot; 43/44 para la fase 2)
  --resume : reanuda desde latest.ckpt en vez de exigir un run_dir vacio
  --no-wait: no esperar a la calibracion de velocidad, devolver el prompt ya
EOF
}

# El lock autoritativo lo mantiene el proceso de entrenamiento (FD 9 heredado
# por python via exec). Esto es solo un sondeo para dar un mensaje rapido.
lock_is_free() { ( exec 9>"$LOCK_FILE"; flock -n 9 ); }

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid; pid=$(cut -d' ' -f1 "$PID_FILE" 2>/dev/null) || return 1
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
preflight() {
  local run_dir="$1" resume="$2" config="$3"

  [[ -d "$REPO_DIR" ]] || die "No existe REPO_DIR: $REPO_DIR"
  [[ -f "$REPO_DIR/train.py" ]] || die "No existe $REPO_DIR/train.py"
  [[ -f "$REPO_DIR/diffusion_policy/config/${config}.yaml" ]] \
    || die "No existe la config: diffusion_policy/config/${config}.yaml"
  [[ -d "$REPO_DIR/data/pusht/pusht_cchi_v7_replay.zarr" ]] \
    || die "Falta el dataset: data/pusht/pusht_cchi_v7_replay.zarr"

  # 1. Ningun proceso CUDA vivo (un huerfano de un run anterior corrompe el siguiente)
  local cuda_pids
  cuda_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  if [[ -n "${cuda_pids//[[:space:]]/}" ]]; then
    nvidia-smi
    die "Hay procesos usando la GPU (PIDs: ${cuda_pids//$'\n'/ }). Liberala antes de lanzar."
  fi

  # 2. VRAM libre
  local vram_free
  vram_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [[ -n "$vram_free" ]] || die "nvidia-smi no responde. Revisa el driver de WSL."
  (( vram_free >= MIN_VRAM_MIB )) \
    || die "Solo ${vram_free} MiB de VRAM libres (minimo ${MIN_VRAM_MIB})."
  info "VRAM libre: ${vram_free} MiB"

  # 3. RAM disponible en la VM de WSL
  local ram_avail
  ram_avail=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 ))
  (( ram_avail >= MIN_RAM_MIB )) \
    || die "Solo ${ram_avail} MiB de RAM disponibles (minimo ${MIN_RAM_MIB}). El replay buffer solo ya ocupa 2.8 GB."
  info "RAM disponible: ${ram_avail} MiB"

  # 4. run_dir limpio salvo que se pida resume explicitamente
  if [[ -d "$REPO_DIR/$run_dir" ]]; then
    if [[ "$resume" != "true" ]]; then
      die "El run_dir ya existe: $run_dir
    Usa --resume para continuarlo, o borralo si quieres empezar de cero:
      rm -rf $REPO_DIR/$run_dir $REPO_DIR/${run_dir}.log"
    fi
    [[ -f "$REPO_DIR/$run_dir/checkpoints/latest.ckpt" ]] \
      || die "--resume pedido pero no hay $run_dir/checkpoints/latest.ckpt"
    info "Reanudando desde $run_dir/checkpoints/latest.ckpt"
  fi

  # 5. Entorno
  [[ -f "$CONDA_SH" ]] || die "No encuentro conda: $CONDA_SH"
}

# --------------------------------------------------------------------------
# Modo interno: es lo que corre ya detachado, con el lock en la mano
# --------------------------------------------------------------------------
if [[ "${1:-}" == "--exec" ]]; then
  variant="$2"; seed="$3"; resume="$4"
  config="${CONFIGS[$variant]}"
  run_dir="$OUT_ROOT/${variant}_seed${seed}"

  # Lock autoritativo. FD 9 sobrevive al exec y lo hereda python: el lock se
  # mantiene exactamente mientras dure el entrenamiento, ni mas ni menos.
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[X] ABORTADO: ya hay otro entrenamiento con el lock ($LOCK_FILE)." >&2
    exit 3
  fi

  cd "$REPO_DIR"
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
  export HYDRA_FULL_ERROR=1
  export PYTHONUNBUFFERED=1
  # Limita la fragmentacion del allocator en una GPU de 8 GB (torch 1.12)
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

  echo "# ---------------------------------------------------------------"
  echo "# variante   : $variant ($config)"
  echo "# seed       : $seed"
  echo "# run_dir    : $run_dir"
  echo "# resume     : $resume"
  echo "# inicio     : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "# python     : $(which python)"
  echo "# lock       : $LOCK_FILE (pid $$)"
  echo "# ---------------------------------------------------------------"

  # $$ no cambia al hacer exec, asi que este es ya el PID de python.
  echo "$$ $variant $seed" > "$PID_FILE"

  # Los overrides NO son opcionales: son exactamente los que uso el run V0 que
  # completo 500 epocas. Sin n_envs=8 el rollout abre 56 procesos pymunk y
  # revienta la RAM de la VM. Verificado en pusht_image_runner.py:145-232 que
  # trocear en chunks no altera las metricas (agrega sobre n_inits=56).
  exec python train.py \
    --config-dir=diffusion_policy/config \
    --config-name="$config" \
    training.num_epochs="$NUM_EPOCHS" \
    training.seed="$seed" \
    training.resume="$resume" \
    dataloader.num_workers=2 \
    dataloader.persistent_workers=false \
    val_dataloader.num_workers=0 \
    task.env_runner.n_envs=8 \
    logging.mode=disabled \
    hydra.run.dir="$run_dir"
fi

# --------------------------------------------------------------------------
# --status
# --------------------------------------------------------------------------
if [[ "${1:-}" == "--status" ]]; then
  if pid=$(running_pid); then
    read -r _ variant seed < "$PID_FILE"
    run_dir="$REPO_DIR/$OUT_ROOT/${variant}_seed${seed}"
    printf '%sCORRIENDO%s  pid=%s  variante=%s  seed=%s\n' "$GRN" "$RST" "$pid" "$variant" "$seed"
    printf '  arrancado : %s\n' "$(ps -o lstart= -p "$pid" | xargs)"
    printf '  cpu/mem   : %s\n' "$(ps -o %cpu=,%mem=,rss= -p "$pid" | xargs)"
    if [[ -f "$run_dir/logs.json.txt" ]]; then
      printf '  epoca     : %s\n' "$(tail -1 "$run_dir/logs.json.txt" | grep -oE '"epoch": [0-9]+' | grep -oE '[0-9]+' || echo '?')"
      printf '  lineas log: %s\n' "$(wc -l < "$run_dir/logs.json.txt")"
    fi
    if [[ -f "${run_dir}.log" ]]; then
      printf '  velocidad : %s\n' "$(tr '\r' '\n' < "${run_dir}.log" | grep -oE '[0-9.]+(it/s|s/it)' | tail -1 || echo '?')"
    fi
    printf '  ckpts     : %s\n' "$(ls "$run_dir/checkpoints" 2>/dev/null | wc -l)"
  else
    printf '%sPARADO%s (sin entrenamiento activo)\n' "$YEL" "$RST"
    lock_is_free || warn "El lock $LOCK_FILE sigue tomado por otro proceso."
  fi
  echo
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
  free -h | head -2
  exit 0
fi

# --------------------------------------------------------------------------
# --stop
# --------------------------------------------------------------------------
if [[ "${1:-}" == "--stop" ]]; then
  pid=$(running_pid) || { info "No hay nada corriendo."; rm -f "$PID_FILE"; exit 0; }
  # setsid hizo al entrenamiento lider de su propia sesion, asi que PGID == PID.
  # Matar el grupo entero se lleva por delante los workers de AsyncVectorEnv y
  # del dataloader; si no, quedan huerfanos ocupando RAM y VRAM.
  info "Parando el grupo de procesos $pid (SIGTERM)..."
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    warn "No respondio a SIGTERM, enviando SIGKILL."
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    sleep 2
  fi
  pkill -f 'diffusion_policy.*train.py' 2>/dev/null || true
  rm -f "$PID_FILE"
  info "Parado. Estado final:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 0
fi

# --------------------------------------------------------------------------
# Lanzamiento
# --------------------------------------------------------------------------
[[ $# -ge 2 ]] || { usage; exit 1; }

VARIANT="$1"; SEED="$2"; shift 2
RESUME="false"; WAIT_CALIB="true"
for arg in "$@"; do
  case "$arg" in
    --resume)  RESUME="true" ;;
    --no-wait) WAIT_CALIB="false" ;;
    *) die "Argumento desconocido: $arg" ;;
  esac
done

[[ -v CONFIGS[$VARIANT] ]] || die "Variante desconocida '$VARIANT'. Validas: ${!CONFIGS[*]}"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "El seed debe ser un entero, no '$SEED'"

CONFIG="${CONFIGS[$VARIANT]}"
RUN_DIR="$OUT_ROOT/${VARIANT}_seed${SEED}"
RUN_LOG="$REPO_DIR/${RUN_DIR}.log"

if ! lock_is_free; then
  if pid=$(running_pid); then
    read -r _ v s < "$PID_FILE"
    die "Ya hay un entrenamiento corriendo (pid $pid, $v seed $s).
    Solo puede haber uno: la GPU es de 8 GB y cada proceso carga 2.8 GB de replay buffer.
    Mira el estado con: $(basename "$0") --status
    Paralo con      : $(basename "$0") --stop"
  fi
  die "El lock $LOCK_FILE esta tomado por un proceso desconocido. Revisa con: fuser -v $LOCK_FILE"
fi

info "Preflight para ${BLD}${VARIANT}${RST} seed ${BLD}${SEED}${RST} (${CONFIG})"
preflight "$RUN_DIR" "$RESUME" "$CONFIG"

mkdir -p "$(dirname "$RUN_LOG")"
info "Lanzando (detachado, sobrevive al cierre del terminal)..."
setsid nohup bash "$(readlink -f "$0")" --exec "$VARIANT" "$SEED" "$RESUME" \
  > "$RUN_LOG" 2>&1 < /dev/null &

sleep 8
if ! pid=$(running_pid); then
  warn "El proceso no arranco. Ultimas lineas de $RUN_LOG:"
  tail -25 "$RUN_LOG" >&2
  exit 1
fi

info "En marcha. pid=${BLD}${pid}${RST}"
info "  log      : $RUN_LOG"
info "  run_dir  : $REPO_DIR/$RUN_DIR"
info "  seguir   : tail -f $RUN_LOG"
info "  estado   : $(basename "$0") --status"
info "  parar    : $(basename "$0") --stop"

[[ "$WAIT_CALIB" == "true" ]] || exit 0

# --------------------------------------------------------------------------
# Calibracion: no dejar 500 epocas corriendo sin saber si la velocidad es sana
# --------------------------------------------------------------------------
echo
info "Calibrando velocidad (${CALIB_WAIT}s)... Ctrl-C aqui NO mata el entrenamiento."
for _ in $(seq 1 "$CALIB_WAIT"); do
  kill -0 "$pid" 2>/dev/null || { warn "El proceso murio durante la calibracion:"; tail -30 "$RUN_LOG" >&2; exit 1; }
  sleep 1
done

speed=$(tr '\r' '\n' < "$RUN_LOG" | grep -oE '[0-9.]+(it/s|s/it)' | tail -1 || true)
echo
if [[ -z "$speed" ]]; then
  warn "No he podido leer la velocidad todavia. Comprueba a mano: tail -f $RUN_LOG"
  exit 0
fi

printf '  velocidad medida : %s%s%s   (referencia V0: 2.11it/s)\n' "$BLD" "$speed" "$RST"
if [[ "$speed" == *s/it ]]; then
  warn "Esta reportando SEGUNDOS POR ITERACION, no it/s: el entrenamiento va lentisimo."
  warn "Sintoma del incidente del 27/07/2026 (RAM desbordada -> swap). Revisa:"
  warn "  free -h        -> si Swap used > 0, hay problema"
  warn "  nvidia-smi     -> solo deberia haber 1 proceso"
  warn "Si es asi: $(basename "$0") --stop  y diagnosticar antes de gastar horas de GPU."
else
  info "Velocidad en rango sano. ~500 epocas ≈ 20-30 h."
fi
info "Estado completo: $(basename "$0") --status"
