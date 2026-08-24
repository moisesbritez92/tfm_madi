# CLAUDE.md — TFM: encoders visuales en Diffusion Policy (Push-T)

Contexto operativo del proyecto. El aporte principal del TFM es comparar 5 backbones
visuales como `obs_encoder` de una Diffusion Policy sobre Push-T. Ver
`diffuser/experimento_encoder_pusht.md` para el diseño experimental completo.

## Topología: dos árboles distintos, no confundirlos

| Ruta | Qué es | ¿git? |
|---|---|---|
| `C:\Users\moise\Documents\0001_MADI\TFM` | Este repo. Documentación, notebooks, scripts. | Sí |
| `~/tfm/diffusion_policy` (WSL2 Ubuntu) | **Donde se entrena de verdad.** Copia de trabajo. | **No** |
| `diffuser/repo/diffusion_policy/` | Respaldo del repo upstream en Windows. | No (`.gitignore:8`) |

Los entrenamientos y sus resultados completos viven en WSL, que es la fuente autoritativa.
Los checkpoints seleccionados de V0, V1 y V2 también están copiados en
`diffuser/models/{V0,V1,V2}/` para inferencia en Windows; esa carpeta está ignorada por
git. Si se toca una config o un módulo del modelo en WSL, no está versionado: hay que
copiarlo a mano al repo.

## Entorno (WSL2 Ubuntu 24.04)

Es un **conda env**, no un venv de pip:

```bash
source ~/mambaforge/etc/profile.d/conda.sh
conda activate robodiff
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH   # necesario para cuDNN
cd ~/tfm/diffusion_policy
```

Python 3.9.15 · PyTorch 1.12.1+cu116 · torchvision 0.13.1 · diffusers 0.11.1 · timm 0.9.16
· huggingface_hub 0.25.2 (pin por `cached_download`) · robomimic 0.2.0 (`--no-deps`).

`run_encoder_exp.sh` ya hace el `source`/`activate`/`export` por su cuenta.

## Hardware y sus límites (importan de verdad)

- GPU: RTX 3070 Ti Laptop, **8 GB VRAM**. El UNet condicional son 277 M params; con AdamW
  + EMA + gradientes se van ~5,5 GB solo en pesos y estados. Entra, pero justo.
- Host: **15,7 GB RAM**, con WSL capado a 12 GB en `C:\Users\moise\.wslconfig`.
- **El replay buffer de Push-T ocupa 2,84 GB de RAM por proceso.** El `img` del zarr es
  `float32` (`<f4`, 25650×96×96×3) y `ReplayBuffer.copy_from_path` lo carga entero en
  memoria. Este dato explica casi todos los problemas de RAM del proyecto.

## Regla dura: un solo entrenamiento a la vez

**Lanzar siempre con `./run_encoder_exp.sh <variante> <seed>`. Nunca `python train.py` a mano.**

```bash
./run_encoder_exp.sh v1 42            # run nuevo
./run_encoder_exp.sh v1 42 --resume   # reanudar desde latest.ckpt
./run_encoder_exp.sh --status
./run_encoder_exp.sh --stop
```

El script está en `~/tfm/diffusion_policy/run_encoder_exp.sh`, versionado en
`diffuser/scripts/run_encoder_exp.sh`. Hace `flock` de instancia única, preflight de
VRAM/RAM/run_dir, `setsid nohup` para sobrevivir al cierre del terminal, y calibra la
velocidad antes de dejarlo horas corriendo.

### Por qué existe esta regla (incidente del 27/07/2026)

El run `v1_seed42` murió tras una sola época. Tres causas encadenadas:

1. **Se lanzaron dos procesos** sobre el mismo `hydra.run.dir`, con 43 s de diferencia.
   Dejó huella en `train.log` (dos inicializaciones) y en `logs.json.txt` (los
   `global_step` 0–19 duplicados exactos, una línea corrupta por escritura entrelazada).
2. **Se perdieron los overrides de Hydra.** Se lanzó solo con `num_epochs`, `seed` y
   `logging.mode`, así que heredó `n_envs: null` → 56 procesos pymunk (V0 usaba 8) y 8
   workers de dataloader (V0 usaba 2). Sumado a los 2×2,84 GB de replay buffer, desbordó
   la VM de 12 GB → swap → **73–90 s/it**, cuando V0 iba a 2,11 it/s.
3. **La VM de WSL se apagó a las 18:04:04** y mató al proceso superviviente.

El proceso duplicado murió antes con `c10::CUDAError: CUDA error: unknown error` (dos
contextos CUDA en 8 GB). Los cientos de `BrokenPipeError` en el log son ruido: los workers
de `AsyncVectorEnv` cayendo en cascada. **No busques el bug ahí.**

## Los overrides de Hydra no son opcionales

Estos son los que usó V0 y completó 500 épocas. El script los fija siempre:

```
dataloader.num_workers=2
dataloader.persistent_workers=false
val_dataloader.num_workers=0
task.env_runner.n_envs=8        # sin esto -> 56 procesos pymunk -> RAM desbordada
logging.mode=disabled           # evita el prompt de wandb en entorno no-TTY
```

Sobre `n_envs=8`: el rollout evalúa 56 condiciones iniciales (6 train + 50 test, seeds
100000–100049), cada una en su propio proceso. Con `n_envs=8` se hacen en 7 tandas.
**Las métricas son idénticas** — verificado en `pusht_image_runner.py:145-232`:
`n_chunks = ceil(n_inits/n_envs)` y la agregación itera `for i in range(n_inits)` sobre
las 56, no sobre `n_envs`. Solo cambia el tiempo de pared del rollout.

## Estado del experimento

| Variante | Backbone | Estado |
|---|---|---|
| V0 | ResNet-18 scratch | ✅ 500 épocas. Mejor `test_mean_score` **0,8645** (época 350) |
| V1 | ResNet-18 ImageNet frozen | ✅ 500 épocas. Mejor checkpoint **0,668** (época 150) |
| V2 | ResNet-18 ImageNet fine-tune | ⏹️ detenido en época 266. Mejor `test_mean_score` **0,6477** (época 150) |
| V3 | DINOv2 ViT-S/14 frozen | ⬜ pendiente |
| V4 | CLIP ViT-B/16 frozen | ⬜ pendiente |

V2 se detuvo de forma deliberada: después del máximo de la época 150, el score bajó a
0,5487 (época 200) y 0,5590 (época 250) mientras la pérdida de entrenamiento seguía
descendiendo, señal compatible con sobreajuste. No debe describirse como un run de 500
épocas completado.

Fase 1 = 1 seed (42) × 500 épocas × 5 variantes; se han ejecutado 3 de 5. Todas las
métricas anteriores agregan los 50 entornos de test con seeds 100000–100049; la seed 42
es la del entrenamiento. Fase 2 = seeds 43 y 44 sobre las 3 variantes seleccionadas.
Referencia de duración: V0 tardó ~17 h.

## Dónde están los resultados

```
data/outputs/encoder_exp/<variante>_seed<seed>/
├── .hydra/config.yaml + overrides.yaml   # config efectiva: comprobar aquí n_envs
├── checkpoints/                          # top-3 por test_mean_score + latest.ckpt
├── logs.json.txt                         # un JSON por línea, fuente de las curvas
└── media/                                # vídeos de rollout
```

Para inferencia gráfica en Windows están los notebooks
`diffuser/inferencia_v{0,1,2}_pusht.ipynb` y las copias de checkpoints en
`diffuser/models/V{0,1,2}/`. Los logs, configs efectivas y medios completos permanecen
en WSL.

`logs.json.txt` se lee con `read_json_log()` de `diffusion_policy/common/json_logger.py`.
Cuidado: `JsonLogger` abre en modo append y hace `json.loads` de la última línea al
arrancar, así que un fichero corrupto rompe el `--resume`. Si un run se ensucia, es más
limpio borrarlo entero que intentar repararlo.

Claves útiles por línea: `train_loss`, `val_loss`, `test/mean_score`, `train/mean_score`,
`train_action_mse_error`, `global_step`, `epoch`.

## Archivos clave del modelo

```
diffusion_policy/model/vision/pretrained_encoders.py   # TimmBackbone + factories V1-V4
diffusion_policy/model/vision/multi_image_obs_encoder.py
diffusion_policy/policy/diffusion_unet_image_policy.py
diffusion_policy/workspace/train_diffusion_unet_image_workspace.py
diffusion_policy/config/pusht_v{0..4}_*.yaml
```

`TimmBackbone` congela bien: `requires_grad=False`, `.eval()` forzado con override de
`train()`, y `torch.no_grad()` en el forward. No hay fuga de gradiente ahí.
