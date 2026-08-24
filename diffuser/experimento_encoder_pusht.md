# Experimento: Encoder visual en Diffusion Policy / Push-T

Aporte principal del TFM: comparar 5 backbones visuales como `obs_encoder`
de una *Diffusion Policy* sobre la tarea Push-T y medir su efecto en
desempeño, eficiencia muestral y coste computacional.

> Documento vivo. Última actualización: **22/05/2026**.

## 1. Entorno de ejecución

| Componente | Valor |
|---|---|
| Host | Windows 11 + WSL2 + Ubuntu 24.04 |
| GPU | NVIDIA RTX 3070 Ti Laptop (8 GB VRAM, CC 8.6) |
| Driver | 596.21 |
| Entorno virtual | **conda env `robodiff`** en `/home/moise/mambaforge/envs/robodiff` |
| Python | 3.9.15 |
| PyTorch | 1.12.1 + CUDA 11.6 (cudatoolkit) |
| torchvision | 0.13.1 |
| diffusers | 0.11.1 |
| timm | 0.9.16 |
| huggingface_hub | 0.25.2 (pin por compat con `cached_download`) |
| robomimic | 0.2.0 (`--no-deps`, no se usa MuJoCo) |

**No es un venv de pip, es un conda env.** Para entrar a la sesión de
trabajo desde Windows:

```powershell
wsl -d Ubuntu
```

Y dentro de WSL:

```bash
source ~/mambaforge/etc/profile.d/conda.sh
conda activate robodiff
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH   # necesario para cuDNN
cd ~/tfm/diffusion_policy
```

## 2. Dataset

Es el dataset oficial de Push-T publicado por los autores (Chi et al.).

| Atributo | Valor |
|---|---|
| Nombre | `pusht_cchi_v7_replay.zarr` |
| Ubicación local | `~/tfm/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr` |
| Tamaño | ~30 MB |
| Episodios | **206** demostraciones humanas (teleoperación) |
| Transiciones totales | 25 650 frames |
| Modalidades | `img (96×96×3 float32)`, `state (5,)`, `keypoint (9,2)`, `action (2,)`, `n_contacts (1,)` |
| Origen | Repo oficial: https://github.com/columbia-ai-robotics/diffusion_policy |
| Descarga | Ya bajado en sesión anterior (fecha de creación 27/02/2023) |

**Comando de descarga** (si hubiera que rehacer):

```bash
cd ~/tfm/diffusion_policy
mkdir -p data && wget -O data/pusht.zip \
  https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip data/pusht.zip -d data/ && rm data/pusht.zip
```

División train/val/test la maneja el `PushTImageDataset` (ver
`diffusion_policy/dataset/pusht_image_dataset.py`): val_ratio=0.02,
max_train_episodes=90 (≈90 de 206 para train, resto descartados — comportamiento
del paper original).

La **evaluación de success rate** no consume del zarr; usa el simulador
pymunk (`PushTImageRunner`, 50 episodios test con seeds 100000–100049).

## 3. Arquitectura compartida

Todas las variantes usan **el mismo workspace y la misma policy**, sólo cambia
el `rgb_model` del `MultiImageObsEncoder`:

```
DiffusionUnetImagePolicy
  ├── obs_encoder: MultiImageObsEncoder
  │   ├── rgb_model: <─── ESTO ES LO QUE VARIAMOS
  │   └── transforms: resize / crop / normalize (varían según rgb_model)
  ├── noise_scheduler: DDPM(100 train steps, fixed_small, eps prediction)
  └── unet1d_cond: down_dims=[512,1024,2048], obs_as_global_cond=True
```

Hiperparámetros fijos (heredados del config oficial):

- `horizon=16`, `n_obs_steps=2`, `n_action_steps=8`
- `num_inference_steps=100` (DDPM)
- `optimizer=AdamW`, `lr=1e-4`, `cosine` con 500 warmup steps
- `EMA` activo
- `num_epochs=1000` en config, **se entrenará a 500 en la fase pilot**
- `rollout_every=50`, `checkpoint_every=50`, `val_every=1`

## 4. Las 5 variantes

| ID | Backbone | Pesos | Estado | Resol. | Norm | Crop |
|---|---|---|---|---|---|---|
| **V0** | ResNet-18 | scratch | train | 96 nativo | ImageNet | RandomCrop 76 |
| **V1** | ResNet-18 (timm) | ImageNet-1k | **frozen** | resize 224 | ImageNet (interna) | — |
| **V2** | ResNet-18 (torchvision) | ImageNet-1k | fine-tune | resize 224 | ImageNet | — |
| **V3** | DINOv2 ViT-S/14 | LVD-142M | **frozen** + proj 512 | resize 224 | ImageNet (interna) | — |
| **V4** | CLIP ViT-B/16 | OpenAI-400M | **frozen** + proj 512 | resize 224 | OPENAI_CLIP (interna) | — |

Todas dan `obs_feature_dim = 512` a la salida del encoder para que el UNet
reciba la misma dimensión de *global cond* (comparación limpia).

### Cifras del dry-run (batch=64 excepto V4 con batch=32, GPU RTX 3070 Ti)

| Variante | Params total | Params train | Step ~ | Inferencia | VRAM peak |
|---|---|---|---|---|---|
| V0 | 288 M | 288 M | 0.5 s | 5.1 s/B=64 | 2.5 GB |
| V1 | 288 M | 277 M | 0.5 s | 6.1 s/B=64 | 3.1 GB |
| V2 | 288 M | 288 M | 0.5 s | 6.5 s/B=64 | 4.8 GB |
| V3 | 299 M | 277 M | 0.6 s | 9.2 s/B=64 | 3.1 GB |
| V4 | 363 M | 277 M | 0.6 s | 7.1 s/B=32 | 3.0 GB |

> "Step" = `compute_loss + backward` post-warmup. El UNet de 250 M params
> domina el total; el backbone visual es secundario en cómputo de
> entrenamiento pero clave en inferencia (los ViT son más lentos).

## 5. Archivos clave

```
~/tfm/diffusion_policy/
├── diffusion_policy/
│   ├── model/vision/
│   │   ├── model_getter.py                  # get_resnet (original del repo)
│   │   ├── multi_image_obs_encoder.py       # encoder agnóstico al backbone
│   │   └── pretrained_encoders.py           # NUEVO: TimmBackbone + factories
│   ├── policy/diffusion_unet_image_policy.py
│   ├── workspace/train_diffusion_unet_image_workspace.py
│   └── config/
│       ├── pusht_v0_resnet18_scratch.yaml         # NUEVO
│       ├── pusht_v1_resnet18_imagenet_frozen.yaml # NUEVO
│       ├── pusht_v2_resnet18_imagenet_ft.yaml     # NUEVO
│       ├── pusht_v3_dinov2_vits14_frozen.yaml     # NUEVO
│       ├── pusht_v4_clip_vitb16_frozen.yaml       # NUEVO
│       └── task/pusht_image.yaml
├── data/pusht/pusht_cchi_v7_replay.zarr
├── dryrun_encoders.py                       # valida configs sin entrenar
├── smoke_encoders.py                        # valida solo backbones
└── train.py                                 # entry-point oficial de Hydra
```

Copia de respaldo (Windows) en `c:\Users\moise\Documents\0001_MADI\TFM\diffuser\repo\diffusion_policy\`.

## 6. Plan de ejecución por fases

### Fase 0 — preparación ✅
- Setup WSL2 + conda + repo + dataset.
- `pretrained_encoders.py` + 5 configs + dry-run.

### Fase 1 — pilot (1 seed × 500 ep × 5 variantes) — **~75 h GPU**
Identificar las 2 mejores y la peor.

### Fase 2 — rigor (2 seeds extra × 3 variantes seleccionadas) — **~60 h GPU**
Validar significancia.

### Fase 3 — ablation de robustez (#4 del documento de aportes)
Reutiliza checkpoints, sólo evaluación con perturbaciones visuales.

## 7. Cómo lanzar un entrenamiento

**Siempre con `run_encoder_exp.sh`. Nunca `python train.py` a mano.**

Desde `~/tfm/diffusion_policy/` (el script activa el conda env por su cuenta):

```bash
./run_encoder_exp.sh v1 42            # run nuevo
./run_encoder_exp.sh v1 42 --resume   # reanudar desde latest.ckpt
./run_encoder_exp.sh --status         # PID, época, it/s, VRAM, RAM
./run_encoder_exp.sh --stop           # parada limpia del grupo de procesos
```

El script hace, en este orden: candado `flock` de instancia única, preflight de
VRAM/RAM/`run_dir`, fija todos los overrides de Hydra, lanza con `setsid nohup`
(sobrevive al cierre del terminal) y calibra la velocidad antes de dejarlo horas
corriendo. Copia versionada en `diffuser/scripts/run_encoder_exp.sh` del repo git.

> **Por qué no se lanza a mano.** El 27/07/2026 se perdió el run `v1_seed42` por
> lanzar el comando suelto dos veces sobre el mismo `hydra.run.dir` y, además,
> sin los overrides de memoria que sí llevaba V0. Detalle completo en `CLAUDE.md`.

Los overrides que el script fija siempre (son los del run V0 que completó 500 épocas):

```
training.num_epochs=500  training.seed=<seed>  training.resume=<bool>
dataloader.num_workers=2  dataloader.persistent_workers=false
val_dataloader.num_workers=0
task.env_runner.n_envs=8
logging.mode=disabled
hydra.run.dir=data/outputs/encoder_exp/<variante>_seed<seed>
```

Notas:
- `logging.mode=disabled` evita el prompt de wandb (env no-TTY).
- `task.env_runner.n_envs=8` es **obligatorio**: con el valor por defecto (`null`)
  el rollout abre 56 procesos pymunk a la vez y desborda la RAM de la VM. No altera
  las métricas — el runner trocea en 7 chunks y agrega sobre las 56 condiciones
  igualmente (`pusht_image_runner.py:145-232`).
- `dataloader.num_workers=2` / `val_dataloader.num_workers=0`: el replay buffer
  ocupa 2,84 GB en RAM (el `img` del zarr es float32), así que cada worker cuenta.
- `resume: True` en config: si se interrumpe, `--resume` reanuda desde el último
  checkpoint. Si el `logs.json.txt` quedó corrupto, borrar el run entero en vez de
  reanudar (`JsonLogger` parsea la última línea al abrir).
- Los checkpoints van a `<run_dir>/checkpoints/` (top-3 por
  `test_mean_score` + `latest.ckpt`, ~2,1 GB cada uno; `latest.ckpt` ~4,3 GB).

## 8. Cómo recuperar resultados

Cada run genera:

```
data/outputs/encoder_exp/v0_seed42/
├── .hydra/                       # configs efectivas
├── checkpoints/
│   ├── epoch=0050-test_mean_score=0.123.ckpt
│   ├── ...
│   └── latest.ckpt
├── logs.json.txt                 # métricas por época (línea por línea)
└── media/                        # vídeos de rollouts seleccionados
```

`logs.json.txt` es una secuencia de objetos JSON, uno por época, con
`train_loss`, `val_loss`, `test_mean_score`, `train_action_mse`, etc.
Ese fichero es la fuente para las curvas y tablas del TFM.

## 9. Reproducibilidad

- Seed por defecto: 42. Para Fase 2 usar 43, 44.
- `torch.use_deterministic_algorithms` NO está activo (lo desactiva
  `cuDNN`); el ruido del scheduler usa `torch.randn` con el seed global.
  Esperamos varianza moderada entre seeds — por eso planeamos múltiples
  semillas en las variantes ganadoras.
- Las versiones exactas de todas las libs están en
  `pip freeze > requirements_robodiff.txt` (pendiente de generar; añadir
  al cerrar el experimento).
