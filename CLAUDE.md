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
Los checkpoints seleccionados de V0 a V4 también están copiados en
`diffuser/models/{V0..V4}/` para inferencia en Windows; esa carpeta está ignorada por
git y ocupa unos 60 GB. Si se toca una config o un módulo del modelo en WSL, no está
versionado: hay que copiarlo a mano al repo.

Para traer un run nuevo desde WSL hay dos scripts, y **no se hace a mano**:

```bash
# logs, config efectiva y metadatos -> logs_entrenamiento/raw/
wsl -d Ubuntu -- bash -c "/tmp/exportar_logs_wsl.sh v3 42"   # diffuser/scripts/exportar_logs_wsl.sh
# CSV de tiempos por epoca + entrada en resumen.json
python memoria/scripts/resumen_entrenamiento.py v3 v4
```

Los checkpoints se copian aparte con `rsync -ah --info=progress2` desde el lado WSL (la
escritura a `/mnt/c` va por 9p, ~50 MB/s) y se verifican comparando `sha256sum`.

## Entorno (WSL2 Ubuntu 24.04)

Es un **conda env**, no un venv de pip:

```bash
source ~/mambaforge/etc/profile.d/conda.sh
conda activate robodiff
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH   # necesario para cuDNN
cd ~/tfm/diffusion_policy
```

Python 3.9.15 · PyTorch 1.12.1+cu116 · torchvision 0.13.1 · diffusers 0.11.1 · timm 0.9.16
· huggingface_hub 0.25.2 (pin por `cached_download`) · robomimic 0.2.0 (`--no-deps`)
· hydra-core 1.2.0 · zarr 2.12.0.

Ojo: el `requirements.txt` de la raíz **no** describe este entorno, sino el de inferencia
en Windows (torch 2.6.0+cu124, hydra-core 1.3.2, zarr 2.18.7). No copiar versiones de ahí.

`run_encoder_exp.sh` ya hace el `source`/`activate`/`export` por su cuenta.

## Hardware y sus límites (importan de verdad)

- GPU: RTX 3070 Ti Laptop, **8 GB VRAM**. El UNet condicional son 277 M params; con AdamW
  + EMA + gradientes se van ~5,5 GB solo en pesos y estados. Entra, pero justo.
- Host: **15,7 GB RAM**, con WSL capado a 12 GB en `C:\Users\moise\.wslconfig`.
- **El replay buffer de Push-T ocupa 2,84 GB de RAM por proceso.** El `img` del zarr es
  `float32` (`<f4`, 25650×96×96×3) y `ReplayBuffer.copy_from_path` lo carga entero en
  memoria. Este dato explica casi todos los problemas de RAM del proyecto. Para analizar
  el zarr sin pagar ese coste, abrirlo con `zarr.open(..., 'r')` y leer solo lo necesario.

## El reparto del dataset no es el que sugiere «206 demostraciones»

De los 206 episodios, **el entrenamiento usa 90 (10.726 ventanas de horizonte 16, 168
actualizaciones por época a lote efectivo 64), la validación 4 y se descartan 112**, el
54,4 %. `task/pusht_image.yaml` fija `dataset.seed: 42` como literal, no como `${seed}`:
la partición **no depende de la semilla de entrenamiento** y será idéntica en la fase 2.
Las demostraciones son las semillas 0 a 205 del propio entorno, y las seis condiciones
`train/` del evaluador (semillas 0-5) cayeron todas en el descarte, así que
`train/mean_score` **no mide condiciones vistas durante el ajuste**. Lo calcula todo
`diffuser/scripts/caracterizar_dataset.py`.

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
| V3 | DINOv2 ViT-S/14 frozen | ⏹️ detenido en época 154 (presupuesto 300). Mejor **0,6224** (época 100) |
| V4 | CLIP ViT-B/16 frozen | ✅ 200 épocas (presupuesto agotado). Mejor **0,5351** (época 100) |

V2 se detuvo de forma deliberada: después del máximo de la época 150, el score bajó a
0,5487 (época 200) y 0,5590 (época 250) mientras la pérdida de entrenamiento seguía
descendiendo, señal compatible con sobreajuste. No debe describirse como un run de 500
épocas completado.

**El presupuesto de épocas no fue el mismo para todas las variantes**: 500 en V0 y V1,
300 en V2 y V3, 200 en V4. Se recortó a medida que el coste por época crecía. Por eso V3
y V4 solo tienen 4 evaluaciones de rollout, frente a las 10 de V0 y V1, y por eso **el
tiempo total no sirve para comparar variantes**: hay que usar los minutos por época.

Fase 1 = 1 seed (42) × 5 variantes, **completada el 26 de agosto de 2026**. Todas las
métricas anteriores agregan los 50 entornos de test con seeds 100000–100049; la seed 42
es la del entrenamiento. Referencia de duración: V0 tardó ~17 h; V1, ~96 h.

**Esas cifras son del conjunto de selección y están sesgadas al alza: no son el
resultado.** El 27 de agosto de 2026 los cinco checkpoints congelados se evaluaron una
sola vez sobre el bloque disjunto `200000-200199` (n = 200), con protocolo preregistrado
en `memoria/preregistro_prueba_final.md` (commit `8bc22e7`). **Resultado final: V0 0,872 ·
V1 0,649 · V2 0,586 · V3 0,578 · V4 0,490.** La ordenación no cambia y la ventaja de V0
crece. Usar estas cifras, no las de la tabla, en cualquier texto que reporte resultados.

**Fase 2 (seeds 43 y 44) queda fuera de alcance**, por decisión del 27 de agosto de 2026:
no se lanza más cómputo y pasa a trabajo futuro. La consecuencia es inferencial y hay que
respetarla en toda la redacción: con una ejecución por variante, **la unidad sobre la que
se infiere es el artefacto entrenado, no la estrategia de entrenamiento**.

### Dos correcciones factuales (verificadas el 27/08/2026, no repetir los errores)

1. **Ninguna variante usa `spatial softmax`.** La config `pusht_image` pasa por
   `MultiImageObsEncoder` + `model_getter.get_resnet`, que hace `fc = Identity`: es
   promediado global en V0, V1 y V2, y componente de clase en V3 y V4. El *spatial
   softmax* del artículo pertenece al codificador de robomimic. Lo que sí distingue a V0
   es `use_group_norm: True` (por eso su checkpoint no tiene `num_batches_tracked`).
2. **V1 y V2 no parten del mismo archivo de pesos.** V1 usa
   `timm.create_model("resnet18", pretrained=True)` → **`resnet18.a1_in1k`**; V2 usa
   `torchvision` con `weights: IMAGENET1K_V1`. Verificado tensor a tensor contra el
   checkpoint congelado de V1: **120/120 idénticos con timm, 0/120 con torchvision**. El
   contraste V1–V2 **no** aísla congelación frente a ajuste fino.

## Dónde están los resultados

```
data/outputs/encoder_exp/<variante>_seed<seed>/
├── .hydra/config.yaml + overrides.yaml   # config efectiva: comprobar aquí n_envs
├── checkpoints/                          # top-3 por test_mean_score + latest.ckpt
├── logs.json.txt                         # un JSON por línea, fuente de las curvas
└── media/                                # vídeos de rollout
```

Para inferencia gráfica en Windows están los notebooks
`diffuser/inferencia_v{0,1,2,3,4}_pusht.ipynb` y las copias de checkpoints en
`diffuser/models/V{0..4}/`. Los logs, configs efectivas y medios completos permanecen
en WSL, con copia de los logs en `logs_entrenamiento/`.

El entorno de inferencia de Windows es `.venv_diffuser_infer` (Python 3.11, torch
2.6.0+cu124, **timm 1.0.7**). Aunque se entrenó con timm 0.9.16, los checkpoints de V3 y
V4 cargan sin desajustes: `load_policy_bundle` fuerza `rgb_model.pretrained = False`, así
que no se descarga nada de HuggingFace y `load_state_dict` no se queja.

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
