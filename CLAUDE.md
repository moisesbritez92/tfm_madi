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

## No reanudar V3 ni V4 sin el parche del planificador (bug del 28/08/2026)

`train_diffusion_unet_image_workspace.py` reconstruye el `lr_scheduler` en cada arranque,
porque es una variable local y **no se guarda en el checkpoint**. Lo reconstruía así:

```python
num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulate_every
last_epoch=self.global_step-1        # <- contador de LOTES
```

Pero `lr_scheduler.step()` solo se ejecuta cuando `global_step % gradient_accumulate_every == 0`,
es decir **una vez por paso de optimizador**. Al reanudar con `gradient_accumulate_every: 2`
el planificador se sitúa al doble de distancia de la que le toca, se pasa del final del
coseno y la tasa **vuelve a subir** en lugar de seguir bajando.

- **Solo afecta a V3 y V4** (`accumulate: 2`). V0, V1 y V2 usan `accumulate: 1`, así que la
  reanudación de V2 es válida y no hay que tocarla.
- **Solo se manifiesta al reanudar.** En un run nuevo `global_step` vale 0 y el cálculo es
  correcto: las ramas originales de V3 y V4 son sanas.
- Magnitud: en la época 150 de V3, la rama limpia da `lr` 5,08e-05 y la defectuosa 1,11e-08.
  En la época 199 de V4, 6,28e-09 frente a 2,14e-05. Tres mil veces en ambos sentidos.

Corregido con `last_epoch=(self.global_step // cfg.training.gradient_accumulate_every)-1`,
verificado en marcha (V3 reanudó en 4,97e-05 y V4 en 1,50e-05, alineados con sus ramas
limpias). **El árbol de WSL no está versionado**: si se recrea, hay que volver a aplicar el
parche desde `diffuser/scripts/train_diffusion_unet_image_workspace.py`.

### Los logs de WSL de V3 y V4 ya no coinciden con lo que reporta la memoria

Ese día se extendieron V3 y V4 hasta la época ~200 y **la extensión se descartó entera**.
La memoria reporta el estado preregistrado (V3 con 155 épocas, V4 con 200, ambas con última
evaluación en la 150), pero `logs.json.txt` en WSL conserva las ramas añadidas:

- V3 tiene **tres** evaluaciones de la época 150: 0,579 (original), 0,538 (defectuosa) y
  0,635 (reentrenada limpia). La última supera el máximo preregistrado de la época 100
  (0,6224), lo que habría movido el checkpoint seleccionado.
- V4 tiene dos de la época 150: 0,435 (original) y 0,388 (limpia). Su máximo no se mueve.

**`exportar_logs_wsl.sh` volvería a traer esas ramas.** Si hay que reexportar V3 o V4, hay
que recortar el log en el punto donde arranca la extensión, o el criterio de «última
aparición gana» seleccionará datos que la memoria no reporta.

Dato que sí merece recordarse: tres evaluaciones de estados de entrenamiento casi idénticos
se separan 0,097 en V3 y 0,047 en V4, del orden del error estándar del propio rollout
(~0,055). **La diferencia entre épocas vecinas de la curva de selección es ruido**, y eso
respalda el argumento de sesgo optimista que ya sostiene el capítulo de resultados.

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

## `V_Paper` es el modelo del artículo, y no es una sexta variante

`diffuser/models/V_Paper/epoch=0500-test_mean_score=0.884.ckpt` (4,0 GB) es el punto de
control que publicaron los autores para Push-T. **No entra en la tabla de «Estado del
experimento»** y no se le llama V5 en ningún sitio: es la referencia externa contra la que
se contrasta V0, no un brazo del experimento.

Su arquitectura no es la de V0-V4. Workspace `TrainDiffusionUnetHybridWorkspace`, política
`DiffusionUnetHybridImagePolicy`, y el codificador es **el de robomimic**: ResNet-18 +
`SpatialSoftmax` de 32 puntos clave + `Linear(64→64)`, con `crop_shape [84,84]`,
`obs_encoder_group_norm` y `eval_fixed_crop`. Es decir, **`V_Paper` es la prueba material de
la corrección factual nº 1**: el *spatial softmax* del artículo existe, pero pertenece al
codificador de robomimic y ninguna de las cinco variantes lo usa. Todo lo demás coincide con
V0: UNet `down_dims [512,1024,2048]`, DDPM de 100 pasos, `horizon 16 / n_obs 2 / n_action 8`
y la misma partición del dataset (`seed 42`, `val_ratio 0.02`, 90 episodios).

**Su `0,884` está medido sobre las semillas `4300000-4300049`**, el bloque con el que los
autores eligieron su propio punto de control. No es comparable con el `100000-100049` de
nuestra selección ni con el `200000-200199` de la prueba final. No citarlo junto a las
cifras del TFM sin decir esto.

### Dónde vive robomimic, y por qué `--no-deps`

- WSL `robodiff`: ya lo tenía, 0.2.0 con torch 1.12.1. `generate_paper_configs` importa.
- Windows `.venv_diffuser_infer`: se instaló el 31 de agosto de 2026 con
  `pip install robomimic==0.2.0 --no-deps`, más `h5py`, `termcolor` y `tensorboardX`.
  Los cuatro están pineados al final de `requirements_inference_windows.txt`.

El `--no-deps` **no es opcional**: robomimic declara torch, torchvision, numpy e imageio sin
techo y arrastraría el entorno de los cinco notebooks. `egl_probe` y `tensorboard` quedan sin
instalar a propósito; solo los importa `env_robosuite`, que no se toca.

Dato que ahorra una tarde: **`torchvision 0.21` todavía acepta el kwarg legacy
`resnet18(pretrained=False)`** por `handle_legacy_interface`, que es exactamente lo que hace
`robomimic.models.base_nets.ResNet18Conv`. Era el riesgo obvio de compatibilidad y está
descartado.

### La guarda de `cfg.policy.obs_encoder`

La política híbrida construye su codificador dentro de robomimic y **no tiene la clave
`cfg.policy.obs_encoder`**. Cualquier código que la lea sin guarda revienta con
`ConfigAttributeError` antes de cargar nada.

- Ya la tiene: `diffuser/v0_inference_utils.py:94` (`load_policy_bundle`) y
  `diffuser/scripts/evaluar_paper_bloque_test.py`.
- **Siguen sin ella**: `diffuser/scripts/evaluar_bloque_test.py:124`,
  `diffuser/scripts/memoria_gpu.py:165` y `memoria/scripts/latencia_inferencia.py:174`.
  Los tres funcionan con V0-V4 y fallarían con `V_Paper`.

### Inferencia gráfica en Windows

`diffuser/inferencia_paper_pusht.ipynb` y `diffuser/paper_inference_utils.py`. El módulo
**no** se llama `v5_inference_utils` a propósito: `servidor_politica.cargar_politica`
resuelve las variantes por ese patrón (`importlib.import_module(f"{variante}_inference_utils")`)
y la demo de Godot no debe cargar el modelo del artículo como si fuera una del experimento.

### La comparación con V0 va en WSL, no en Windows

Preregistro en `memoria/preregistro_comparacion_paper.md`, cerrado el 31 de agosto de 2026
antes de ejecutar. Compara V0 con `V_Paper` sobre el mismo bloque `200000-200199`, con
permutación por inversión de signo, TOST de margen `δ = 0,05` (responde al hallazgo M9) y
**dos realizaciones de ruido de difusión por brazo**, semillas base `20260827` y `20260831`
(responde al M5).

**La evaluación de `V_Paper` se hace en WSL con `robodiff`, no en `.venv_diffuser_infer`**,
aunque en Windows también carga. Un cambio de torch entre los dos brazos sería una
diferencia de procedimiento dentro del propio contraste pareado.

Piezas: `diffuser/scripts/evaluar_paper_bloque_test.{py,sh}` y
`memoria/scripts/analisis_comparacion_paper.py`. El hermano `evaluar_bloque_test.py`
**no se toca**: es el artefacto que produjo los cinco JSON congelados.

Lo que el contraste **no** aísla, y hay que repetirlo en cualquier texto que lo cite:
difieren a la vez el codificador, el presupuesto de entrenamiento (3050 épocas con punto de
control en la 500, frente a 500 con punto de control en la 350), el bloque de selección y el
linaje de implementación. La unidad de inferencia sigue siendo el artefacto entrenado.

### Resultado, ejecutado el 31 de agosto de 2026

Los tres portones pasaron: V0 reprodujo `prueba_v0.json` bit a bit sobre ocho condiciones,
V_Paper dio **0,8623** sobre su propio bloque `4300000-4300049` frente al 0,884 publicado
(desvío 0,0217 con tolerancia 0,07), y los dos brazos muestrean el mismo primer tensor de
ruido.

| Brazo | Realización A (`20260827`) | Realización B (`20260831`) | Media |
|---|---|---|---|
| V0 | 0,8719 | 0,9019 | **0,8869** |
| V_Paper | 0,8500 | 0,8422 | **0,8461** |

Diferencia media pareada **+0,0408** a favor de V0 (EE 0,0189), IC95 BCa
`[+0,0049, +0,0792]`, permutación **p = 0,036**. La casilla de la regla de decisión es
**«diferencia relevante»**, pero hay que leerla con cuidado y así debe escribirse: el
IC90 es `[+0,0107, +0,0736]` y **no cabe en ±0,05**, así que no se puede declarar
equivalencia; y la estimación puntual, 0,041, **está por debajo del margen**, así que
tampoco está demostrado que la ventaja lo supere. Lo que sí queda establecido es que **V0 no
es peor que el punto de control publicado**. Eso valida la premisa que el evaluador da por
buena en las líneas 470 y 582.

Dos cosas más que conviene no volver a descubrir:

- **La permutación y Wilcoxon discrepan cruzando el 0,05** (0,036 frente a 0,092). Manda la
  permutación, que es la primaria y apunta a la media, que es el estimando; Wilcoxon opera
  sobre rangos. Es el mismo tipo de discrepancia de borde que ya aparecía en V2-V4 y V3-V4.
- **En la tasa de éxito no hay ninguna diferencia**: 101/200 y 100/200 en V0 frente a 94/200
  y 92/200, McNemar exacto `p = 0,52` y `p = 0,47`.

**El dato más incómodo, y el más útil: cambiar solo la semilla del ruido movió la media de V0
de 0,8719 a 0,9019**, tres centésimas, el 73 % del tamaño de la propia diferencia entre los
dos brazos. La varianza intra-condición entre realizaciones es 0,0272 en V0 y 0,0239 en
V_Paper. Es decir, **el 0,872 que la memoria reporta para V0 es una realización del ruido, no
una constante**, y eso es exactamente lo que denunciaba el hallazgo M5. Ninguna cifra de la
prueba final de las cinco variantes lleva esa incertidumbre incorporada.

Datos: `memoria/datos/comparacion_paper_{episodios,resumen,contrastes}.csv` y los cuatro
JSON en `logs_entrenamiento/prueba_final/`.

## `diffuser/godot/` es un proyecto Godot, y no produce cifras reportables

Desde el 29 de agosto de 2026 esa carpeta dejó de ser solo el manual de Godot: es la raíz
de un proyecto Godot 4.7.2 que reimplementa Push-T y pone el punto de control congelado de
V0 en el bucle. El manual vive ahora en `diffuser/godot/manual/`, con dos módulos nuevos
(06 el puente y el port, 07 la escena y los modos).

**Es una demostración visual para la defensa.** Lo que muestre el panel es ilustrativo. El
resultado del TFM sigue siendo la pasada preregistrada sobre `200000-200199` que está en
`logs_entrenamiento/prueba_final/`. No mezclar las dos cosas en ningún texto.

```powershell
cd diffuser\godot
.\lanzar.ps1                  # condición A: Godot simula, la imagen la dibuja Python
.\lanzar.ps1 -Obs godot       # condición B: la política ve los píxeles 3D de Godot
.\lanzar.ps1 -Modo reproducir # sin GPU ni servidor; la red de seguridad de la defensa
```

Dos condiciones, una bandera del servidor, el mismo código en Godot:

- **A, `--obs estado`.** Godot manda el estado y `servidor/rasterizador_pusht.py` dibuja el
  96×96 con el código del entrenamiento. Solo ha cambiado el motor de física.
- **B, `--obs godot`.** La observación sale del `SubViewport` ortográfico cenital. Han
  cambiado la física y los píxeles.

La física se simula en **2D** aunque la vista sea 3D. Con gravedad y fricción de mesa la T
volcaría y la tarea dejaría de ser la que V0 aprendió.

Cuatro cosas que ya están comprobadas y no hay que volver a descubrir:

1. **`space.damping = 0` es amortiguación total**, no rozamiento suave: anula la velocidad
   al principio de cada subpaso. Se replica en `_integrate_forces` de `bloque_t.gd`. Sin
   eso la pieza conserva inercia y la tarea cambia.
2. **El centro de gravedad es `(0, 45)`, no el origen**, así que fijar el ángulo desplaza la
   posición. Para reproducir una pose conocida: ángulo primero, posición después. El orden
   `legacy` de `_set_state` es el contrario, pero eso pertenece al muestreo del reinicio.
3. **El rasterizador arrastraba puntos de contacto** del reinicio aleatorio con el que se
   construye. `_vaciar_contactos` lo deja determinista; a cambio nunca dibuja contactos,
   uno o dos píxeles de 9216.
4. **El estado inicial lo sortea Python**, con el `PushTEnv` de verdad, y Godot recibe los
   cinco números ya resueltos. Verificado exacto contra `seed(s); reset()`.
5. **Los `_process` corren antes del dibujado.** Al capturar las dos observaciones de la
   condición B, `vista3d._process` sobrescribía la pose histórica por la actual y las dos
   imágenes salían iguales, sin ningún síntoma salvo una puntuación algo peor (0,9159 en
   vez de 0,9453 en la semilla 10000). Lo evita el cerrojo `_congelado`, y el servidor
   avisa por consola si vuelve a pasar.

Referencia de las comprobaciones, para detectar una regresión: cobertura frente a shapely
3,1e-07; física 2,3 px de salto en el primer contacto y 0,30 px de separación tras 250
pasos de control; observación de Godot frente al renderizador original, 2,9 % de píxeles
con diferencia mayor que 8 y media 1,0 sobre 255. Los comandos están en el módulo 06.

El servidor ya no está atado a V0: `--variante {v0..v4}` pone cualquiera de los cinco
puntos de control congelados en el bucle, y Godot lee cuál es de la respuesta a `hola`.
`perturbacion={ninguna,t_roja,sombras}` cambia la escena y solo afecta a la condición B.
Lo que salga de ahí va a `diffuser/godot/perturbaciones.md`, que es una bitácora y **no
una medición**: un episodio por celda y semillas elegidas condicionando en el éxito. Dos
cosas de ahí conviene no volver a descubrir:

- **Las semillas se eligen con `servidor/elegir_semillas.py`, que cruza dos filtros.** El
  segundo no es evidente: la puntuación es el *máximo* de la cobertura del episodio, así
  que una condición inicial que ya arranca parcialmente resuelta da el mismo número para
  todas las variantes y todas las perturbaciones. Las semillas 200007 y 200019 arrancan a
  0,665 y 0,373, e invalidaron un barrido entero.
- **V3 no sobrevive al cambio de simulador.** Sobre cinco semillas donde V0 y V3 puntúan
  las dos 1,000 en el preregistro, en Godot condición B V0 promedia 0,995 y V3, 0,429. Y
  con la pieza pintada de rojo las dos caen a ~0,13 y 0,00. El preentrenamiento congelado
  no compró invariancia.
- **El fallo por color es una pendiente, no un acantilado** (`diffuser/godot/barrido_color.md`).
  Correlaciona −0,97 con la distancia RGB, y un gris neutro degrada tanto como un rojo a
  la misma distancia: no hay clasificación por tono, hay rasgos sintonizados en un punto
  del espacio de color. La tolerancia son ~30 unidades RGB. El candidato obvio para
  arreglarlo es aumento fotométrico, que V0 no tuvo: su único aumento es el recorte.

**El tiempo real es imposible y no hay que pelearlo**: V0 tarda 1743,9 ms de mediana por
llamada y cada llamada cubre 0,8 s de simulación, así que el techo es ×0,46. El panel lo
rotula.

**Los episodios no son reproducibles bit a bit.** La semilla 200003 en la condición A dio
0,9726 en 183 pasos una vez y 0,9545 en 242 otra. El ruido de difusión sí está sembrado;
lo que no es determinista son las convoluciones en GPU, y la simulación con contactos
amplifica la diferencia. Por eso el flujo para la defensa es `grabar` y luego `reproducir`,
no relanzar en vivo esperando el mismo episodio.

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
