# Prompt — M4: medir y reportar latencia de inferencia y pico de VRAM

> Prompt autocontenido para ejecutar la corrección **P2.3** del informe de evaluación
> (`EVALUACION_TFM.md`, §22 y hallazgo **M4**). Pegar en una sesión de Claude Code abierta
> en la raíz del repo, **en la máquina con la RTX 3070 Ti** (Windows + WSL2).
> Coste estimado: 3 h. No requiere reentrenar.

---

## Contexto

En este TFM se comparan cinco codificadores visuales (V0–V4) como `obs_encoder` de una
Diffusion Policy sobre Push-T. El estado del experimento, el entorno de WSL, los
checkpoints y las reglas de ejecución están en `CLAUDE.md`; léelo antes de tocar nada.

El §3.6 de la memoria (`memoria/secciones/02-metodologia.tex`, ~línea 361) promete
caracterizar el coste computacional con **cuatro** indicadores:

1. parámetros totales y entrenables — reportado en `tab:variantes`;
2. tiempo por paso de optimización — reportado en `tab:coste`;
3. **latencia de inferencia para un lote de tamaño fijo — NO reportado**;
4. **pico de memoria gráfica reservada durante el entrenamiento — NO reportado**.

El tribunal tiene una pregunta preparada sobre esto (P18 del informe, dificultad 3, «sin
escapatoria»). Además, la VRAM es el argumento con el que §3.3 justifica descartar el
ajuste fino de los ViT y §3.4 justifica reducir el lote de V3/V4: hoy está argumentada, no
medida.

**Objetivo:** medir los indicadores 3 y 4 sobre los checkpoints ya guardados e integrarlos
en la memoria, o —si la medición resulta imposible— reducir explícitamente la promesa del
§3.6 a los dos indicadores efectivamente medidos. No hay tercera salida.

---

## Antes de medir: verifica estos supuestos, no los des por ciertos

Todo lo que sigue asume hechos que debes confirmar leyendo las fuentes, no este prompt:

- `num_inference_steps` del planificador de difusión (creo que **100**; verifica en
  `logs_entrenamiento/raw/v3_hydra_config.yaml:44` y en el `cfg` de cada checkpoint).
- `n_obs_steps` (2) y `n_action_steps` (8): cada llamada a la política consume 2
  observaciones y emite 8 acciones. La latencia por acción es la cifra que importa para el
  control, y no coincide con la latencia por llamada.
- **Tamaño de lote de entrenamiento por variante.** V3 y V4 usan `batch_size: 32`; V0–V2,
  comprobar. Se lee del `cfg` embebido en cada checkpoint o de
  `data/outputs/encoder_exp/<variante>_seed42/.hydra/config.yaml` en WSL. El pico de VRAM
  **no es comparable entre variantes con lotes distintos**: la tabla debe declarar el lote.
- Resolución de entrada por variante (96 px en V0–V2; 224 px en V3/V4) y si hay recorte.
- Qué checkpoint es el «seleccionado» de cada variante (V0: época 350; V1: 150; V2: 150;
  V3: 100; V4: 100). La latencia depende de la arquitectura, no de los pesos, pero usa el
  checkpoint seleccionado de todas formas para que la cifra sea trazable.

---

## Parte A — Latencia de inferencia

### Qué se mide

Una **llamada completa a la política**: `policy.predict_action(obs_dict)` con un lote fijo,
que incluye el paso por el codificador visual y los `num_inference_steps` pasos de
eliminación de ruido del UNet.

Mide y reporta **tres magnitudes por variante**:

| Magnitud | Por qué |
|---|---|
| Latencia total de la llamada | Es el indicador prometido en §3.6 |
| Latencia solo del codificador (`policy.obs_encoder`) | Es lo único que cambia entre variantes |
| Latencia por acción emitida (total / `n_action_steps`) | Es la que determina la frecuencia de control alcanzable |

**Este desglose no es opcional.** El UNet es idéntico en las cinco variantes y se ejecuta
~100 veces por llamada, así que domina el tiempo total y diluye la diferencia entre
codificadores. Si solo reportas el total, la tabla dirá que las cinco variantes cuestan
casi lo mismo en inferencia y ocultará el hecho relevante. Reporta las dos y coméntalo.

### Protocolo

Sobre GPU, precisión `float32`, política en `eval()` y dentro de `torch.no_grad()`:

1. **Lote fijo**: `B = 1` como cifra principal —es lo que ejecuta un robot— y `B = 8` como
   cifra secundaria de rendimiento agregado. Declara ambos en la tabla. Entrada sintética
   con la forma correcta de `shape_meta` (no hace falta el simulador), pero **verifica que
   los valores estén en el mismo rango que las observaciones reales** (normalización): un
   tensor de ceros puede tomar rutas distintas en la GPU.
2. **Calentamiento**: ≥ 20 llamadas descartadas antes de cronometrar. `cudnn.benchmark`
   distorsiona las primeras iteraciones; declara qué valor usaste.
3. **Medición**: ≥ 50 repeticiones cronometradas con `time.perf_counter()`, con
   `torch.cuda.synchronize()` **antes y después de cada repetición**. Sin sincronizar, mides
   el encolado, no la ejecución.
4. **Estadístico**: reporta **mediana e intervalo intercuartílico** (o p5–p95), no solo la
   media. La distribución tiene cola por la planificación del sistema y la media miente.
5. **Deriva térmica**: es una GPU de portátil. Alterna las repeticiones entre variantes en
   ronda (round-robin) en lugar de medir las 50 de V0, luego las 50 de V1: así la deriva
   térmica afecta por igual a todas. Portátil conectado a la corriente, perfil de energía
   fijo, sin otros procesos CUDA (la regla de instancia única de `run_encoder_exp.sh`).
6. **Contexto**: registra GPU, versión de torch/CUDA, driver y si se midió en WSL o en el
   entorno de inferencia de Windows.

### Entorno

La latencia puede medirse en el entorno de inferencia de Windows (`.venv_diffuser_infer`,
torch 2.6.0+cu124) reutilizando `diffuser/v{0..4}_inference_utils.py::load_policy_bundle`,
que ya resuelve la carga de los cinco checkpoints (fuerza `rgb_model.pretrained = False`).
**Pero entonces la cifra no describe el entorno en el que se entrenó** (torch 1.12.1+cu116
en WSL). Elige un entorno, mide las cinco variantes en él, y **declara cuál** en la nota de
la tabla. No mezcles entornos entre variantes bajo ningún concepto.

---

## Parte B — Pico de memoria gráfica

### La trampa que hay que evitar

El §3.6 promete el pico reservado **durante el entrenamiento**. Eso no está dentro del
checkpoint: hay que **reproducir unos pocos pasos de optimización** para observarlo. El
informe de evaluación dice que es medible «en minutos sobre los checkpoints ya guardados»
y tiene razón en el coste, pero no en que baste con cargar el fichero. Si mides solo la
memoria de inferencia y la presentas como el indicador prometido, el error es peor que la
omisión actual.

Mide **dos columnas distintas y etiquétalas sin ambigüedad**:

- **Pico en entrenamiento** — el indicador prometido. Requiere el paso completo.
- **Pico en inferencia** — barato, y es el que importa para el despliegue. Añádelo.

### Protocolo (entrenamiento)

**Debe ejecutarse en WSL, en el entorno `robodiff` (torch 1.12.1+cu116)**, sobre la misma
GPU de 8 GB. El asignador de memoria y los buffers de trabajo de cuDNN cambiaron entre
torch 1.12 y 2.6: una medida tomada en el entorno de Windows no describe los
entrenamientos reportados en `tab:coste` y no sirve para justificar las decisiones de §3.3
y §3.4.

```
source ~/mambaforge/etc/profile.d/conda.sh && conda activate robodiff
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd ~/tfm/diffusion_policy
```

Para cada variante, una a una, sin nada más en la GPU:

1. Reconstruye el `workspace` desde el `cfg` del checkpoint (como `load_policy_bundle`,
   **pero sin excluir el optimizador**: el estado de AdamW y el modelo EMA son parte del
   pico y son justamente lo que llena los 8 GB).
2. Toma un lote real del dataloader con el **`batch_size` propio de esa variante**.
3. `torch.cuda.empty_cache()` y `torch.cuda.reset_peak_memory_stats()`.
4. Ejecuta **≥ 10 pasos completos**: forward, `loss.backward()`, `optimizer.step()`,
   `lr_scheduler.step()`, actualización de EMA, `optimizer.zero_grad()`.
5. Lee `torch.cuda.max_memory_reserved()` (el indicador prometido) **y**
   `torch.cuda.max_memory_allocated()`. Reporta el reservado como cifra principal y el
   asignado como referencia; son cosas distintas y confundirlas es un error clásico.
6. Libera todo y reinicia el proceso antes de la siguiente variante. Medir dos variantes en
   el mismo proceso contamina el pico.

**Si V2 no cabe o roza el límite, eso es un hallazgo, no un fallo**: es exactamente el
argumento que §3.3 y §3.4 hacen sin datos. Documéntalo con la cifra.

### Comprobación de coherencia (obligatoria)

Con los números en la mano, verifica si sostienen lo que la memoria ya afirma:

- §3.3 descarta el ajuste fino de los ViT por VRAM. ¿La cifra medida de V2 (ResNet-18 con
  ajuste fino) extrapolada a un ViT lo respalda? Si el argumento no se sostiene, **dilo y
  reformula §3.3**; no ajustes la interpretación de la medida para que encaje.
- §3.4 reduce el lote de V3/V4 por VRAM. ¿El pico medido con lote 32 estaba realmente cerca
  del límite de 8 GB? Si sobraba memoria, la decisión necesita otra justificación (o
  reconocerse como cautela).

---

## Entregables

Sigue las convenciones ya establecidas en el repo — mira `memoria/scripts/analisis_dispersion.py`
y `memoria/scripts/coste_parada_uniforme.py` como modelo: docstring en inglés explicando
qué se mide y por qué, `ROOT = Path(__file__).resolve().parents[2]`, salida a CSV en
`memoria/datos/`, aserciones que fallen ruidosamente si un supuesto se rompe.

1. `memoria/scripts/latencia_inferencia.py` → `memoria/datos/latencia_inferencia.csv`
2. `memoria/scripts/memoria_gpu.py` → `memoria/datos/memoria_gpu.csv`
   (si el script de VRAM debe vivir en WSL, versiónalo en `diffuser/scripts/` y documenta
   cómo se invoca, igual que `exportar_logs_wsl.sh`)
3. Los CSV deben incluir, por variante: checkpoint usado, lote, resolución,
   `num_inference_steps`, mediana, IQR, nº de repeticiones, entorno y versión de torch.
   Un CSV sin las condiciones de medida no es reproducible.
4. **Tabla nueva en LaTeX.** Recomiendo `tab:coste-inferencia` como tabla independiente en
   lugar de añadir columnas a `tab:coste`: esa tabla tiene una fila «Total» que no admite
   suma de latencias ni de picos de memoria, y mezcla unidades de tiempo con unidades de
   memoria. Colócala en §3.6 o en §4.4 (`sec:coste-resultados`) y referénciala con `\cref`.
5. **Prosa en §4.4** interpretando las cifras: la dilución del codificador en el total por
   los ~100 pasos de difusión, la frecuencia de control alcanzable, y qué variante es
   viable en 8 GB. Dos o tres párrafos, no una tabla huérfana.
6. **Revisa el §3.6**: si tras medir sigue habiendo algún matiz no cubierto (p. ej. mides
   latencia en un entorno distinto al de entrenamiento), ajústalo ahí para que promesa y
   entrega coincidan literalmente.
7. Anota la decisión en `memoria/MEMORY.md` con fecha, siguiendo el formato del registro
   de decisiones existente.
8. Recompila (`memoria/compilar.sh`) y confirma que no hay referencias rotas ni
   desbordamientos de caja en la tabla nueva.

### Estilo de la memoria

Español impersonal, sin primera persona, tono sobrio; unidades con `siunitx`
(`\SI{}{\milli\second}`, `\SI{}{\giga\byte}`), números con `\num{}`, coma decimal; citas
IEEE con bibtex clásico. Aplica `memoria/normas_redaccion.md` y la skill `redactor-tesis`.
Tres decimales como máximo, coherente con la corrección P1.4 ya aplicada.

---

## Reglas duras

- **Ninguna cifra sin ejecución.** Nada de estimar, interpolar ni «aproximadamente». Si una
  medida no se pudo tomar, se dice cuál y por qué.
- **Ninguna latencia sin declarar `num_inference_steps`, lote y resolución.** Sin esos tres
  datos la cifra no significa nada y es indefendible ante el tribunal.
- **No mezcles entornos** entre variantes, ni `reserved` con `allocated`, ni memoria de
  entrenamiento con memoria de inferencia.
- **Un proceso a la vez en la GPU.** Vale la regla de `CLAUDE.md`: nada de dos mediciones
  en paralelo, y nada mientras haya un entrenamiento vivo.
- No reentrenes nada. No toques los checkpoints. No modifiques configuraciones en WSL sin
  copiarlas al repo (no están versionadas).
- Si los checkpoints de V3/V4 fallan al cargar por el desajuste de `timm` (0.9.16 en
  entrenamiento frente a 1.0.7 en inferencia), repórtalo y para: no improvises una
  conversión de pesos.

### Plan B, si la GPU no está disponible

No inventes las cifras ni las dejes pendientes. Aplica la salida honesta que contempla el
propio informe: **reducir la promesa del §3.6 a los dos indicadores efectivamente
medidos**, y declarar latencia y pico de VRAM como medición pendiente en §5.2
(limitaciones y trabajo futuro), explicando que son medibles sobre los checkpoints
conservados. Deja el documento coherente consigo mismo; eso ya cierra M4, aunque con menos
valor que medirlo.

---

## Criterio de aceptación

Se considera M4 resuelto cuando el capítulo 4 contenga los **cuatro** indicadores
anunciados en §3.6 con sus condiciones de medida declaradas, o cuando el §3.6 prometa
exactamente los dos que se reportan. Y cuando la pregunta P18 del tribunal —«su §3.6
promete cuatro indicadores de coste, ¿dónde están la latencia de inferencia y el pico de
memoria?»— tenga una respuesta de una frase señalando a una tabla.
