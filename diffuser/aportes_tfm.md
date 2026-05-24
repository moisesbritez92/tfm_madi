# Posibles aportes al TFM sobre Diffusion Policy

Líneas de contribución viables sobre la baseline reproducida (Push-T) con hardware RTX 3070 (8 GB VRAM). Todas pensadas para ser ejecutables en tiempos razonables y para conectar con la línea VLA/VLM del TFM.

---

## 1. Estudio de eficiencia de inferencia: DDPM vs DDIM vs Consistency Models

**Hipótesis**: el número de pasos de denoising en inferencia es el principal cuello de botella para desplegar Diffusion Policy en bucle cerrado a alta frecuencia.

**Experimento**:
- Entrenar una única política Push-T (image, CNN) con la configuración por defecto.
- Variar el *scheduler* y nº de pasos de inferencia:
  - DDPM con 100, 50, 25, 10 pasos.
  - DDIM con 16, 10, 5, 2 pasos.
  - (Opcional) Consistency Distillation con 1–4 pasos.
- Reportar: éxito (`test/mean_score`), latencia por inferencia (ms en RTX 3070), throughput de control (Hz).

**Aporte**: el paper lo menciona pero no tabula la curva completa éxito-latencia. Genera una figura defendible y útil para argumentar viabilidad de Diffusion Policy en hardware modesto.

**Coste estimado**: 1 entrenamiento (~horas), N evaluaciones (~minutos cada una).

---

## 2. Sensibilidad al horizonte de acción $T_a$

**Hipótesis**: el valor óptimo $T_a = 8$ reportado por los autores es dependiente de la tarea; en Push-T una ventana distinta puede mejorar consistencia o reactividad.

**Experimento**:
- Barrer $T_a \in \{1, 2, 4, 8, 16\}$ manteniendo $T_p = 16$ constante.
- 3 semillas por configuración para significancia estadística.
- Métricas: éxito medio, varianza entre semillas, número de re-planificaciones por episodio.

**Aporte**: reproducir y matizar la ablation de la Sección 5.4 del paper con datos propios. Resultado verificable y citable.

**Coste estimado**: 5 configs × 3 seeds = 15 entrenamientos (paralelizables vía `ray_train_multirun.py` si dispones de más GPU, secuenciales en local).

---

## 3. Encoder visual: ResNet18 entrenado vs backbones pre-entrenados (VLM)

**Hipótesis**: backbones visuales pre-entrenados con supervisión visión-lenguaje (CLIP, DINOv2) reducen el número de demostraciones necesarias para alcanzar un nivel dado de éxito.

**Experimento**:
- Sustituir el encoder ResNet18 *from scratch* por:
  - ResNet18 ImageNet pre-entrenado (congelado / fine-tuned).
  - CLIP ViT-B/16 (congelado / fine-tuned).
  - DINOv2 ViT-S/14 (congelado / fine-tuned).
- Submuestrear el dataset Push-T a 25 %, 50 %, 75 %, 100 % de demostraciones.
- Reportar curva éxito vs fracción de datos para cada encoder.

**Aporte**: conecta Diffusion Policy con la pregunta central VLA/VLM del TFM (¿qué aportan los backbones visión-lenguaje a la eficiencia de muestras en políticas robóticas?). Hueco real en la literatura para tareas de manipulación con presupuesto bajo.

**Coste estimado**: 4 encoders × 4 fracciones = 16 entrenamientos. VRAM crítica con ViT-B/16 + fine-tuning en 8 GB; probablemente requerirá batch reducido o gradient checkpointing.

---

## 4. Robustez a perturbaciones de observación en tiempo de evaluación

**Hipótesis**: Diffusion Policy es robusta a multimodalidad de acciones pero su comportamiento ante observaciones degradadas (ruido sensorial, oclusiones parciales, cambios de iluminación) no ha sido sistemáticamente caracterizado.

**Experimento**:
- Entrenar política Push-T baseline (sin perturbaciones).
- En evaluación, aplicar a las imágenes de entrada:
  - Ruido gaussiano con $\sigma \in \{0.01, 0.05, 0.1, 0.2\}$.
  - Oclusión rectangular aleatoria con cobertura del 10 %, 20 %, 30 %.
  - Cambios de brillo/contraste $\pm 30 \%$.
  - Desenfoque gaussiano radio $\{1, 3, 5\}$ px.
- Comparar caída de éxito con baseline LSTM-GMM bajo las mismas perturbaciones.

**Aporte**: hueco real y poco explorado. Resultado interpretable y con implicaciones directas para despliegue en robots reales con cámaras imperfectas.

**Coste estimado**: 1 entrenamiento + N evaluaciones (las evaluaciones son baratas, ~ minutos cada barrido).

---

## Comparativa rápida

| Aporte | Esfuerzo cómputo | Originalidad | Conexión con VLA/VLM | Recomendado para empezar |
|---|---|---|---|---|
| 1. Eficiencia inferencia | Bajo | Media | Baja | ⭐ Sí (validación rápida del setup) |
| 2. Horizonte $T_a$ | Medio | Baja-Media | Baja | No (ablation, poco novedoso) |
| 3. Encoder pre-entrenado | Alto | Alta | **Muy alta** | ⭐⭐ Sí (núcleo del TFM) |
| 4. Robustez perturbaciones | Bajo-Medio | Alta | Media | ⭐ Sí (complementario y barato) |

**Sugerencia**: empezar por **(1)** para validar pipeline en una semana, luego abordar **(3)** como contribución principal del TFM y cerrar con **(4)** como sección de robustez. **(2)** queda como apéndice opcional.

---

## Estado actual (22/05/2026)

- **Decisión:** se elige la línea **(3) Encoder visual** como aporte principal del TFM.
- **Implementación:** módulo `diffusion_policy/model/vision/pretrained_encoders.py` + 5 configs `pusht_v0…v4_*.yaml` (ver detalle en [experimento_encoder_pusht.md](experimento_encoder_pusht.md)).
- **Validación:** dry-run completado para las 5 variantes; todas caben en 8 GB VRAM.
- **Próximo paso:** lanzar entrenamiento de V0 (baseline ResNet-18 *from scratch*) a 500 épocas como referencia.
- **(4) Robustez** se mantiene como ablation final reutilizando los checkpoints de (3).

