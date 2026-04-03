

  



Read [](file:///c%3A/Users/moise/Documents/0001_MADI/TFM/VLT/world_model.md)

Fetched https://www.q2bstudio.com/nuestro-blog/743502/nvidia-presenta-dreamdojo-modelo-robots-entrenado-datos-video-humanos-revolucionar-industria-robotica-descubre-innovacion-cambiando-juego-inteligencia-artificial?scriptscookies=1

Compacted conversationLa página es un resumen superficial/promocional. Voy a buscar la fuente original de NVIDIA (paper de investigación) para obtener detalles técnicos reales. 


---

## DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos

**Paper**: [arXiv:2602.06949](https://arxiv.org/abs/2602.06949) — Febrero 2026  
**Autores**: Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, et al. (NVIDIA GEAR Lab + UC Berkeley, CMU, KAIST, NTU)  
**Código**: [github.com/NVIDIA/DreamDojo](https://github.com/NVIDIA/DreamDojo)  
**Project page**: [dreamdojo-world.github.io](https://dreamdojo-world.github.io/)

---

### 1. Problema que resuelve

Los *world models* robóticos actuales están limitados por:
- **Escasez de datos robóticos**: Recopilar trayectorias de teleoperación es costoso y cubre distribuciones estrechas.
- **Ausencia de etiquetas de acción**: Los videos de internet carecen de anotaciones de acciones fine-grained.
- **Generalización pobre**: Los modelos existentes solo funcionan en escenarios in-distribution, fallando ante objetos o entornos nuevos.

### 2. Dataset: DreamDojo-HV

El **mayor dataset de video egocéntrico humano** para preentrenamiento de world models:

| Métrica | DreamDojo-HV | Next Largest |
|---|---|---|
| **Duración** | **44,711 horas** | ~2,900 h (AgiBot-World) |
| **Escenas únicas** | **9,869** | ~564 (DROID) |
| **Tareas únicas** | **6,015** | ~87 (AgiBot-World) |
| **Objetos únicos** | **43,237** | — |

- **15×** más duración, **96×** más habilidades, **2,000×** más escenas que el dataset previo más grande.
- Fuentes: (1) **In-lab** (guantes Manus + Vive Tracker), (2) **EgoDex** (829h, Apple Vision Pro), (3) **DreamDojo-HV** (crowdsourcing masivo).

### 3. Arquitectura

Construido sobre **Cosmos-Predict2.5** (modelo de difusión de video latente con tokenizador WAN2.2 y bloques DiT). Dos mejoras clave:

**a) Acciones relativas**: En vez de poses absolutas, se transforman a acciones relativas re-baselineadas al inicio de cada frame latente → reduce complejidad de modelado y mejora generalización.

**b) Inyección de acciones por chunks**: 4 acciones consecutivas se concatenan como un chunk y se envían al frame latente correspondiente (ratio de compresión temporal del tokenizador = 4). Esto respeta la **causalidad** — el modelo no ve acciones futuras que aún no han ocurrido.

### 4. Modelo de Acciones Latentes (clave de la innovación)

Un **VAE de 700M parámetros** (Transformer espacio-temporal) que resuelve la escasez de etiquetas de acción:

- **Encoder**: Toma 2 frames consecutivos $f_{t:t+1}$ → produce vector latente $\hat{a}_t \in \mathbb{R}^{32}$
- **Decoder**: Recibe $\hat{a}_t + f_t$ → reconstruye $f_{t+1}$
- **Information bottleneck**: La baja dimensionalidad (32) fuerza al modelo a disentangle solo la información de acción del contexto.

$$\mathcal{L}_{\theta,\phi}^{pred}(f_{t+1}) = \mathbb{E}_{q_\phi(\hat{a}|f_{t:t+1})} \log p_\theta(f_{t+1}|\hat{a}, f_t) - \beta D_{KL}(q_\phi(\hat{a}|f_{t:t+1}) \| p(\hat{a}))$$

Con $\beta = 10^{-6}$. Esto permite que **videos humanos sin etiquetas de acción** contribuyan a la controlabilidad del world model.

### 5. Pipeline de Entrenamiento (3 fases)

| Fase | Datos | GPUs | Steps | Batch |
|---|---|---|---|---|
| **Pretraining** | In-lab + EgoDex + DreamDojo-HV (ratio 1:2:10) | 256 × H100 | 140k | 1024 |
| **Post-training** | Robot target (GR-1, G1, AgiBot) | 128 × H100 | 50k | 512 |
| **Distillation** | Mismo dataset robot | 64 × H100 | 13k | 256/64 |

**Pérdida de entrenamiento** — Flow matching + consistencia temporal:

$$\mathcal{L}_{final}(\theta) = \mathcal{L}_{flow}(\theta) + \lambda \mathcal{L}_{temporal}(\theta), \quad \lambda = 0.1$$

Donde la pérdida temporal nueva supervisa las **transiciones entre frames** (no solo frames individuales):

$$\mathcal{L}_{temporal}(\theta) = \mathbb{E}\left[\sum_{i=1}^{K-1} \|(z_{i+1} - z_i) - (v_{i+1} - v_i)\|^2\right]$$

### 6. Destilación (real-time)

Basada en **Self Forcing**:
- Reemplaza atención bidireccional → **atención causal** (ventana deslizante de 12 frames)
- Reduce pasos de denoising de **35 → 4**
- Dos etapas: warmup (regresión a ODE del teacher) + distillation (KL divergence, distribución matching)

| Modelo | FPS | Horizonte | Contexto |
|---|---|---|---|
| **Teacher** | 2.72 | 12 frames | 1 frame |
| **Student (destilado)** | **10.81** | 4 frames (autoregresivo) | 12 frames |

El modelo destilado genera **>1 minuto de video en tiempo real** sin degradación, y es más robusto a oclusiones gracias al contexto multi-frame.

### 7. Resultados Principales

**Preferencia humana (vs Cosmos-Predict2.5):**

| Comparación | Physics Correctness | Action Following |
|---|---|---|
| DreamDojo-2B > baseline | 62.50% | 63.45% |
| DreamDojo-14B > baseline | **73.50%** | **72.55%** |
| DreamDojo-14B > 2B | 72.50% | 65.53% |

**Evaluación de políticas**: Correlación Pearson $r = 0.995$, MMRV = 0.003 (casi perfecta alineación con el mundo real).

**Planificación basada en modelo**: +17% tasa de éxito, ~2× mejora sobre muestreo uniforme de propuestas.

### 8. Aplicaciones Downstream

1. **Policy Evaluation** — Evaluar políticas robóticas sin despliegue real (Pearson 0.995).
2. **Model-Based Planning** — Simular múltiples propuestas de acción → seleccionar la mejor con un value model (DINOv2-based).
3. **Live Teleoperation** — Teleoperar un robot virtual G1 con PICO VR controller en tiempo real (RTX 5090).

### 9. Limitaciones Reconocidas

- Acciones poco comunes (golpear, agitar rápido) → simulación deficiente.
- Tasas de éxito absolutas más altas que la realidad (no genera fallos matizados).
- **Sin soporte multi-vista** — crucial para políticas SOTA como GR00T N1.5 y π₀.
- Retención de conocimiento pre-entrenado durante fine-tuning no bien estudiada (sugieren explorar LoRA).
- Espacio para mejora de velocidad de inferencia.

### 10. Contexto en el campo

DreamDojo es el **primer foundation world model para manipulación diestra** que:
- Escala a datos humanos de internet (no solo datos robóticos)
- Usa acciones latentes continuas como proxy unificado cross-embodiment
- Demuestra generalización zero-shot a objetos y entornos no vistos
- Opera en tiempo real tras destilación

Se posiciona como la contraparte "simulador neural" de los simuladores clásicos (Isaac Sim, MuJoCo), pero aprendido directamente de píxeles.


