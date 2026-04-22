# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

**Autores:** Cheng Chi\*, Zhenjia Xu\*, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song  
**Afiliaciones:** Columbia University, Toyota Research Institute, MIT, Stanford University  
**Publicación:** *The International Journal of Robotics Research*, Vol. 44(10-11), pp. 1684–1704, 2025  
**DOI:** 10.1177/02783649241273668  
**Versión de conferencia previa:** RSS 2023 (Chi et al., 2023)

---

## 1. ¿Qué propone?

El artículo introduce **Diffusion Policy**, un nuevo paradigma para el aprendizaje de políticas visuomotoras en robots que formula la política como un **proceso de difusión condicional denoising** sobre el espacio de acciones. En lugar de predecir directamente una acción, la política aprende el gradiente de la función de puntuación (*score function*) de la distribución de acciones y lo optimiza iterativamente durante la inferencia mediante pasos de dinámica de Langevin estocástica (*Stochastic Langevin Dynamics*).

La formulación base es la de los **Modelos de Difusión Probabilísticos Denoising (DDPM)** (Ho et al., 2020), adaptada al dominio de control robótico. El proceso de denoising sigue:

$$\mathbf{x}^{k-1} = \alpha\left(\mathbf{x}^k - \gamma \varepsilon_\theta(\mathbf{x}^k, k) + \mathcal{N}(0, \sigma^2 I)\right)$$

donde $\varepsilon_\theta$ es la red de predicción de ruido que aproxima el campo de gradiente $\nabla E(\mathbf{x})$.

---

## 2. Motivación y problema que resuelve

El aprendizaje de políticas por clonación de comportamiento (*behavior cloning*) presenta desafíos particulares frente al aprendizaje supervisado estándar:

- **Distribuciones de acción multimodales**: un mismo estado admite múltiples acciones válidas (p. ej., rodear un obstáculo por la izquierda o por la derecha).
- **Espacios de acción de alta dimensión**: la predicción de secuencias de acciones crece exponencialmente en complejidad.
- **Inestabilidad de entrenamiento**: los modelos implícitos basados en energía (EBM/IBC) requieren muestreo negativo para estimar la constante de normalización, lo que provoca inestabilidad.

Las políticas previas —explícitas (regresión directa, MDN, BET) e implícitas (IBC)— no resuelven simultáneamente los tres problemas.

---

## 3. Metodología

### 3.1 Formulación de la política

La Diffusion Policy modela la distribución condicional $p(\mathbf{A}_t | \mathbf{O}_t)$ donde:
- $\mathbf{O}_t$: ventana de observaciones visuales de los últimos $T_o$ pasos.
- $\mathbf{A}_t$: secuencia de $T_p$ acciones predichas, de las cuales sólo $T_a$ se ejecutan.

La ecuación de denoising condicionado es:

$$\mathbf{A}_t^{k-1} = \alpha\left(\mathbf{A}_t^k - \gamma \varepsilon_\theta(\mathbf{O}_t, \mathbf{A}_t^k, k) + \mathcal{N}(0, \sigma^2 I)\right)$$

La pérdida de entrenamiento es MSE sobre el ruido predicho:

$$\mathcal{L} = \text{MSE}(\varepsilon^k,\ \varepsilon_\theta(\mathbf{O}_t,\ \mathbf{A}_t^0 + \varepsilon^k,\ k))$$

### 3.2 Contribuciones técnicas principales

| Contribución | Descripción |
|---|---|
| **Control de horizonte retrocedente** (*receding horizon*) | La política re-planifica continuamente en bucle cerrado, ejecutando $T_a < T_p$ pasos antes de volver a inferir. Equilibra consistencia temporal y capacidad de reacción. |
| **Condicionamiento visual** | Las observaciones $\mathbf{O}_t$ se tratan como condicionamiento (no como parte de la distribución generada), lo que reduce el coste computacional y permite inferencia en tiempo real. |
| **Time-Series Diffusion Transformer (DiffusionPolicy-T)** | Arquitectura Transformer para la red $\varepsilon_\theta$ que minimiza el sobre-suavizado (*over-smoothing*) propio de las CNNs en señales de alta frecuencia. |

### 3.3 Arquitecturas de red para $\varepsilon_\theta$

**DiffusionPolicy-C (CNN-based):**
- CNN temporal 1D con condicionamiento FiLM (*Feature-wise Linear Modulation*) de las observaciones en cada capa.
- Robusto y fácil de ajustar; recomendado como punto de partida.
- Limitación: sesgo de baja frecuencia en tareas con cambios de acción rápidos.

**DiffusionPolicy-T (Transformer-based):**
- Basado en minGPT; las acciones ruidosas se pasan como tokens al decoder con atención causal.
- Superior en tareas de alta complejidad y control de velocidad.
- Más sensible a hiperparámetros.

### 3.4 Codificador visual

- ResNet-18 entrenado *end-to-end* con la política.
- Sustitución de *global average pooling* por *spatial softmax pooling* (preserva información espacial).
- Sustitución de BatchNorm por GroupNorm (estabilidad con EMA).
- El mejor resultado se obtiene haciendo *fine-tuning* de un ViT-B/16 preentrenado con CLIP (98% en tarea Square con solo 50 épocas).

### 3.5 Aceleración de inferencia

- Uso de **DDIM** (*Denoising Diffusion Implicit Models*, Song et al., 2021) para desacoplar iteraciones de entrenamiento e inferencia.
- En experimentos reales: 100 iteraciones de entrenamiento → 10 (simulación) / 16 (real) de inferencia.
- Latencia de inferencia: **0.1 s** en GPU NVIDIA 3080 → control en bucle cerrado a 10 Hz.

---

## 4. Ventajas fundamentales del enfoque difusivo

### 4.1 Multimodalidad de acciones

La difusión expresa distribuciones arbitrarias normalizables, incluyendo multimodales, gracias al muestreo estocástico de Langevin y a la inicialización aleatoria $\mathbf{A}_t^K \sim \mathcal{N}(0, I)$.

### 4.2 Sinergia con control posicional

Contrariamente a la literatura dominante (que usa control de velocidad), Diffusion Policy obtiene mejor rendimiento con **control posicional**, ya que la multimodalidad es más pronunciada en ese espacio y se acumulan menos errores de compounding.

### 4.3 Predicción de secuencias de acciones

La predicción de horizontes largos resuelve:
- **Consistencia temporal**: se compromete con un modo en toda la trayectoria.
- **Robustez a acciones inactivas** (*idle actions*): frecuentes en teleoperación; las políticas de un paso tienden a sobreajustarse a ellas.

Horizonte de acción óptimo encontrado: **$T_a = 8$ pasos**.

### 4.4 Estabilidad de entrenamiento

A diferencia de IBC (EBM con pérdida InfoNCE y muestreo negativo), Diffusion Policy modela directamente la función de puntuación:

$$\nabla_\mathbf{a} \log p(\mathbf{a}|\mathbf{o}) \approx -\varepsilon_\theta(\mathbf{a}, \mathbf{o})$$

que es independiente de la constante de normalización $Z(\mathbf{o}, \theta)$, eliminando la inestabilidad.

---

## 5. Evaluación experimental

### 5.1 Benchmarks evaluados (15 tareas en total)

| Benchmark | Tareas | Entorno |
|---|---|---|
| **RoboMimic** (Mandlekar et al., 2021) | Lift, Can, Square, Transport, ToolHang | Simulación + real |
| **Push-T** (Florence et al., 2021) | Push-T | Simulación + real |
| **Multimodal Block Pushing** (Shafiullah et al., 2022) | BlockPush | Simulación |
| **Franka Kitchen** (Gupta et al., 2019) | Kitchen (7 objetos, 4 sub-tareas) | Simulación |

### 5.2 Baselines comparados

- **LSTM-GMM** (BC-RNN, política explícita con mezcla de Gaussianas)
- **IBC** (política implícita basada en energía)
- **BET** (Behavior Transformers, política explícita con categorización)

### 5.3 Resultados principales

**Mejora promedio: +46.9%** sobre el mejor baseline en todas las tareas.

#### Simulación — política de estado (Tabla 1, selección):

| Tarea | LSTM-GMM | IBC | BET | DiffPolicy-C | DiffPolicy-T |
|---|---|---|---|---|---|
| Lift | 1.00 | 0.79 | 1.00 | **1.00** | **1.00** |
| Square | 0.95 | 0.00 | 0.76 | **1.00** | **1.00** |
| ToolHang | 0.67 | 0.00 | 0.58 | 0.50 | **1.00** |
| Kitchen p4 | 0.34 | 0.24 | 0.44 | **0.99** | 0.96 |

#### Experimentos reales:

| Tarea | Métrica | Human | LSTM-GMM | Diffusion Policy |
|---|---|---|---|---|
| Push-T | Success % | 100% | 20% | **95%** |
| Mug flip | Success % | 100% | 0% | **90%** |
| Sauce pouring | IoU | 0.79 | 0.06 | **0.74** |
| Sauce spreading | Coverage | 0.79 | 0.27 | **0.77** |

### 5.4 Tareas bimanual (extensión del artículo de revista)

| Tarea | Demostraciones | Success % |
|---|---|---|
| Egg beater | 210 | 55% |
| Mat unrolling | 162 | 75% |
| Shirt folding | 284 | 75% |

Estas tareas se teleoperaron con VR (Meta Quest Pro) y dispositivos hápticos (Haption Virtuose 6D). La Diffusion Policy funcionó **sin ajuste de hiperparámetros** adicional para las versiones bimanual.

### 5.5 Hallazgos del estudio de ablación

- **Horizonte de acción $T_a$**: valor óptimo = 8 pasos; valores mayores reducen la responsividad.
- **Robustez a latencia**: mantiene rendimiento máximo con hasta 4 pasos de latencia simulada.
- **Encoder visual**: el fine-tuning de ViT-B/16 (CLIP) supera a ResNet entrenado desde cero; los encoders congelados dan resultados pobres.
- **Horizonte de observación**: $T_o = 2$ es suficiente para la mayoría de tareas; valores mayores no mejoran en estado y perjudican en visión (CNN).

---

## 6. Limitaciones

1. **Herencia de limitaciones del behavior cloning**: rendimiento subóptimo con datos de demostración escasos o de baja calidad.
2. **Coste computacional**: mayor latencia de inferencia que LSTM-GMM; requiere optimizaciones (DDIM, consistency models).
3. **Dependencia de demostraciones de alta calidad**: en bimanual (egg beater), se necesitó retroalimentación háptica para obtener demostraciones viables.
4. **Transferencia limitada fuera de distribución**: en mat unrolling, los fallos se deben a condiciones iniciales fuera del dominio visto en entrenamiento.

---

## 7. Relevancia para el TFM (VLA/VLM)

Diffusion Policy es un **precursor directo de los modelos VLA generativos**: demuestra que los modelos difusivos son una representación de política expresiva y estable para el control visuomotor. Su arquitectura (condicionamiento visual + predicción de secuencias de acciones + DDPM) es la base sobre la que se construyen modelos VLA posteriores como:
- **Octo** y **OpenVLA** (que usan cabezas de acción difusivas).
- **π0** (Black et al., 2024) que integra difusión con modelos de lenguaje visual.
- **GROOT**, **RoboFlamingo** y derivados que extienden el condicionamiento a instrucciones lingüísticas.

El paper establece además que el **control posicional** supera al de velocidad en políticas difusivas, observación que los modelos VLA han heredado.

---

## Fuentes consultadas

- Chi et al. (2024). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. IJRR 44(10-11). DOI: 10.1177/02783649241273668. [PDF adjunto]
- Archivo `diffuser/start.md` del workspace (versión markdown del paper).
