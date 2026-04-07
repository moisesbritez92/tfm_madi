# Revisión Bibliográfica: Vision-Language Models (VLM) y Vision-Language-Action Models (VLA)

**Fuentes principales:**
- Kawaharazuka, K., Oh, J., Yamada, J., Posner, I. & Zhu, Y. (2025). *Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications*. IEEE Access, 13, 162467–162504. arXiv:2510.07077.
- Brohan, A. et al. (2023). *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. arXiv:2307.15818. Google DeepMind.
- Kim, M. J. et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246. Stanford University.
- Black, K. et al. (2024). *π₀: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164. Physical Intelligence. RSS 2025.
- Octo Model Team et al. (2024). *Octo: An Open-Source Generalist Robot Policy*. arXiv:2405.12213. UC Berkeley.
- NVIDIA (2025). *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*. arXiv:2503.14734.

**Documento complementario:** Esta revisión extiende cronológicamente la *Revisión Bibliográfica de Vision Language Transformers* (Fields & Kennington, 2023) ya elaborada, cubriendo la evolución desde los VL transformers clásicos hacia los modelos multimodales con capacidad de acción robótica (2023–2025).

---

## 1. Contexto y Motivación

### 1.1 De la Percepción a la Acción

La intersección de Visión por Computador (CV) y Procesamiento de Lenguaje Natural (NLP) dio lugar al dominio de **Visión-Lenguaje (VL)**, cuyo progreso ha sido documentado extensamente en la revisión complementaria sobre Vision Language Transformers. Los modelos VL clásicos — desde CLIP (Radford et al., 2021) hasta Flamingo (Alayrac et al., 2022) y BLIP-2 (Li et al., 2023) — demostraron capacidades extraordinarias en tareas como Visual Question Answering, Image Captioning y retrieval multimodal. Sin embargo, estas capacidades permanecían confinadas al dominio **digital**: comprender imágenes y generar texto, sin posibilidad de actuar sobre el mundo físico.

La pregunta fundamental que motivó la siguiente generación de modelos fue: **¿Puede un modelo que entiende el mundo visual y lingüístico también aprender a actuar en él?** Esta pregunta conecta directamente con el **symbol grounding problem** (Harnad, 1990) — cómo los símbolos lingüísticos adquieren significado a través de la interacción con el mundo perceptual — y lo extiende hacia el *embodied grounding*: cómo la comprensión multimodal se traduce en acción física.

### 1.2 La Emergencia de Embodied AI

**Embodied AI** se refiere a sistemas inteligentes integrados en entidades físicas — principalmente robots — que aprenden no solo de datos estáticos, sino de su interacción con el entorno. De manera análoga a cómo los humanos aprenden de la experiencia sensoriomotora, los agentes embodied AI aprenden a través de señales visuales, instrucciones lingüísticas y retroalimentación de sus propias acciones.

La evolución histórica puede trazarse en tres etapas:

1. **Políticas específicas por tarea (pre-2020):** Cada tarea robótica requería un modelo entrenado *ad hoc* — una política de *pick-and-place* no era transferible a *door opening*. Los sistemas eran rígidos, costosos de desarrollar y confinados a entornos estructurados.

2. **Planificación con LLMs (2022–2023):** Modelos como **SayCan** (Ahn et al., 2022) demostraron que los LLMs podían funcionar como planificadores de alto nivel para robots, descomponiendo instrucciones complejas en subtareas ejecutables. Sin embargo, la ejecución motora seguía dependiendo de políticas de bajo nivel preentrenadas por separado.

3. **Modelos unificados VLA (2023–presente):** La propuesta de unificar percepción visual, comprensión lingüística y generación de acciones en un solo modelo marca la frontera actual de investigación. El término **Vision-Language-Action (VLA)** fue acuñado formalmente en el paper de **RT-2** (Brohan et al., 2023), que demostró por primera vez que un modelo VL preentrenado podía ser co-fine-tuneado para generar acciones robóticas directamente.

### 1.3 Definiciones Operacionales

Para evitar ambigüedad terminológica en esta revisión, establecemos las siguientes definiciones:

- **Vision-Language Model (VLM):** Modelo que procesa conjuntamente información visual (imágenes/video) y textual. Sus salidas son exclusivamente textuales o de representación latente. Ejemplos: CLIP, PaLI-X, PaLM-E, GPT-4V, PaliGemma.

- **Vision-Language-Action Model (VLA):** Extensión de un VLM que, además de visión y lenguaje, incorpora **tokens de estado** (observaciones del robot: posiciones articulares, valores de sensores, estado del gripper) y genera **tokens de acción** (comandos motores que controlan directamente el efector final del robot). La secuencia completa es: observación visual + instrucción lingüística + estado robótico → secuencia de acciones.

- **Generalist Robot Policy (GRP):** Política robótica generalista — un modelo unificado capaz de resolver múltiples tareas downstream o adaptarse a nuevas tareas sin fine-tuning específico por tarea. A diferencia de las políticas tradicionales, una GRP desarrolla comportamientos emergentes que generalizan a escenarios no vistos, nuevas plataformas hardware y objetos novedosos.

- **Embodied AI:** Sistemas de inteligencia artificial integrados en entidades físicas que aprenden mediante interacción con el mundo real, combinando percepción, razonamiento y acción.

### 1.4 Alcance y Estructura de esta Revisión

Esta revisión cubre el periodo 2023–2025, desde la emergencia formal del paradigma VLA hasta el estado del arte actual, con una base de datos de más de 257 modelos catalogados en el survey de Kawaharazuka et al. (2025). La estructura sigue una progresión cronológica:

- **Secciones 2–3:** Fundamentos previos necesarios y evolución de los VLMs
- **Sección 4:** Transición conceptual de VLMs a VLAs
- **Sección 5:** Taxonomía arquitectónica de los VLAs
- **Sección 6:** Análisis detallado de los modelos SOTA
- **Secciones 7–8:** Paradigmas de entrenamiento, datos y plataformas robóticas
- **Sección 9:** Benchmarks y evaluación sistemática
- **Sección 10:** Análisis crítico, limitaciones y direcciones futuras

---

## 2. Fundamentos Previos

> Esta sección resume los pilares conceptuales necesarios para comprender los VLA. Para una revisión exhaustiva de los mecanismos transformer y las arquitecturas VL clásicas, véase el documento complementario *Revisión Bibliográfica: Vision Language Transformers*.

### 2.1 Recapitulación: Mecanismo de Atención y Transformer

La arquitectura transformer (Vaswani et al., 2017) se fundamenta en el mecanismo de **atención multi-cabeza (MHA)**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

Los VLAs heredan esta maquinaria pero la extienden para operar sobre **tres modalidades simultáneas**: tokens visuales (patches de imagen), tokens lingüísticos (subwords) y tokens de acción/estado (representaciones de las posiciones y comandos del robot).

### 2.2 Recapitulación: De LLMs a VLMs

La línea evolutiva que conduce a los VLAs puede sintetizarse como:

**GPT/BERT (2018)** → Solo texto, preentrenamiento masivo en corpus lingüísticos
↓
**ViT (2020)** → Imágenes como secuencias de patches, unificación de la representación
↓
**CLIP/ALIGN (2021)** → Alineación visión-lenguaje mediante aprendizaje contrastivo
↓
**Flamingo/BLIP-2 (2022–2023)** → VLMs generativos con LLMs como backbone
↓
**PaLM-E (2023)** → Primeros pasos hacia embodied VLMs con datos robóticos
↓
**RT-2 (2023)** → Primer VLA formal: acciones como tokens de texto

Cada paso hereda las capacidades del anterior y añade una nueva dimensión de funcionalidad.

### 2.3 Políticas Robóticas: Formulación Clásica

En robótica, una **política** $\pi$ es una función que mapea estados a acciones:

$$\pi: \mathcal{S} \rightarrow \mathcal{A}$$

donde $\mathcal{S}$ es el espacio de estados (observaciones sensoriales del robot) y $\mathcal{A}$ es el espacio de acciones (comandos motores). Existen dos paradigmas principales:

- **Políticas deterministas:** $a = \pi(s)$ — mapeo directo estado → acción.
- **Políticas estocásticas:** $a \sim \pi(\cdot | s)$ — distribución de probabilidad sobre acciones dado el estado.

Un **episodio** robótico es una secuencia de interacciones desde el estado inicial $s_0$ hasta un estado terminal $s_T$:

$$(s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$$

donde $r_t$ es la recompensa en el paso $t$. Los agentes se entrenan típicamente mediante:

1. **Reinforcement Learning (RL):** Maximización de la recompensa acumulada $\sum_{t=0}^{T} \gamma^t r_t$ mediante exploración.
2. **Behavioral Cloning (BC) / Imitation Learning:** Dada una colección de trayectorias de demostración $(o_t, a_t)$ de un experto, se entrena un modelo que predice acciones correctas — aprendizaje supervisado directo sin necesidad de función de recompensa.

Los VLAs adoptan predominantemente **behavioral cloning** como paradigma de entrenamiento, aprovechando grandes colecciones de demostraciones robóticas para aprender políticas generalistas.

### 2.4 Del Control Clásico a las Políticas Neuronales

El paradigma clásico de control robótico sigue un pipeline secuencial:

$$\text{Percepción} \rightarrow \text{Planificación} \rightarrow \text{Control Motor}$$

Cada etapa es implementada por módulos independientes (detector de objetos → planificador de trayectorias → controlador PID). Este enfoque modular tiene ventajas en interpretabilidad, pero sufre de:

- **Propagación de errores:** Cada módulo acumula errores que se propagan al siguiente.
- **Ingeniería manual intensiva:** Cada módulo requiere diseño, calibración y mantenimiento específico.
- **Rigidez:** Cambios en el entorno o la tarea pueden requerir rediseñar múltiples módulos.

Los VLAs proponen un paradigma **end-to-end**:

$$\text{Imagen} + \text{Instrucción} + \text{Estado Robot} \xrightarrow{\text{VLA}} \text{Acciones Motoras}$$

Un único modelo neuronal realiza percepción, razonamiento y generación de acciones simultáneamente, eliminando la propagación de errores entre módulos y permitiendo que el modelo aprenda representaciones internas optimizadas para la tarea final.

---

## 3. Vision-Language Models (VLMs): La Generación Puente

Los VLMs representan la generación de modelos que sirve de puente entre los VL transformers clásicos y los VLAs. Su contribución fundamental es demostrar que un modelo preentrenado en datos web a escala de internet puede adquirir conocimiento semántico transferible a dominios especializados.

### 3.1 Modelos Fundacionales VLM

#### 3.1.1 CLIP y la Revolución del Aprendizaje Contrastivo

**CLIP** (Contrastive Language-Image Pre-training; Radford et al., 2021) demostró que alinear representaciones de imagen y texto mediante aprendizaje contrastivo en 400M de pares imagen-texto de internet producía representaciones de zero-shot extraordinariamente robustas. Su función de pérdida maximiza la similitud coseno de los $N$ pares correctos y minimiza los $N^2 - N$ pares incorrectos dentro de cada batch:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(v_i, t_j)/\tau)} + \log\frac{\exp(\text{sim}(t_i, v_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(t_j, v_i)/\tau)}\right]$$

donde $\text{sim}(v, t) = \frac{v \cdot t}{\|v\|\|t\|}$ es la similitud coseno y $\tau$ es un parámetro de temperatura aprendido.

CLIP estableció un paradigma clave: **las representaciones visuales alineadas con lenguaje son intrínsecamente más transferibles** que las aprendidas solo con supervisión visual. Esta propiedad es la que permite a los VLAs heredar conocimiento semántico del mundo.

#### 3.1.2 SigLIP: Simplificación Escalable

**SigLIP** (Zhai et al., 2023) reemplaza la softmax de CLIP por una función sigmoide por par, eliminando la necesidad de computar todas las combinaciones dentro del batch:

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}\left[y_{ij}\log\sigma(s_{ij}) + (1-y_{ij})\log(1-\sigma(s_{ij}))\right]$$

donde $y_{ij} = 1$ si $i=j$ (par correcto) y $s_{ij} = \text{sim}(v_i, t_j)/\tau$. Esta reformulación permite entrenamiento con batches más grandes sin las limitaciones de memoria de softmax global, y SigLIP se convierte en el **encoder visual preferido** de modelos VLA posteriores como PaliGemma (base de π₀) y OpenVLA.

#### 3.1.3 DINOv2: Representaciones Visuales Auto-Supervisadas

**DINOv2** (Oquab et al., 2024) entrena un ViT con auto-supervisión (sin pares imagen-texto), produciendo representaciones que capturan ricas relaciones espaciales y geométricas. Mientras SigLIP excela en alineación semántica con lenguaje, DINOv2 ofrece representaciones de **estructura espacial fina** — crítica para la manipulación robótica, donde la localización precisa de objetos es esencial.

**OpenVLA** (Kim et al., 2024) explota esta complementariedad usando un **dual vision encoder**: DINOv2 (~300M) + SigLIP (~400M), fusionando las fortalezas de ambos.

### 3.2 VLMs Generativos de Gran Escala

#### 3.2.1 PaLI-X y PaLM-E

**PaLI-X** (Chen et al., 2023) es un VLM de 55B parámetros que combina un ViT-22B con un encoder-decoder de texto. Su escala masiva le confiere capacidades de razonamiento visual sofisticado.

**PaLM-E** (Driess et al., 2023) representa el primer paso explícito hacia *embodied VLMs*. Construido sobre PaLM (540B parámetros), integra señales multimodales incluyendo observaciones robóticas — imágenes de cámara y estados del robot — como tokens continuos en el espacio de embedding del LLM. PaLM-E demostró que un único modelo podía:

- Realizar VQA y captioning estándar
- Planificar secuencias de acciones robóticas en lenguaje natural
- Razonar espacialmente sobre escenas del mundo real

La importancia de PaLM-E fue conceptual: demostró la viabilidad de integrar datos robóticos en modelos de lenguaje a escala, abriendo el camino directo hacia los VLAs.

#### 3.2.2 PaliGemma

**PaliGemma** (Beyer et al., 2024) es un VLM de ~3B parámetros que combina SigLIP como encoder visual con **Gemma** (Google) como modelo de lenguaje. Su tamaño relativamente compacto, combinado con excelentes capacidades de razonamiento visual, lo convierte en el backbone VLM elegido por **π₀** (Physical Intelligence) para su modelo VLA.

#### 3.2.3 Multimodal LLMs (MLLMs): GPT-4V, Gemini, Claude Vision

La generación más reciente de MLLMs — GPT-4V (OpenAI, 2023), Gemini (Google, 2024) y Claude Vision (Anthropic, 2024) — integra nativamente capacidades visuales en LLMs de escala masiva. Estos modelos son relevantes para el ecosistema VLA de dos maneras:

1. **Como planificadores de alto nivel:** Los MLLMs pueden descomponer tareas complejas en subtareas, generando instrucciones que un VLA de menor escala ejecuta.
2. **Como backbone directo:** Gemini Robotics (Google DeepMind, 2025) demuestra que un MLLM puede ser fine-tuneado directamente para control robótico end-to-end.

### 3.3 La Transición VLM → VLA: Insights Clave

Tres insights fundamentales de los VLMs motivaron el desarrollo de los VLAs:

1. **Conocimiento del mundo transferible:** Los VLMs preentrenados en datos web contienen conocimiento semántico rico sobre objetos, relaciones espaciales y física intuitiva — conocimiento directamente útil para manipulación robótica.

2. **Formato unificado de secuencias:** Si imágenes y texto pueden representarse como secuencias de tokens, las acciones también pueden serlo. Esta observación de unificación de formato es la base del diseño de RT-2.

3. **Propiedades emergentes con escala:** Los VLMs exhiben capacidades que no fueron explícitamente entrenadas. RT-2 demostró que esta propiedad se transfiere al dominio robótico — el modelo podía interpretar comandos y razonar sobre objetos nunca vistos durante el entrenamiento robótico.

---

## 4. De VLMs a VLAs: La Transición Conceptual

### 4.1 El Problema de la Brecha de Modalidad

Los LLMs operan exclusivamente en el dominio textual; los VLMs extienden esto al dominio visual. Pero los robots necesitan más: necesitan **generar acciones físicas**. Esta transición requiere resolver tres desafíos fundamentales:

1. **Representación de acciones:** ¿Cómo codificar comandos motores continuos (posiciones articulares, velocidades, fuerzas) en un formato compatible con la arquitectura transformer?

2. **Grounding físico:** Los modelos web conocen el concepto abstracto de "levantar un vaso", pero no las fuerzas, trayectorias y restricciones cinemáticas necesarias para hacerlo físicamente.

3. **Frecuencia de control:** Un LLM genera tokens a ~10 tokens/segundo. Un controlador robótico puede requerir actualizaciones a 50–200 Hz. Esta diferencia de tres órdenes de magnitud requiere soluciones arquitectónicas específicas.

### 4.2 Estrategias de Representación de Acciones

#### 4.2.1 Acciones Discretizadas (Tokenización Directa)

**RT-2** (Brohan et al., 2023) propuso la solución más elegante y directa: **expresar las acciones como tokens de texto**. El espacio de acciones continuo de 7 dimensiones (3 traslaciones + 3 rotaciones + 1 gripper) se discretiza en 256 bins por dimensión. Cada acción se convierte en una secuencia de tokens numéricos que el VLM genera de manera autoregresiva, igual que genera palabras.

**OpenVLA** (Kim et al., 2024) refina esta estrategia demostrando que solo se necesitan **255 tokens de acción** para representar el espacio completo de acciones de un robot, y que el entrenamiento puede realizarse mediante **next-token prediction** con pérdida de cross-entropy — idéntico al entrenamiento de un LLM.

**Ventajas:** Simplicidad conceptual; reutilización directa de la maquinaria de entrenamiento e inferencia de LLMs.
**Desventajas:** Pérdida de resolución por discretización; generación autoregresiva lenta para control de alta frecuencia.

#### 4.2.2 Acciones Continuas con Diffusion Heads

**Octo** (Octo Model Team et al., 2024) adopta un enfoque distinto: en lugar de discretizar acciones, usa un **diffusion head** que genera secuencias de acciones continuas. Partiendo de ruido gaussiano, el decoder de difusión produce iterativamente acciones más refinadas condicionadas en la representación del transformer.

**Ventajas:** Resolución continua; puede modelar distribuciones de acciones multimodales (múltiples trayectorias válidas para la misma tarea).
**Desventajas:** Proceso iterativo de denoising más costoso; requiere múltiples pasos de difusión por acción.

#### 4.2.3 Flow Matching

**π₀** (Black et al., 2024) introduce **conditional flow matching** como alternativa a la difusión. A diferencia de la difusión que opera en pasos discretos de denoising, flow matching aprende un campo vectorial continuo $v_\theta$ que transporta distribuciones de ruido hacia distribuciones de acciones:

$$A_\tau = \tau A_t + (1-\tau) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

donde $A_t$ es la acción objetivo y $\tau \in [0,1]$ controla la interpolación entre ruido y acción. El modelo aprende a predecir el vector de flujo:

$$v_\theta(A_\tau, \tau) \approx \epsilon - A_t$$

y durante la inferencia, se integra el campo de velocidad para transformar ruido en acciones:

$$A_{t+1} = A_t + \Delta\tau \cdot v_\theta(A_\tau, \tau)$$

**Ventajas:** Convergencia más rápida que difusión; mejor modelado de acciones suaves y continuas; natural para control de alta frecuencia (π₀ opera a 50 Hz).
**Desventajas:** Requiere diseño cuidadoso del esquema de interpolación.

#### 4.2.4 Frequency-Space Action Sequence Tokenization (FAST)

**π₀-FAST** (Pertsch et al., 2025) introduce una innovación en compresión de secuencias de acción usando la **Transformada Discreta del Coseno (DCT)**. La secuencia temporal de acciones se transforma al dominio de frecuencias:

$$F_k = \sum_{n=0}^{N-1} a_n \cos\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right], \quad k = 0,1,\ldots,N-1$$

Los **componentes de baja frecuencia** representan movimientos suaves y sostenidos, mientras los de **alta frecuencia** representan cambios abruptos. Para escenas típicas del mundo real, los componentes de baja frecuencia contienen la información más relevante, permitiendo una compresión significativa. Los coeficientes DCT resultantes se discretizan y tokenizan, permitiendo generación autoregresiva eficiente de secuencias de acción largas.

### 4.3 Tabla Comparativa de Representaciones de Acción

| Método | Modelos | Tipo de Acción | Frecuencia | Ventaja clave | Limitación clave |
|---|---|---|---|---|---|
| Discretización directa | RT-2, OpenVLA | Tokens discretos | 5 Hz | Simplicidad, compatibilidad LLM | Baja resolución temporal |
| Diffusion head | Octo | Continuas | ~10 Hz | Multimodalidad, resolución | Costo iterativo |
| Flow matching | π₀ | Continuas (chunks de 50) | 50 Hz | Suavidad, velocidad | Complejidad de diseño |
| FAST (DCT) | π₀-FAST | Tokens frecuenciales | 50 Hz | Compresión, autoregresivo | Pérdida de alta frecuencia |
| Diffusion Transformer | GR00T N1, Helix | Continuas | 120–200 Hz | Máxima frecuencia | Costo computacional |

---

## 5. Taxonomía Arquitectónica de los VLAs

El survey de Kawaharazuka et al. (2025) cataloga 257 modelos VLA. A partir de su análisis y del artículo de LearnOpenCV (Jaykumaran, 2025), podemos establecer una taxonomía de cinco tipos fundamentales, más una sexta categoría emergente.

### 5.1 Tipo 1: VLM/LLM como Planificador de Alto Nivel

**Concepto:** Un VLM o LLM interpreta la instrucción y la escena visual, generando un plan en lenguaje natural que es ejecutado por políticas de bajo nivel independientes.

**Arquitectura:**
$$\text{Instrucción + Imagen} \xrightarrow{\text{VLM/LLM}} \text{Plan textual} \xrightarrow{\text{Políticas de bajo nivel}} \text{Acciones}$$

**Modelos representativos:**
- **SayCan** (Ahn et al., 2022): Usa un LLM para generar candidatos de subtareas, y políticas preentrenadas de bajo nivel que proporcionan una función de *affordance* (probabilidad de éxito) para grounding. El LLM propone acciones lingüísticamente plausibles; las políticas filtran las físicamente ejecutables.
- **PaLM-E** (Driess et al., 2023): Integra observaciones multimodales directamente en el espacio de embedding del LLM, planificando secuencias de acciones más sofisticadas que SayCan.
- **Code as Policies** (Liang et al., 2023): El LLM genera código ejecutable (Python) que invoca APIs de control robótico, aprovechando las capacidades de codificación de los modelos de lenguaje.

**Fortalezas:** Aprovecha modelos VLM/LLM existentes sin modificación; interpretabilidad del plan generado; modularidad.
**Limitaciones:** Dependencia de políticas de bajo nivel preentrenadas; latencia alta por pipeline secuencial; capacidad de planificación limitada por los affordances disponibles.

### 5.2 Tipo 2: Modelos Generativos de Imagen/Video como Planificadores

**Concepto:** Un modelo de generación de imágenes o video predice el estado futuro deseado (goal image), y una política de bajo nivel ejecuta las acciones para alcanzar ese estado.

**Arquitectura:**
$$\text{Estado actual + Instrucción} \xrightarrow{\text{Generador}} \text{Imagen/Video objetivo} \xrightarrow{\text{Política inversa}} \text{Acciones}$$

**Modelos representativos:**
- **SuSIE** (Black et al., 2023, UC Berkeley): Usa un modelo de difusión condicionado en la imagen actual y la instrucción para generar una imagen del estado futuro deseado.
- **UniPi** (Du et al., 2024): Genera videos completos de trayectorias planeadas; un modelo inverso extrae acciones.

**Fortalezas:** Los modelos de generación de imagen tienen ricas priors sobre la estrucura del mundo visual; planificación implícita en espacio visual.
**Limitaciones:** Complejidad computacional de generación de imagen/video; dificultad de extraer acciones precisas de imágenes generadas; acumulación de errores en el modelo inverso.

### 5.3 Tipo 3: Enfoques Híbridos

**Concepto:** Combinan planificación VLM/LLM con planificación visual generativa, aprovechando las fortalezas de ambos enfoques.

**Modelos representativos:**
- **HybridVLA** (Li et al., 2025): Usa un VLM para razonamiento semántico de alto nivel y un modelo de difusión para planificación espacial detallada, fusionando ambas señales para generar acciones.

### 5.4 Tipo 4: VLM End-to-End para Control Directo

**Concepto:** Un único VLM genera acciones directamente como tokens, sin módulos intermedios. Las acciones se codifican en el mismo vocabulario que el texto.

**Arquitectura:**
$$\text{Imagen + Instrucción + Estado} \xrightarrow{\text{VLM}} \text{Tokens de acción}$$

**Modelos representativos:**
- **RT-2** (Brohan et al., 2023): El modelo fundacional de esta categoría. Co-fine-tunea PaLI-X (55B) o PaLM-E (12B) en datos web + datos robóticos, representando acciones de 7-DoF como strings de tokens numéricos.
- **OpenVLA** (Kim et al., 2024): VLA open-source de 7B parámetros basado en Llama 2 + DINOv2 + SigLIP, entrenado en 970K episodios del Open X-Embodiment Dataset.
- **QUAR-VLA** (Ren et al., 2024): Especializado en robots cuadrúpedos, discretiza el espacio de acciones en 255 bins para control de locomoción.

**Fortalezas:** Simplicidad arquitectónica; hereda directamente capacidades de razonamiento del VLM preentrenado; entrenamiento como next-token prediction.
**Limitaciones:** Generación autoregresiva lenta; inadecuado para control de alta frecuencia; pérdida de precisión por discretización.

### 5.5 Tipo 5: VLM + Diffusion/Flow Expert (Arquitectura Dual-System)

**Concepto:** La arquitectura más sofisticada y actualmente dominante. Inspirada en la **teoría de procesos duales** de Kahneman (2011), separa el procesamiento en:

- **Sistema 2 ("pensar lento"):** Un VLM que interpreta la escena visual y las instrucciones, realizando razonamiento semántico de alto nivel y planificación contextual. Opera a frecuencia baja (~1–10 Hz).

- **Sistema 1 ("pensar rápido"):** Un modelo de difusión o flow matching como **action expert**, que traduce el contexto del Sistema 2 en secuencias suaves de acciones motoras de alta frecuencia. Opera a 50–200 Hz.

**Arquitectura:**
$$\text{Imagen + Instrucción} \xrightarrow{\underbrace{\text{VLM}}_{\text{Sistema 2}}} \text{Latent context} \xrightarrow{\underbrace{\text{Diffusion/Flow}}_{\text{Sistema 1}}} \text{Acciones continuas}$$

**Modelos representativos:**
- **π₀** (Black et al., 2024): PaliGemma (~3B) como VLM + action expert de 300M parámetros con flow matching. Genera chunks de 50 acciones a 50 Hz. El VLM no observa los tokens de acción futuros para preservar su conocimiento visual preentrenado.

- **GR00T N1** (NVIDIA, 2025): Eagle2 VLM (~2B, basado en SigLIP + Llama) + Diffusion Transformer (DiT) de 16 bloques como Sistema 1. Opera a 120 Hz. Entrenado con mezcla heterogénea de trayectorias reales, videos humanos y datos sintéticos de Omniverse/Cosmos.

- **Helix** (Figure AI, 2025): VLM de 7B parámetros (Sistema 2) + transformer visuomotor de 80M parámetros (Sistema 1) con cross-attention. Opera a 200 Hz. Primer VLA capaz de controlar todo el cuerpo superior del humanoide (cabeza, torso, muñecas, dedos individuales). Entrenado en ~500 horas de datos multi-robot.

**Fortalezas:** Mejor trade-off entre razonamiento semántico y control motor fluido; permite operación en tiempo real; desacoplamiento permite optimizar cada sistema independientemente.
**Limitaciones:** Mayor complejidad de entrenamiento y despliegue; los dos sistemas deben estar alineados para actuar coherentemente.

### 5.6 Tipo Emergente: VLMs Unificados para Robótica

**Gemini Robotics** (Google DeepMind, 2025) representa una dirección emergente donde un MLLM masivo (basado en Gemini 2.0) se fine-tunea directamente para control robótico sin arquitectura dual:

$$\text{Pipeline tradicional: } \text{Percepción} \rightarrow \text{Planificación} \rightarrow \text{Control}$$
$$\text{Gemini Robotics: Modelo unificado que maneja todo}$$

Este modelo opera a 20 Hz con control motor dextero y razonamiento de alto nivel en un solo forward pass. Para compensar la latencia de red (el VLM se aloja en la nube), un action decoder local corre en el computador onboard del robot.

### 5.7 Resumen Taxonómico

| Tipo | Paradigma | Modelos | Params | Hz | Open Source |
|---|---|---|---|---|---|
| 1 | VLM/LLM planner + low-level policies | SayCan, PaLM-E, Code as Policies | 6B–540B | Variable | Parcial |
| 2 | Image/Video generation planner | SuSIE, UniPi | ~1–5B | ~5 Hz | Sí |
| 3 | Hybrid planner | HybridVLA | ~7B | Variable | Parcial |
| 4 | VLM end-to-end (tokens discretos) | RT-2, OpenVLA, QUAR-VLA | 7B–55B | 2–5 Hz | OpenVLA sí |
| 5 | VLM + Diffusion/Flow expert | π₀, GR00T N1, Helix | 2B–7B | 50–200 Hz | π₀, GR00T N1 |
| Emergente | MLLM unificado | Gemini Robotics | >>10B | 20 Hz | No |

---

## 6. Análisis Detallado de Modelos SOTA

### 6.1 RT-2 (Google DeepMind, 2023)

> Brohan, A. et al. (2023). *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. arXiv:2307.15818.

**Significado histórico:** RT-2 es el modelo que acuñó el término "Vision-Language-Action" y demostró por primera vez que un VLM preentrenado en datos web podía transferir conocimiento directamente a control robótico.

**Arquitectura:**
- **Backbone:** PaLI-X (55B) o PaLM-E (12B)
- **Entrada:** Imagen de cámara del robot + instrucción en lenguaje natural
- **Salida:** Secuencia de tokens de acción representando posición (Δx, Δy, Δz), orientación (Δroll, Δpitch, Δyaw), y estado del gripper — codificados como strings de enteros en el rango [0, 255]
- **Control:** Closed-loop a ~3 Hz en un brazo robótico de 7-DoF

**Innovación clave — Co-fine-tuning:** RT-2 no entrena desde cero en datos robóticos. En su lugar, co-fine-tunea el VLM preentrenado en una **mezcla ponderada** de datos web (VQA, captioning) y datos robóticos (demostraciones de manipulación). Los datos robóticos se sobrepesan (upweighted) para compensar su menor volumen.

**Propiedades emergentes:** Evaluado en 6,000 trials, RT-2 exhibió capacidades nunca entrenadas explícitamente:
- **Reconocimiento de símbolos:** "Coloca el objeto sobre el número 3" — comprende numerales sin entrenamiento robótico específico.
- **Razonamiento semántico:** "Escoge algo que pueda servir como martillo improvisado" — selecciona una piedra.
- **Chain-of-thought:** Con prompts apropiados, RT-2 ejecuta razonamiento multi-paso: "Escoge la bebida más adecuada para alguien cansado" → razona → selecciona bebida energética.

**Limitaciones:**
- Modelo cerrado (propietario de Google)
- Tamaño masivo (55B) impracticable para deployment en robots reales
- Frecuencia de control baja (~3 Hz)
- Evaluado principalmente en un solo robot (manipulador de oficina)

### 6.2 Octo (UC Berkeley, 2024)

> Octo Model Team et al. (2024). *Octo: An Open-Source Generalist Robot Policy*. arXiv:2405.12213.

**Significado:** Primer modelo generalista robótico open-source a gran escala, demostrando que un modelo de **93M parámetros** puede competir con RT-2 (55B).

**Arquitectura:**
- **Transformer backbone:** Diseño modular con tokenización flexible
- **Encoder visual:** CNN para patches de imagen
- **Encoder lingüístico:** Modelo de lenguaje para instrucciones
- **Action decoder:** **Diffusion head** — a diferencia de RT-2, no discretiza acciones sino que genera acciones continuas mediante denoising iterativo
- **Observaciones:** Soporta múltiples configuraciones de cámara (tercera persona, muñeca) — **flexible a diferentes setups sensoriales**

**Datos de entrenamiento:** 800,000 trayectorias del **Open X-Embodiment Dataset** — la colección de datos de manipulación robótica más grande disponible públicamente.

**Window size:** 2 timesteps (observa los últimos 2 frames + sus estados), prediciendo acciones de 7 dimensiones a 4 pasos en el futuro.

**Variantes:**
| Variante | Parámetros | Uso |
|---|---|---|
| Octo-Tiny | <93M | Dispositivos con recursos limitados |
| Octo-Small | ~93M | Balance rendimiento/eficiencia |
| Octo-Base | ~93M | Versión completa evaluada |

**Contribución clave — Adaptabilidad:** Octo fue diseñado para ser fine-tuneado eficientemente a nuevas configuraciones robóticas en pocas horas con GPUs de consumo. Funciona como **inicialización generalista** que se especializa con pocos datos in-domain.

**Resultados:** Evaluado en 9 plataformas robóticas distintas, demostrando transferencia efectiva entre diferentes morphologías y espacios de acción.

### 6.3 OpenVLA (Stanford, 2024)

> Kim, M. J. et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246.

**Significado:** VLA open-source de 7B parámetros que supera a RT-2-X (55B) con 7× menos parámetros.

**Arquitectura (tres componentes):**

1. **Dual Vision Encoder:**
   - **DINOv2** (~300M): Excela en relaciones espaciales y estructura geométrica
   - **SigLIP** (~400M): Fuerte alineación semántica con lenguaje
   - Ambos producen embeddings de patches que se concatenan

2. **Projector MLP:** Las representaciones visuales se proyectan al espacio de embedding del LLM mediante un perceptrón multicapa

3. **LLM (Llama 2, 7B):** Tokeniza la instrucción de lenguaje. Los embeddings visuales proyectados y los tokens de texto se concatenan como secuencia de entrada. El LLM genera tokens de acción de manera autoregresiva

**Espacio de acciones:** 7-DoF discretizado: Δx, Δy, Δz (posición), Δroll, Δpitch, Δyaw (orientación), open/close (gripper).

**Datos:** 970K episodios del **Open X-Embodiment Dataset**.

**Resultados cuantitativos:**
- +16.5% de success rate absoluto sobre RT-2-X (55B) en 29 tareas con múltiples embodiments
- +20.4% sobre Diffusion Policy (entrenamiento from-scratch) en entornos multi-tarea
- Fine-tunable con **LoRA** en GPUs de consumo sin pérdida de rendimiento
- Cuantizable (4-bit) para servir eficientemente sin degradación

**Evaluación:** Robot arm Franka Emika Panda de 7-DoF a 5 Hz.

**Insight clave:** Los autores demuestran que **los VLAs se pueden entrenar igual que LLMs** — next-token prediction con cross-entropy loss — necesitando solo 255 tokens de acción para representar el espacio completo de acciones robóticas.

**Limitación importante:** OpenVLA no rinde bien en datos out-of-distribution comparado con RT-2, ya que no fue entrenado en datos web de la misma escala. Fine-tuning en distribuciones no vistas ayuda significativamente a mitigar esta limitación.

### 6.4 π₀ — Physical Intelligence (2024)

> Black, K. et al. (2024). *π₀: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164. RSS 2025.

**Significado:** Modelo VLA fundacional que introduce la arquitectura dual-system con flow matching, demostrando manipulación dextera a 50 Hz en múltiples plataformas.

**Arquitectura (~3.3B parámetros):**

**Sistema 2 (VLM):**
- **Encoder visual:** SigLIP
- **Modelo de lenguaje:** Gemma (base de PaliGemma)
- El VLM procesa imagen + instrucción y genera embeddings contextuales
- **Crucialmente:** El VLM **no observa** los tokens de acción/estado del robot, preservando su conocimiento preentrenado sin contaminación

**Sistema 1 (Action Expert, ~300M):**
- Atiende a todos los tokens previos de visión/lenguaje/estado del robot (cross-attention)
- Genera secuencias de **$H = 50$ tokens de acción** mediante conditional flow matching
- Las acciones son vectores n-dimensionales con magnitud y dirección

**Training pipeline:**
1. Pre-training del VLM en datos web (heredado de PaliGemma)
2. Pre-training del action expert en el **π Cross-Embodiment Robot Dataset** (datos propios de Physical Intelligence, de alta calidad)
3. Fine-tuning conjunto end-to-end en tareas específicas

**Plataformas evaluadas:** Single-arm robots, dual-arm robots y mobile manipulators.

**Tareas demostradas:** Doblar ropa, limpiar mesas, ensamblar cajas — tareas que requieren destreza y planificación multi-paso.

**Resultados:** π₀ supera a OpenVLA y Octo por márgenes significativos en todas las tareas evaluadas, incluyendo las más difíciles que requieren manipulación fina.

**π₀-FAST:** Variante autoregresiva que reemplaza flow matching con tokenización FAST (DCT). Opera en modo autoregresivo para mayor eficiencia en ciertos escenarios, manteniendo la capacidad de control a 50 Hz gracias a la compresión en dominio de frecuencias.

### 6.5 Helix (Figure AI, 2025)

> Comunicación oficial de Figure AI: Helix Technical Blog Post (2025).

**Significado:** Primer VLA comercialmente desplegado en fábricas (BMW) y primero capaz de controlar todo el cuerpo superior de un robot humanoide, incluyendo dedos individuales.

**Arquitectura Dual-System (propietaria, cerrada):**

**Sistema 2 (S2):**
- VLM de ~7B parámetros (open-source, open-weight) preentrenado en datos web
- Procesa imágenes monoculares + información de estado (posición de muñecas, posiciones de dedos) proyectadas al espacio de embedding visión-lenguaje
- Combina con instrucciones en lenguaje natural
- Genera un **vector latente continuo** que codifica toda la información semántica relevante de la tarea

**Sistema 1 (S1):**
- Transformer encoder-decoder de 80M parámetros con cross-attention
- Backbone visual convolucional multi-escala (inicializado con preentrenamiento en simulación)
- Recibe la misma imagen y estado que S2, pero a mayor frecuencia
- El vector latente de S2 se proyecta al espacio de tokens de S1 y se concatena con features visuales
- Opera a **200 Hz** — la frecuencia más alta reportada para un VLA

**Datos de entrenamiento:** ~500 horas de datos multi-robot de alta calidad, recopilados con supervisión humana.

**Capacidades únicas:**
- Control de todo el cuerpo superior: gaze de cabeza, postura del torso, muñecas, y **dedos individuales**
- **Coordinación multi-robot:** Múltiples robots Figure F02 colaborando en tareas coordinadas en tiempo real
- **Generalización out-of-distribution** sin fine-tuning específico por tarea
- Ejecución **completamente onboard** con bajo consumo energético

**Contexto histórico:** Figure AI originalmente utilizaba GPT-4o (OpenAI) como backbone. La decisión de desarrollar Helix in-house refleja la necesidad de control de latencia, privacidad de datos y optimización hardware-specific para despliegue comercial.

### 6.6 GR00T N1 (NVIDIA, 2025)

> NVIDIA (2025). *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*. arXiv:2503.14734.

**Significado:** Modelo fundacional **open-source** para robots humanoides con 2B parámetros, integrado en el ecosistema NVIDIA (Isaac Sim, Omniverse, Cosmos).

**Arquitectura Dual-System (~2B parámetros):**

**Sistema 2 (VLM — Eagle2):**
- **Encoder visual:** SigLIP Vision Transformer (27 capas de SiglipEncoderLayer, 1152 dimensiones, patches 14×14)
- **Modelo de lenguaje:** LlamaForCausalLM (12 capas LlamaDecoderLayer, 2048 dimensiones, 8192 FFN)
- **MLP projector:** LayerNorm → Linear(4608→2048) → GELU → Linear(2048→2048)
- Interpretación semántica del entorno mediante visión e instrucciones

**Sistema 1 (Diffusion Transformer — DiT):**
- 16 bloques BasicTransformerBlock con AdaLayerNorm, Self-Attention, y FeedForward
- Dimensión: 1536
- **TimestepEncoder** para codificación del paso de difusión
- **CategorySpecificMLP** para codificación de estado y acción (adaptable a diferentes embodiments)
- **MultiEmbodimentActionEncoder** con codificación posicional sinusoidal
- Opera a **120 Hz**

**Datos de entrenamiento (heterogéneos):**
- **Trayectorias de robot real:** Episodios de manipulación bimanual
- **Videos humanos:** Demostraciones de tareas cotidianas
- **Datos sintéticos:** Generados con NVIDIA Omniverse y Cosmos — simulación fotorrealista

**Configuración de acción:**
- **Horizonte de acción:** 16 timesteps futuro
- **Brazos:** 7 DoF por brazo (shoulder pitch/roll/yaw, elbow pitch, wrist yaw/roll/pitch)
- **Manos:** 6 DoF por mano (little, ring, middle, index finger + thumb rotation/bending)
- **Torso:** 3 DoF (waist yaw/pitch/roll)

**Resultados:** Supera baselines de imitation learning en benchmarks de simulación estándar en múltiples embodiments. Desplegado en robot humanoide Fourier GR-1 para tareas de manipulación bimanual condicionadas por lenguaje con alta eficiencia de datos.

**Ecosistema:** Integra con LeRobot (Hugging Face) para formato de datos, Isaac Lab para simulación, y el framework Isaac GR00T para despliegue.

### 6.7 ALOHA (Google DeepMind) — Plataforma Hardware

> Zhao, T. Z. et al. (2023). *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. RSS 2023.

ALOHA no es un VLA per se, sino una **plataforma hardware open-source** de bajo costo (~$20,000) para teleoperación bimanual que se ha convertido en el testbed estándar para evaluación de VLAs.

**Características:**
- Dos brazos robóticos con grippers para manipulación bimanual
- Sistema de teleoperación humana para recolección de datos
- Hardware open-source: diseños 3D, instrucciones de montaje, documentación completa

**Importancia para el ecosistema VLA:** ALOHA es usado como benchmark de evaluación por Mobile ALOHA, OpenVLA-OFT, π₀, y RDT-1B, estableciéndose como la plataforma estándar de facto para manipulación bimanual.

### 6.8 QUAR-VLA (MiLAB, 2024)

> Ren, Q. et al. (2024). *QUAR-VLA: Vision-Language-Action Model for Quadruped Robots*. arXiv.

**Especialización:** VLA específico para **robots cuadrúpedos**, abordando el problema de locomoción en terrenos complejos — un dominio donde los robots con patas requieren coordinación multi-articular sofisticada distinta a la manipulación.

**QUART-2:** Toma observaciones visuales e instrucciones, las tokeniza y las pasa al VLM preentrenado para generar tokens de acción discretos (255 bins) a 2 Hz. El dataset de entrenamiento combina datos reales de navegación/locomoción en terreno complejo con datos sintéticos.

### 6.9 Gemini Robotics (Google DeepMind, 2025)

> Gemini Robotics Team (2025). *Gemini Robotics: Bringing AI into the Physical World*. Google DeepMind Technical Report.

**Modelo unificado** basado en Gemini 2.0 multimodal, sin arquitectura dual explícita.

**Variantes:**
- **Gemini Robotics-ER (Embodied Reasoning):** Gemini 2.0 nativo, sin fine-tuning robótico. Aprovecha exclusivamente el conocimiento preentrenado para razonamiento espacial, pointing, y comprensión de escenas robóticas. Supera a GPT-4o, Claude y Molmo en benchmarks de 2D pointing.
- **Gemini Robotics:** Extiende ER con fine-tuning en datos de acción robótica para control directo. Opera a 20 Hz con manipulación dextera.

**Arquitectura de despliegue:**
- VLM distilado alojado en **nube** para razonamiento de alto nivel
- **Action decoder** local en computador onboard del robot para generar acciones compensando la latencia de red

### 6.10 LingBot-VLA (2026)

> Wu, W. et al. (2026). *A Pragmatic VLA Foundation Model*. arXiv:2601.18692.

**Significado:** Modelo VLA pragmático y open-source que ataca el desafío de la generalización bimanual a través de un escalado masivo de datos del mundo real y conciencia espacial profunda (Depth).

**Arquitectura (~4B parámetros):**
Diseñado priorizando la eficiencia computacional. Los autores reportan que su infraestructura (*codebase* basado en VeOmni) logra acelerar el entrenamiento entre un **1.5× y 2.8×** comparado con arquitecturas VLA contemporáneas.

**Datos de entrenamiento:** Representa uno de los saltos cuantitativos más agresivos en datos de entorno real, pre-entrenado en **20,000 horas** de captura provenientes de 9 configuraciones distintas de robots de brazo dual.

**Variantes publicadas:**
- **LingBot-VLA-4B:** Modelo multimodelo estándar (RGB-Language-Action).
- **LingBot-VLA-4B-Depth:** Variante poco común que integra profundidad explícita en su entrada, reduciendo los errores críticos de coordinación espacial 3D y dependencia angular de cámara.

**Relevancia:** Confirma las leyes empíricas emergentes en VLAs: el volumen y calidad real de los datos superan a los enfoques puramente sintéticos o de LLMs inflados.

### 6.11 Tabla Comparativa de Modelos SOTA

| Modelo | Año | Tipo | Parámetros | Frecuencia | Datos | Open Source | Robot Platform |
|---|---|---|---|---|---|---|---|
| RT-2 | 2023 | 4 | 12B–55B | ~3 Hz | Web + Robot | No | Manipulador Google |
| Octo | 2024 | 5* | 93M | ~10 Hz | 800K traj. (OXE) | Sí | 9 plataformas |
| OpenVLA | 2024 | 4 | 7B | 5 Hz | 970K ep. (OXE) | Sí | Franka Panda |
| π₀ | 2024 | 5 | ~3.3B | 50 Hz | π dataset | Parcial | Multi-plataforma |
| Helix | 2025 | 5 | ~7.08B | 200 Hz | ~500h | No | Figure F02 |
| GR00T N1 | 2025 | 5 | ~2B | 120 Hz | Real + Sintético | Sí | Fourier GR-1 |
| QUAR-VLA | 2024 | 4 | ~7B | 2 Hz | Real + Sintético | Parcial | Cuadrúpedos |
| Gemini Rob. | 2025 | Emergente | >>10B | 20 Hz | Web + Robot | No | Multi-plataforma |
| LingBot-VLA | 2026 | Eficiente | ~4B | No rep. | 20,000h (dual-arm)| Sí | 9 dual-arm |

*Nota: Octo usa diffusion head pero no tiene VLM como Sistema 2 explícito; es un diseño anterior al paradigma dual-system.

---

## 7. Paradigmas de Entrenamiento

### 7.1 Behavioral Cloning (Imitación Directa)

El paradigma dominante en VLAs. Dada una colección de demostraciones de experto $\mathcal{D} = \{(o_t, a_t)\}_{t=1}^{T}$ donde $o_t$ son observaciones (imagen + estado) y $a_t$ son acciones, se minimiza:

$$\mathcal{L}_{BC} = \mathbb{E}_{(o,a) \sim \mathcal{D}} \left[-\log \pi_\theta(a | o)\right]$$

Para VLAs tipo 4 (RT-2, OpenVLA), esto se reduce a **cross-entropy loss** entre tokens de acción predichos y tokens ground-truth:

$$\mathcal{L}_{CE} = -\sum_{i=1}^{L} \log P_\theta(a_i | a_{<i}, o, l)$$

donde $a_i$ es el $i$-ésimo token de acción, $o$ la observación visual y $l$ la instrucción en lenguaje.

Para VLAs tipo 5 con diffusion/flow matching, la loss toma la forma de un **score matching** o **flow matching objective**:

$$\mathcal{L}_{FM} = \mathbb{E}_{t, A_0, \epsilon}\left[\|v_\theta(A_t, t) - ({\epsilon} - A_0)\|^2\right]$$

### 7.2 Co-Fine-Tuning con Datos Web

Introducido por RT-2, esta estrategia entrena el modelo simultáneamente en:
- **Datos web:** VQA, captioning, comprensión visual (preserva conocimiento semántico)
- **Datos robóticos:** Demostraciones de manipulación (enseña grounding físico)

Los datos robóticos se sobrepesan (upweighted) típicamente por un factor 10–100× para compensar su menor volumen relativo. El insight es que mantener datos web previene **catastrofic forgetting** del conocimiento semántico preentrenado.

### 7.3 Entrenamiento por Etapas

Varios modelos adoptan un enfoque de entrenamiento **multi-etapa** para estabilizar el proceso:

**π₀:**
1. PaliGemma preentrenado en datos web (heredado)
2. Pre-training del action expert en datos robóticos heterogéneos
3. Fine-tuning conjunto end-to-end en tareas específicas

**GR00T N1:**
1. Eagle2 VLM preentrenado (heredado de NVIDIA)
2. DiT action head preentrenado en datos de simulación
3. Entrenamiento conjunto con mezcla heterogénea de datos reales + sintéticos + videos humanos

**Helix:**
1. VLM backbone preentrenado en datos web (open-source)
2. S1 vision backbone inicializado con preentrenamiento en simulación
3. Entrenamiento supervisado conjunto con regression loss

### 7.4 Parameter-Efficient Fine-Tuning (PEFT)

**OpenVLA** demuestra que los VLAs pueden fine-tunearse eficientemente con técnicas PEFT:

- **LoRA (Low-Rank Adaptation):** Inserta matrices de bajo rango $\Delta W = BA$ donde $B \in \mathbb{R}^{d \times r}$ y $A \in \mathbb{R}^{r \times d}$ con $r \ll d$ en las capas de atención. Solo se entrenan $A$ y $B$, manteniendo los pesos originales congelados.

Los resultados de OpenVLA muestran que modelos fine-tuneados con LoRA rinden **a la par** que los fine-tuneados completamente, con una fracción del costo computacional — habilitando adaptación en **GPUs de consumo**.

Además, OpenVLA puede servirse eficientemente con **cuantización** (4-bit, 8-bit) sin degradación medible en success rate, democratizando el despliegue de VLAs de 7B+ parámetros.

### 7.5 Cross-Embodiment Training

El entrenamiento **cross-embodiment** — usar datos de múltiples robots con diferentes morfologías — es una estrategia clave para la generalización:

**Octo:** Entrenado en 800K trayectorias de >20 embodiments distintos del Open X-Embodiment Dataset. Demuestra transferencia efectiva entre robots con diferentes espacios de observación y acción.

**GR00T N1:** Usa **CategorySpecificMLP** y **MultiEmbodimentActionEncoder** — módulos que adaptan automáticamente la codificación/decodificación de estado y acciones según el embodiment (tag de embodiment como input), permitiendo que un mismo modelo controle robots con diferente número de articulaciones.

### 7.6 Aprendizaje desde Videos Humanos

Una tendencia emergente es entrenar VLAs parcialmente con **videos de demostración humana** (sin acciones robóticas directas). GR00T N1 incluye videos humanos en su mezcla de entrenamiento, aprendiendo intenciones y planificación visual de los movimientos humanos aunque no pueda copiar las acciones exactas (mapeo embodiment distinto).

---

## 8. Datos y Plataformas Robóticas

### 8.1 Datasets Públicos

#### 8.1.1 Open X-Embodiment Dataset

El **Open X-Embodiment (OXE) Dataset** (Collaboration et al., 2024) es la piedra angular del entrenamiento VLA moderno. Resultado de una colaboración de 21 instituciones, reúne:

| Atributo | Valor |
|---|---|
| Episodios totales | >1 millón |
| Embodiments | >20 plataformas robóticas |
| Tareas | Manipulación diversa |
| Formato | RLDS (TensorFlow Datasets) |
| Acceso | Público (Google Cloud) |

Modelos entrenados con OXE: RT-2-X, Octo, OpenVLA.

#### 8.1.2 Bridge Dataset (UC Berkeley)

Dataset de manipulación table-top con un brazo WidowX:
- Tareas: pick-and-place, reordenamiento de objetos
- Usado extensivamente como benchmark de evaluación en Octo y OpenVLA
- Formato compatible con OXE

#### 8.1.3 DROID Dataset

Dataset de manipulación a gran escala con estandarización de hardware y procedimientos de recolección, contribuyendo a la diversidad del ecosistema OXE.

### 8.2 Datos Sintéticos

**NVIDIA Omniverse + Cosmos:** GR00T N1 utiliza datos generados en simulación fotorrealista como componente de su mezcla de entrenamiento. Las ventajas son:
- Escalabilidad ilimitada
- Control total sobre variaciones (iluminación, objetos, configuraciones)
- Etiquetado automático preciso

**Desafío principal:** El **sim-to-real gap** — las políticas entrenadas en simulación pueden no transferirse directamente al mundo real sin técnicas de transfer learning adicionales (domain randomization, domain adaptation).

### 8.3 Estrategias de Recolección de Datos

| Estrategia | Descripción | Usado por |
|---|---|---|
| **Teleoperación** | Humano controla robot remotamente | ALOHA, π₀ dataset, Helix |
| **Kinesthetic teaching** | Humano guía físicamente el brazo | Bridge, algunos OXE |
| **Autonomy + human correction** | Robot actúa, humano corrige errores | DAgger-based approaches |
| **Simulation** | Generados programáticamente | GR00T N1 (Omniverse) |
| **Video humano** | Captura de videos de demostración humana | GR00T N1 |

### 8.4 Plataformas Robóticas

#### 8.4.1 Brazos Manipuladores

| Plataforma | DoF | Coste Aprox. | Usado por |
|---|---|---|---|
| **Franka Emika Panda** | 7 | ~$30,000 | OpenVLA, benchmark estándar |
| **WidowX** | 6 | ~$3,000 | Bridge Dataset, OXE |
| **ALOHA** (bimanual) | 2×6 | ~$20,000 | π₀, OpenVLA-OFT, RDT-1B |
| **UR5/UR10** | 6 | ~$35,000 | Varios OXE contributors |
| **Google Robot** | 7 | Propietario | RT-2, RT-2-X |

#### 8.4.2 Robots Humanoides

| Plataforma | Aplicación | VLA Asociado |
|---|---|---|
| **Figure F02** | Logística industrial (BMW) | Helix |
| **Fourier GR-1** | Investigación, manipulación | GR00T N1 |
| **Tesla Optimus** | General-purpose (futuro) | No publicado |
| **Boston Dynamics Atlas** | Locomoción, demostración | No VLA público |

#### 8.4.3 Robots Cuadrúpedos

| Plataforma | Aplicación | VLA Asociado |
|---|---|---|
| **Unitree Go2/B2** | Navegación, patrullaje | QUAR-VLA |
| **ANYmal** | Inspección industrial | No VLA público |

### 8.5 Requerimientos de Frecuencia por Plataforma

Un aspecto crítico y frecuentemente subestimado es la **frecuencia de control** requerida por diferentes tareas y plataformas:

| Tarea | Frecuencia mínima | Modelos adecuados |
|---|---|---|
| Pick-and-place simple | 2–5 Hz | Todos (RT-2, OpenVLA, Octo) |
| Manipulación dextera | 20–50 Hz | π₀, Gemini Robotics |
| Control de dedos individuales | 50–120 Hz | GR00T N1, Helix |
| Locomoción dinámica | 100–200 Hz | Solo Helix (200 Hz) |

Esta tabla ilustra que **la elección de arquitectura VLA no es independiente de la plataforma target** — los modelos Tipo 4 (tokens discretos, autoregresivos) son inadecuados para tareas que requieren alta frecuencia.

---

## 9. Benchmarks y Evaluación

### 9.1 Métricas Principales

#### 9.1.1 Success Rate (Tasa de Éxito)

La métrica más reportada: porcentaje de episodios en los que el robot completa exitosamente la tarea instruccionada.

$$\text{Success Rate} = \frac{\text{Episodios exitosos}}{\text{Episodios totales}} \times 100\%$$

**Limitación:** Binaria — no captura calidad de ejecución, eficiencia temporal, o suavidad del movimiento.

#### 9.1.2 Generalization Metrics

Se evalúa la capacidad del modelo en condiciones progresivamente diferentes al entrenamiento:

| Nivel | Descripción | Ejemplo |
|---|---|---|
| **In-distribution** | Mismos objetos, entorno, instrucciones | Baseline de referencia |
| **Novel objects** | Objetos no vistos en training | "Agarra el aguacate" (solo visto manzanas) |
| **Novel instructions** | Comandos lingüísticos nuevos | "Colócalo al lado del..." |
| **Novel environments** | Fondos, iluminación, configuraciones distintas | Nuevo laboratorio |
| **Novel embodiments** | Robot diferente al de entrenamiento | Entrenado en Panda, evaluado en UR5 |

RT-2 demostró +3× mejora en tareas con **objetos y comandos novedosos** respecto a RT-1, gracias al conocimiento web transferido.

### 9.2 Benchmarks de Simulación

| Benchmark | Tipo | Tareas | Usado por |
|---|---|---|---|
| **SIMPLER** | Simulación de manipulación | Pick-and-place, stacking | GR00T N1, OpenVLA |
| **RLBench** | Simulación multi-tarea | 100+ tareas | Varios |
| **CALVIN** | Manipulación secuencial | Cadenas de subtareas | VLAs de planificación |
| **Meta-World** | Manipulación parametrizada | 50 tareas | Octo ablations |

### 9.3 Evaluación en el Mundo Real

La evaluación real sigue siendo el gold standard pero presenta desafíos:

**RT-2:** 6,000 trials de evaluación — la evaluación más extensiva publicada.
**OpenVLA:** 29 tareas en múltiples embodiments.
**π₀:** Tareas complejas de manipulación dextera (doblar ropa, ensamblar cajas) — cualitativamente más difíciles.
**Helix:** Despliegue comercial en fábricas BMW — la primera evaluación de VLA en producción industrial real.

### 9.4 Resultados Comparativos Clave

| Comparación | Resultado | Significado |
|---|---|---|
| OpenVLA (7B) vs. RT-2-X (55B) | +16.5% success rate | Más pequeño no es peor; la calidad de datos y arquitectura importa más que la escala |
| OpenVLA vs. Diffusion Policy | +20.4% en multi-tarea | Pre-training VLM supera entrenamiento from-scratch |
| π₀ vs. OpenVLA/Octo | Márgenes significativos | Flow matching + dual-system > tokens discretos |
| GR00T N1 vs. baselines IL | Superior en simulación | Datos heterogéneos + dual-system escalan bien |

### 9.5 Limitaciones de la Evaluación Actual

> **Pregunta abierta:** No existe un benchmark estandarizado y universalmente adoptado para evaluar VLAs en el mundo real. Cada grupo de investigación evalúa en sus propias configuraciones, haciendo las comparaciones directas difíciles.

Problemas específicos:
- **Variabilidad hardware:** Diferentes robots, grippers, cámaras
- **Condiciones no controladas:** Iluminación, temperatura, desgaste mecánico
- **Métricas insuficientes:** Success rate no captura calidad del movimiento
- **Reproducibilidad limitada:** Configuraciones físicas no replicables entre laboratorios
- **Sesgo de evaluación:** Tendencia a reportar solo tareas donde el modelo funciona bien

### 9.6 ROBOGATE: Evaluación Adaptativa de Fronteras de Fallo (2026)

> Kim, A. (2026). *ROBOGATE: Adaptive Failure Discovery for Safe Robot Policy Deployment via Two-Stage Boundary-Focused Sampling*. AgentAI Co., Ltd.

**ROBOGATE** propone un framework open-source de gestión de riesgo pre-despliegue que aborda directamente las limitaciones de las evaluaciones estáticas. Combina simulación física (NVIDIA Isaac Sim) con una **estrategia de muestreo adaptativo en dos etapas** para descubrir eficientemente las fronteras de fallo de una política robótica en un espacio de 8 parámetros (fricción, masa, offset de centro de masa, tamaño, ruido IK, obstáculos, geometría y modo de colocación):

1. **Etapa 1 — Exploración uniforme (20,000 experimentos):** Latin Hypercube Sampling (LHS) cubre el espacio de parámetros de manera uniforme y produce un mapa grueso de zonas seguras, fronterizas y peligrosas.
2. **Etapa 2 — Muestreo enfocado en la frontera (10,000 experimentos):** Se identifican los bins donde la tasa de éxito cae entre el 30% y el 70%, concentrando los experimentos en la zona de transición crítica. Esto mejora la cobertura de la frontera en un 31.1% y eleva el AUC del modelo de riesgo de 0.754 a 0.780.

**Validación cross-embodiment:** La evaluación paralela sobre Franka Panda (7-DOF, pinza paralela) y UR5e (6-DOF, pinza de succión) en 30,000 experimentos totales revela **cuatro zonas de peligro universales** independientes del embodiment, todas asociadas a masas de objeto superiores a 0.935 kg.

**Modelo de riesgo interpretable:** En lugar de modelos de caja negra, ROBOGATE ajusta una regresión logística que produce una ecuación de frontera en formato cerrado:

$$\mu^*(m) = \frac{1.469 + 0.419m}{3.691 - 1.400m}$$

Esta ecuación se traduce directamente en restricciones operativas (ej. "no desplegar si la fricción del objeto < 0.49").

**Evaluación de Octo-Small:** Sometido al suite de 68 escenarios adversarios de ROBOGATE (iluminación baja, objetos transparentes, oclusiones, posiciones atípicas), el VLA generalista **Octo-Small logra un 0.0% de tasa de éxito** frente al 100% del controlador scripted de referencia, incluso bajo condiciones nominales — lo que apunta a una discordancia perceptivo-motora fundamental y no a la dificultad del entorno.

---

## 10. Análisis Crítico: Limitaciones, Preguntas Abiertas y Direcciones Futuras

### 10.1 Fortalezas del Paradigma VLA

**Unificación y simplificación:** El paradigma VLA reduce la complejidad del pipeline robótico clásico (percepción → planificación → control) a un único modelo entrenado end-to-end, eliminando la propagación de errores inter-módulo y la ingeniería manual intensiva.

**Transferencia de conocimiento web:** La demostración de RT-2 de propiedades emergentes — razonamiento semántico, reconocimiento de símbolos, chain-of-thought — establece que el preentrenamiento en datos web es una fuente de conocimiento insustituible para robótica generalista.

**Democratización progresiva:** Desde RT-2 (55B, cerrado) hasta OpenVLA (7B, open-source, fine-tunable con LoRA en GPU de consumo) y GR00T N1 (2B, open-source), hay una tendencia clara hacia modelos más accesibles y eficientes.

**Versatilidad cross-embodiment:** Octo, π₀ y GR00T N1 demuestran que un solo modelo puede operar en múltiples plataformas robóticas con diferentes morfologías — un paso hacia la "política universal de manipulación".

### 10.2 Limitaciones Técnicas Actuales

#### 10.2.1 Eficiencia de Datos

Los VLAs actuales requieren **cientos de miles a millones** de episodios de demostración para alcanzar rendimiento competitivo. En contraste, un humano puede aprender una nueva tarea de manipulación con 1–5 demostraciones. Esta brecha de eficiencia de datos es el obstáculo más significativo para el despliegue escalable.

| Modelo | Datos de entrenamiento robótico | Tareas aprendidas |
|---|---|---|
| RT-2 | ~130K episodios + datos web masivos | ~200 variaciones |
| Octo | 800K trayectorias | Generalista + fine-tuning |
| OpenVLA | 970K episodios | 29 tareas evaluadas |
| π₀ | Propio (tamaño no público) | Multi-plataforma dextero |
| GR00T N1 | Real + sintético (tamaño no público) | Manipulación bimanual |

#### 10.2.2 Generalización y Robustez

Aunque los VLAs muestran mejoras significativas en generalización respecto a políticas específicas, los límites de generalización son aún estrechos:

- **Domain shift:** Cambios de iluminación, fondos o tipos de superficie pueden degradar el rendimiento.
- **Catastrophic forgetting:** El fine-tuning en nuevas tareas puede degradar las capacidades preentrenadas.
- **Open-world generalization:** Los VLAs actuales no logran la adaptación fluida a entornos completamente no estructurados que los humanos manejan sin esfuerzo.

**Evidencia empírica cuantitativa — ROBOGATE (Kim, 2026):** El framework de evaluación adversaria ROBOGATE reveló que **Octo-Small obtiene un 0.0% de tasa de éxito** en 68 escenarios adversarios frente al 100% de un controlador scripted, incluso bajo condiciones nominales. Esta brecha de 100 puntos porcentuales confirma que los modelos VLA generalistas actuales no superan un umbral de validación industrial básico. La causa principal es la **discordancia perceptivo-motora** entre la distribución de entrenamiento (datos reales del Open X-Embodiment) y la distribución de evaluación (simulación con dominio visual distinto), agravada por una escala de acción de 2 cm por paso que impide la precisión sub-centimétrica requerida para tareas industriales.

> **Pregunta abierta:** ¿Cuántos datos robóticos de calidad se necesitan verdaderamente para que un VLA generalice a nivel humano? ¿Existe un umbral de datos a partir del cual emergen capacidades de adaptación comparable a las humanas, análogo a los scaling laws de los LLMs?

#### 10.2.3 Escalabilidad Computacional

El trade-off entre capacidad del modelo y costo de inferencia en tiempo real es un desafío fundamental:

| Modelo | Parámetros | GPU Inference | Latencia estimada |
|---|---|---|---|
| RT-2 (55B) | 55B | TPU cluster | ~300ms/acción |
| OpenVLA | 7B | 1× A100 | ~200ms/acción |
| π₀ | 3.3B | 1× A100 | ~20ms/acción |
| GR00T N1 | 2B | 1× GPU | ~8ms/acción |
| Helix S1 | 80M | Onboard | ~5ms/acción |

La tendencia hacia **modelos más pequeños pero más eficientes** (GR00T N1 con 2B, Helix S1 con 80M) refleja las restricciones prácticas del despliegue robótico, donde el cómputo onboard es limitado y la latencia importa.

#### 10.2.4 Alineación Multi-Modal

La precisión del **grounding** entre las tres modalidades (visión, lenguaje, acción) no está completamente resuelta:

- El modelo puede "entender" la instrucción y "ver" el objeto pero generar una trayectoria imprecisa
- La calibración cámara-robot introduce errores que el modelo debe absorber
- Las representaciones visuales optimizadas para comprensión semántica no son necesariamente óptimas para estimación de pose 6D

### 10.3 Preguntas Abiertas de Investigación

1. **Tokenización óptima:** ¿Discretización (OpenVLA), difusión (Octo), flow matching (π₀), o FAST (π₀-FAST)? No existe evidencia empírica controlada que establezca cuándo usar cada enfoque. Los benchmarks actuales favorecen modelos distintos en tareas distintas.

2. **Mínimo de datos para fine-tuning:** ¿Cuántas demostraciones se necesitan para adaptar un VLA preentrenado a una nueva tarea con rendimiento confiable? OpenVLA muestra que LoRA funciona con datos limitados, pero no hay cuantificación sistemática.

3. **Coordinación multi-agente:** Helix es el único VLA con capacidades multi-robot demostradas. ¿Cómo escalar los paradigmas VLA a flotas de robots coordinados? ¿Se necesita un único modelo centralizado o múltiples agentes independientes?

4. **Safety y robustez:** ¿Cómo garantizar que un VLA no ejecute acciones peligrosas en entornos con humanos? Los modelos actuales no incorporan restricciones de seguridad explícitas — un obstáculo fundamental para certificación industrial.

5. **Razón vs. acción:** El paradigma dual-system (Tipo 5) asume que separar razonamiento lento de ejecución rápida es óptimo. ¿Existe evidencia de que un modelo verdaderamente unificado (como Gemini Robotics) puede competir con la eficiencia del diseño dual?

6. **Evaluación estandarizada:** La falta de benchmarks universales hace imposible la comparación justa entre modelos. La comunidad necesita un equivalente al ImageNet o GLUE para VLAs.

### 10.4 Direcciones Futuras

#### 10.4.1 Scaling Laws para VLAs

Los LLMs obedecen **scaling laws** predecibles (Kaplan et al., 2020): el rendimiento mejora como función potencial de parámetros, datos y cómputo. ¿Existen scaling laws análogas para VLAs? Los datos iniciales son prometedores (OpenVLA con 7B supera a RT-2-X con 55B en muchas tareas), pero la relación exacta entre escala y capacidad robótica no está establecida.

#### 10.4.2 World Models

Una dirección emergente es integrar **world models** — modelos que predicen cómo el mundo evoluciona en respuesta a las acciones del robot — como componente central del VLA. Esto permitiría planificación prospectiva: "Si hago X, el mundo pasará a estado Y, ¿es Y deseable?"

#### 10.4.3 Aprendizaje Continuo

Los VLAs actuales se entrenan offline en datasets fijos. Para despliegue real a largo plazo, los robots necesitan **continual learning**: actualizar sus conocimientos incrementalmente sin olvidar tareas anteriores y sin reentrenamiento completo.

#### 10.4.4 VLAs Multimodales Extendidos

La integración de modalidades adicionales — audio, táctil, fuerza, propiocepción — podría cerrar la brecha entre la percepción robótica y la humana. ONE-PEACE (Wang et al., 2023) y VALOR (Chen et al., 2023) ya exploran visión + lenguaje + audio, pero la incorporación de sensores táctiles en VLAs permanece largamente inexplorada.

#### 10.4.5 Convergencia Industria-Academia

La trayectoria desde RT-2 (Google, cerrado) → OpenVLA (Stanford, open-source) → GR00T N1 (NVIDIA, open-source) → Helix (Figure AI, comercial) ilustra una convergencia saludable entre investigación abierta y despliegue industrial. El desafío es mantener este equilibrio a medida que los VLAs se vuelven comercialmente valiosos.

---

## 11. Referencias

### Surveys y Meta-Análisis
- Kawaharazuka, K., Oh, J., Yamada, J., Posner, I. & Zhu, Y. (2025). Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications. *IEEE Access*, 13, 162467–162504. arXiv:2510.07077.
- Fields, C. & Kennington, C. (2023). Vision Language Transformers: A Survey. *arXiv:2307.03254v1*.

### Fundamentos: Transformer y Preentrenamiento
- Vaswani, A. et al. (2017). Attention is All You Need. *NeurIPS 2017*.
- Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv:1810.04805*.
- Radford, A. et al. (2018). Improving Language Understanding by Generative Pre-Training. OpenAI.
- Dosovitskiy, A. et al. (2020). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. *arXiv:2010.11929*.
- Touvron, H. et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv:2302.13971*.

### Vision-Language Models (VLMs)
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). *ICML 2021*.
- Jia, C. et al. (2021). Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN). *ICML 2021*.
- Zhai, X. et al. (2023). Sigmoid Loss for Language Image Pre-Training (SigLIP). *ICCV 2023*.
- Oquab, M. et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. *TMLR 2024*.
- Chen, X. et al. (2023). PaLI-X: On Scaling up a Multilingual Vision and Language Model. *arXiv:2305.18565*.
- Driess, D. et al. (2023). PaLM-E: An Embodied Multimodal Language Model. *ICML 2023*. arXiv:2303.03378.
- Beyer, L. et al. (2024). PaliGemma: A versatile 3B VLM for transfer. *arXiv:2407.07726*.
- Alayrac, J.-B. et al. (2022). Flamingo: a Visual Language Model for Few-Shot Learning. *NeurIPS 2022*. arXiv:2204.14198.
- Li, J. et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. *ICML 2023*. arXiv:2301.12597.
- OpenAI (2023). GPT-4 Technical Report. *arXiv:2303.08774*.

### VLA Tipo 1: Planificadores de Alto Nivel
- Ahn, M. et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan). *arXiv:2204.01691*.
- Liang, J. et al. (2023). Code as Policies: Language Model Programs for Embodied Control. *ICRA 2023*.

### VLA Tipo 2: Generadores de Imagen como Planificadores
- Black, K. et al. (2023). Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models (SuSIE). *arXiv:2310.10639*.
- Du, Y. et al. (2024). Learning Universal Policies via Text-Guided Video Generation (UniPi). *NeurIPS 2024*.

### VLA Tipo 4: End-to-End con Tokens Discretos
- Brohan, A. et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *arXiv:2307.15818*. Google DeepMind.
- Brohan, A. et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. *arXiv:2212.06817*. Google.
- Kim, M. J. et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. *arXiv:2406.09246*. Stanford University.
- Ren, Q. et al. (2024). QUAR-VLA: Vision-Language-Action Model for Quadruped Robots. *arXiv*. MiLAB.

### VLA Tipo 5: Dual-System (VLM + Diffusion/Flow)
- Black, K. et al. (2024). π₀: A Vision-Language-Action Flow Model for General Robot Control. *arXiv:2410.24164*. Physical Intelligence. RSS 2025.
- Pertsch, K. et al. (2025). π₀-FAST: Efficient Robot Control with Frequency-space Action Sequence Tokenization. *Physical Intelligence*.
- NVIDIA (2025). GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. *arXiv:2503.14734*.
- Figure AI (2025). Helix: A Generalist VLA Model for Humanoid Control. Technical Blog Post.

### VLA Emergente: MLLMs Unificados
- Gemini Robotics Team (2025). Gemini Robotics: Bringing AI into the Physical World. *Google DeepMind Technical Report*.

### Datasets y Plataformas
- Open X-Embodiment Collaboration et al. (2024). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. *ICRA 2024*.
- Zhao, T. Z. et al. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ALOHA). *RSS 2023*.
- Chi, C. et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *RSS 2023*.

### Evaluación y Seguridad en Despliegue
- Kim, A. (2026). ROBOGATE: Adaptive Failure Discovery for Safe Robot Policy Deployment via Two-Stage Boundary-Focused Sampling. AgentAI Co., Ltd. Repositorio: github.com/liveplex-cpu/robogate. Dataset: huggingface.co/datasets/liveplex/robogate-failure-dictionary.

### Paradigmas de Aprendizaje y Representación
- Ho, J., Jain, A. & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020*.
- Lipman, Y. et al. (2023). Flow Matching for Generative Modeling. *ICLR 2023*.
- Hu, E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
- Kaplan, J. et al. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.

### Multimodal Extendido y Futuro
- Wang, P. et al. (2023). ONE-PEACE: Exploring One General Representation Model Toward Unlimited Modalities. *arXiv:2305.11172*.
- Chen, S. et al. (2023). VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset. *arXiv:2304.08345*.

### Marco Conceptual
- Harnad, S. (1990). The Symbol Grounding Problem. *Physica D, 42(1):335–346*.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.

---

## 12. Síntesis Final

Los Vision-Language-Action Models representan la convergencia de tres líneas de investigación previamente independientes — visión por computador, procesamiento de lenguaje natural e inteligencia robótica — en un paradigma unificado. En apenas dos años (2023–2025), el campo ha progresado desde RT-2 (primer VLA formal, 55B parámetros, 3 Hz, cerrado) hasta GR00T N1 (2B parámetros, 120 Hz, open-source) y Helix (200 Hz, desplegado comercialmente en fábricas), una trayectoria de compresión, aceleración y democratización sin precedentes.

El progreso se sustenta en **cinco pilares fundamentales**:

1. **Herencia de conocimiento web:** Los VLAs no aprenden robótica desde cero; heredan representaciones semánticas ricas de VLMs preentrenados en datos web a escala de internet. El insight fundacional de RT-2 — que las propiedades emergentes de los VLMs se transfieren al dominio robótico — sigue siendo la contribución conceptual más importante del campo.

2. **Unificación de representaciones:** La representación de visión, lenguaje y acciones como secuencias de tokens procesables por transformers elimina las barreras artificiales entre modalidades. Desde la discretización directa (RT-2/OpenVLA) hasta flow matching (π₀) y diffusion transformers (GR00T N1), cada innovación expande los límites de lo representable.

3. **La arquitectura dual-system como paradigma dominante:** La separación en Sistema 2 (VLM para razonamiento "lento") y Sistema 1 (diffusion/flow para ejecución "rápida") — inspirada en la teoría de procesos duales de Kahneman — ha emergido como el diseño más efectivo, adoptado independientemente por π₀, GR00T N1 y Helix. Este diseño resuelve elegantemente la tensión entre razonamiento semántico profundo y control motor de alta frecuencia.

4. **Ecosistema de datos abierto:** El Open X-Embodiment Dataset, ALOHA como plataforma estándar, y la adopción de formatos como LeRobot (Hugging Face) y RLDS (TensorFlow) han creado un ecosistema de datos compartido que permite comparación, reproducibilidad y entrenamiento cross-embodiment a escala.

5. **Convergencia open-source:** La trayectoria RT-2 (cerrado) → Octo / OpenVLA (open-source) → GR00T N1 (open-source, incluyendo pesos y framework) demuestra que el modelo de investigación abierta es viable y competitivo incluso frente a modelos comerciales masivos.

Las **preguntas abiertas más urgentes** al momento de esta revisión (marzo 2025) son: (i) si existen scaling laws predecibles para VLAs análogas a las de LLMs, (ii) cómo reducir los requerimientos de datos robóticos en órdenes de magnitud (del millón de episodios a las pocas demostraciones), (iii) cómo garantizar seguridad certificable para despliegue en entornos con humanos, y (iv) si el paradigma dual-system es óptimo o si los MLLMs unificados (Gemini Robotics) eventualmente lo superarán.

La tendencia es clara: la inteligencia robótica generalista — robots que ven, comprenden instrucciones y actúan competentemente en entornos no estructurados — ya no es ciencia ficción. Es un problema de ingeniería a escala, cuya resolución depende de datos, cómputo y las decisiones arquitectónicas correctas. Los VLAs son la arquitectura candidata más prometedora para esta resolución.

---

*Revisión elaborada sobre: Kawaharazuka et al. (2025), Brohan et al. (2023), Kim et al. (2024), Black et al. (2024), Octo Model Team et al. (2024), NVIDIA (2025), y fuentes complementarias. Marzo 2025.*
