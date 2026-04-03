# Revisión Bibliográfica Completa: Vision Language Transformers

**Fuente principal:** Fields, C. & Kennington, C. (2023). *Vision Language Transformers: A Survey*. arXiv:2307.03254v1. Boise State University.

---

## 1. Contexto y Motivación

La **Visión-Lenguaje (VL)** es el dominio donde la Visión por Computador (CV) y el Procesamiento de Lenguaje Natural (NLP) se intersectan. Las tareas VL típicas incluyen:

- **Visual Question Answering (VQA):** Dada una imagen y una pregunta, el modelo elige la respuesta correcta.
- **Image Captioning:** Dado una imagen, el modelo genera texto descriptivo.
- **Visual Grounding / Referential Resolution:** Dada una expresión lingüística, localizar el objeto en la imagen al que se refiere.
- **Image-Text Retrieval:** Dada una consulta de texto, recuperar y ordenar imágenes relevantes.

Históricamente, estas tareas fueron extremadamente difíciles para las computadoras. Los modelos anteriores eran conceptualmente complejos y estaban confinados a un rango muy limitado de usos: por ejemplo, DGAF (Gao et al., 2019) para VQA, MAttNet (Yu et al., 2018) para referential resolution, y R2C (Zellers et al., 2018) para razonamiento visual. Cada modelo era diseñado *ad hoc* para una tarea específica y difícilmente adaptable a otras.

La irrupción de los **transformers preentrenados** cambió radicalmente el panorama, primero en NLP y luego en CV, haciendo natural su extensión al dominio VL.

---

## 2. Fundamentos: La Arquitectura Transformer

> Vaswani, A. et al. (2017). *Attention is all you need*. NeurIPS.

Esta es la arquitectura base de todos los modelos discutidos.

### 2.1 Encoder y Decoder

Un transformer original sigue diseño **encoder-decoder**:

- **Encoder:** Apila $N$ capas idénticas. Cada capa tiene dos sub-capas: Multi-Head Attention (MHA) y Feed-Forward Network (FFN). Alrededor de cada sub-capa hay una conexión residual seguida de normalización de capa:

$$\text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$$

- **Decoder:** También apila $N$ capas, pero añade una tercera sub-capa de *cross-attention* al output del encoder. Además, la self-attention del decoder **enmascara posiciones futuras** para que las predicciones sólo dependan de outputs pasados — ideal para tareas generativas.

### 2.2 Mecanismo de Atención Multi-Cabeza (MHA)

La innovación central del transformer. La **atención escalada dot-product** se define como:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Donde $Q$ (queries), $K$ (keys), $V$ (values) son matrices derivadas del input. El escalado por $\sqrt{d_k}$ estabiliza los gradientes.

La **atención multi-cabeza** proyecta $Q$, $K$, $V$ linealmente $h$ veces con distintas matrices aprendidas $W^Q_i$, $W^K_i$, $W^V_i$, aplica atención en paralelo y concatena los resultados:

$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

Esto permite al modelo atender información de **diferentes subespacios de representación** simultáneamente.

### 2.3 Feed-Forward Network (FFN)

Cada capa posee una red feed-forward de dos capas lineales y ReLU:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

$W_1$ proyecta de $d_{model}$ a $d_{ff}$, y $W_2$ proyecta de vuelta. El transformer original usó $N=6$ capas, $d_{model}=512$, $d_{ff}=2048$.

### 2.4 Codificación Posicional

La atención no tiene orden intrínseco. Por ello, se añade una **codificación posicional** a los embeddings de entrada para inyectar información de orden.

---

## 3. Transformers Preentrenados: De NLP a CV

### 3.1 En NLP

**GPT** (Radford et al., 2018): Stack de bloques decoder preentrenado en BooksCorpus con objetivo de **Language Modeling** — predecir el siguiente token dado $k$ tokens anteriores. Establece nuevo SotA en diversas tareas al ser fine-tuneado. Escala hasta **GPT-4** (OpenAI, 2023), combinando LM con Reinforcement Learning.

**BERT** (Devlin et al., 2018): Stack de bloques encoder. Introduce el objetivo de **Masked Language Modeling (MLM)** — predice tokens "[MASK]" condicionado bidireccionalmente en los tokens no enmascarados. Extraordinariamente capaz en tareas de comprensión; se convirtió en el modelo NLP más usado.

Extensiones notables: **RoBERTa** (Liu et al., 2019) — BERT con hiperparámetros optimizados; **ALBERT** (Lan et al., 2019) — versión más compacta; **DistilBERT** (Sanh et al., 2019) — comprimido para inferencia rápida.

### 3.2 En Visión por Computador

**ViT — Vision Transformer** (Dosovitskiy et al., 2020): Las imágenes se dividen en parches $P \times P$, se aplanan, se proyectan linealmente como *patch embeddings* y se introducen como secuencia al encoder. Igualan o superan CNNs con suficientes datos. **Desventaja:** Requieren más datos de preentrenamiento por carecer de los inductive biases de las CNNs (localidad, estructura 2D, equivariancia traslacional).

**BEiT** (Bao et al., 2021): Arquitectura ViT con **Masked Image Modeling (MIM)** — análogo al MLM pero para parches de imagen. Los parches se "tokenizan" a representaciones discretas con un auto-encoder; el modelo predice el token del parche enmascarado. Funciona con menos datos que ViT.

**CoAtNet** (Dai et al., 2021): Combina capas de convolución en profundidad con atención.

**Swin** (Liu et al., 2021) y **CSwin** (Dong et al., 2021): Transformers visuales jerárquicos que aplican self-attention sobre ventanas 2D deslizantes.

---

## 4. Estrategias de Embedding en Modelos VL

Para procesar conjuntamente imagen y texto, los modelos VL deben convertir ambas modalidades a secuencias de vectores en el mismo espacio de representación.

### 4.1 Embeddings Textuales

Prácticamente todos los modelos VL adoptan la estrategia de BERT:

1. **Tokenización:** WordPiece (Wu et al., 2016), BPE/Byte-Pair Encoding (Sennrich et al., 2015) o SentencePiece (Kudo & Richardson, 2018).
2. La secuencia comienza con **[CLS]** y los segmentos se separan con **[SEP]**.
3. Se suman embeddings de **posición** y de **segmento** al embedding del token.

### 4.2 Embeddings Visuales

Esta es la dimensión de mayor variación entre modelos.

#### 4.2.1 Region Features (Características de Región)

*Usadas por:* UNITER, ViLBERT, VL-BERT, VisualBERT, Oscar, VinVL, LXMERT, VL-T5, Unicoder-VL, UNIMO.

Se extrae mediante una red de detección de objetos (Fast R-CNN, YOLO) cuadros delimitadores de regiones de interés (RoI). Para cada RoI: vector del penúltimo hidden state + embedding de posición del bounding box → proyección lineal → concatenar con embeddings textuales.

**Limitaciones:**
- Solo reconocen las categorías del detector utilizado (techo teórico de vocabulario visual).
- Computacionalmente costosas — cuello de botella en inferencia.

#### 4.2.2 Grid Features (Características de Cuadrícula)

*Usadas por:* PixelBERT, SOHO, E2E-VLP, CLIP, ALIGN, LiT, GPV, mDETR, DQ-DETR, UniTAB, Referring Transformer, KD-VLP.

Una CNN extrae un mapa de características $f \in \mathbb{R}^{C \times H \times W}$. Una convolución $1\times1$ reduce a dimensión $d$, y se aplana a una secuencia de $H \cdot W$ vectores.

**Ventajas:** Elimina el techo de categorías; representación densa.  
**Desventajas:** Aún requiere CNN preentrenada; el preprocesamiento sigue siendo el mayor tiempo de inferencia (>90% en PixelBERT).

#### 4.2.3 Patch Embeddings (Embeddings de Parche)

*Usados por:* ViLT, VLMo, ALBEF, BEiT-3, BLIP-2, CoCa, SimVLM, OFA, METER, mPLUG, BridgeTower, DaVinci, Florence, FLAVA, X2-VLM.

Originados en ViT; introducidos en VLT por **ViLT** (Kim et al., 2021). Una imagen $I \in \mathbb{R}^{3 \times H \times W}$ se divide en $N = HW/P^2$ parches, se aplanan y se proyectan linealmente:

$$p \in \mathbb{R}^{N \times (P^2 \cdot C)} \xrightarrow{\text{proyección lineal}} \text{patch embeddings}$$

Con $P=32$, ViLT usa una fracción mínima del cómputo de modelos anteriores.

**Debate abierto:** ViLT rendía peor que modelos con region features, pero METER y BridgeTower con patch embeddings superan SotA — sugiere que la diferencia no es inherente al tipo de embedding.

---

## 5. Arquitecturas de Modelos VL

La arquitectura determina **cómo interactúan** las representaciones visuales y textuales.

### 5.1 Dual Encoders

*Modelos:* **CLIP**, **ALIGN**, **LiT**, **Florence**.

Codifican las modalidades **por separado**; la interacción se produce mediante similitud coseno (o producto punto). Sin interacción profunda dentro del modelo.

| Modelo | Imagen Encoder | Texto Encoder | Dato clave |
|---|---|---|---|
| CLIP | ResNet50 / ViT | Transformer 12L + BPE | 400M pares propietarios |
| ALIGN | EfficientNet | BERT | 1.8B pares, mínimo filtrado |
| LiT | ViT (pesos congelados) | Transformer | 4B pares; encoder imagen freezado |
| Florence | CSwin (jerárquico) | RoBERTa | +Cabezas para tareas adicionales y video |

**Fortalezas:** Inferencia rápida; excelentes en zero-shot classification y retrieval.  
**Limitaciones:** Rendimiento pobre en clasificación VL compleja (NLVR2) — la interacción superficial no basta.

### 5.2 Fusion Encoders

La interacción ocurre *dentro* del modelo de deep learning.

#### 5.2.1 Single-Tower (Una Torre)

*Modelos:* ViLT, VL-BERT, UNITER, Oscar, SOHO, UNIMO, PixelBERT, Unicoder-VL, VisualBERT, VILLA, VinVL, KD-VLP.

Un único transformer opera sobre una **concatenación** de embeddings visuales y textuales. Arquitectura más simple; menos parámetros que el diseño de dos torres. Muchos se inicializan con pesos BERT (ViLT con pesos ViT).

#### 5.2.2 Two-Tower (Dos Torres)

*Modelos:* ViLBERT, LXMERT, METER, BridgeTower.

Dos stacks transformer separados (uno por modalidad) interactúan mediante **cross-attention**. En ViLBERT los keys y values de cada modalidad se intercambian con la otra:

$$\mathbf{H}^{(i)}_V, \mathbf{H}^{(i)}_T \xrightarrow{\text{co-attention}} \text{feature multimodal}$$

- **ViLBERT** (Lu et al., 2019): Texto inicializado con BERT-base.
- **LXMERT** (Tan & Bansal, 2019): Entrenado desde cero; similar a ViLBERT.
- **METER** (Dou et al., 2022): Amplio estudio de arquitecturas; RoBERTa + CLIP-ViT-224/32.
- **BridgeTower** (Xu et al., 2022): Añade *bridge connections* — las 6 capas superiores de cada encoder unimodal conectan a cada capa del encoder multimodal antes del cross-attention; supera a METER en varios benchmarks.

### 5.3 Combination Encoders

*Modelos:* VLMo, ALBEF, BEiT-3, FLAVA, X2-VLM.

Combinan encoders unimodales **más** un módulo de fusión, aprovechando las fortalezas de dual encoders y fusion encoders.

- **VLMo** (Bao et al., 2022): Reemplaza el FFN de cada capa por un pool de expertos de modalidad: V-FFN, L-FFN, VL-FFN. Para pares imagen-texto: expertos unimodales en capas bajas → VL-FFN en capas superiores.
- **BEiT-3** (Wang et al., 2023): Arquitectura similar a VLMo, masivamente escalado.
- **ALBEF** (Li et al., 2021): Encoders unimodales + alineación coseno (como CLIP) + fusion encoder de 6 capas con cross-attention.
- **X2-VLM** (Zeng et al., 2022): Tres módulos + maneja entradas de video.

### 5.4 Encoder-Decoder Models

*Modelos:* VL-T5, OFA, OmniVL, PaLI, E2E-VLP, SimVLM, mPLUG, Flamingo, CoCa, GIT, LEMON, DaVinci; grounded: mDETR, DQ-DETR, UniTAB, KD-VLP, Referring Transformer.

Siguen el diseño original del transformer. **Son los más versátiles**: permiten tareas generativas, de comprensión y unificadas.

- **VL-T5** (Cho et al., 2021): Reformula *todas* las tareas VL como generación de texto — sin cambios de arquitectura entre tareas.
- **Flamingo** (Alayrac et al., 2022): Módulo **Perceiver Resampler** convierte output visual a número fijo de tokens. El decoder alterna bloques de un LLM congelado con nuevas capas de cross-attention. Hasta **80B parámetros**; excelente en few-shot.
- **OmniVL** (Wang et al., 2022): Dos decoders: uno para alineación/comprensión y otro para generación. Maneja imagen y video.
- **CoCa** (Yu et al., 2022): Image encoder + decoder unimodal + decoder multimodal.
- **OFA** (Wang et al., 2022) y **DaVinci** (Diao et al., 2022): El decoder puede generar tanto texto *como* imagen — también realizan tareas unimodales (GLUE, ImageNet).
- **mDETR / DQ-DETR** (Kamath et al., 2021 / Liu et al., 2022): Basados en DETR para visual grounding. CNN (imagen) + transformer encoder (texto) → encoder conjunto → decoder con *learned query embeddings* → FFN.

---

## 6. Tareas de Preentrenamiento

El preentrenamiento es el factor más determinante del rendimiento.

### 6.1 Masked Language Modeling (MLM)

*Usado por:* todos los fusion/combination encoders; algunos encoder-decoder (VL-T5).

Extensión del MLM de BERT con acceso a tokens visuales $\mathbf{v}$:

$$\mathcal{L}_{MLM}(\theta) = -\mathbb{E}_{(t,v) \sim D} \log P_\theta(t_m \mid \mathbf{t}_{\backslash m}, \mathbf{v})$$

Variaciones:
- **ViLT:** Enmascara palabras completas (no subwords).
- **UNIMO:** Enmascara spans contiguos (estilo SpanBERT).
- **BEiT-3:** Enmascara 40% de tokens (vs. 15% estándar).

### 6.2 Masked Image Modeling (MIM)

*Usado por:* ViLBERT, UNITER, VL-BERT, LXMERT, Unicoder-VL, BEiT-3, UNIMO, SOHO, FLAVA.

15% de regiones se enmascaran (puestas a 0). El modelo predice la distribución de categorías; minimiza KL divergence con la distribución del R-CNN original.

- **SOHO** (grid features): Usa índice del diccionario visual como etiqueta.
- **BEiT-3 / FLAVA** (patch embeddings): Usa dVAE (discrete Variational Autoencoder) para asignar códigos visuales discretos a cada parche.
- **UNITER, LXMERT:** Extienden con *region-feature regression* (predecir valores exactos con MSE).

### 6.3 Image-Text Matching (ITM)

*Usado por:* casi todos los fusion/combination encoders; OFA, mPLUG.

Clasificación binaria: dado (imagen, texto), ¿corresponden?

$$\mathcal{L}_{ITM}(\theta) = -\mathbb{E}_{(t,v) \sim D}\left[y \log S_\theta(\mathbf{t},\mathbf{v}) + (1-y)\log(1 - S_\theta(\mathbf{t},\mathbf{v}))\right]$$

Los pares negativos se crean sustituyendo imagen o texto por uno aleatorio del training set.

> **Pregunta abierta:** Por analogía con el *next sentence prediction* (NSP) de BERT, que RoBERTa demostró ser prescindible, ¿contribuye realmente el ITM a la alineación visión-lenguaje?

### 6.4 Contrastive Learning

*Usado por:* CLIP, ALIGN, LiT, Florence, ALBEF, VLMo, X2-VLM, FLAVA, CoCa, mPLUG, Flamingo.

Dado un batch de $N$ pares codificados (imagen, texto), se maximiza la similitud de los $N$ pares correctos y se minimiza la de los $N^2-N$ incorrectos. Para valores normalizados, la similitud coseno reduce a producto punto. Se optimiza con **binary cross-entropy**.

- **Florence:** Unified image-text contrastive learning (Yang et al., 2022) — un texto puede asociarse a múltiples imágenes.
- **UNIMO:** Aumentado con text rewriting y recuperación imagen/texto para crear grandes volúmenes de pares.

### 6.5 Visual Question Answering como Tarea Proxy

*Usado por:* LXMERT, OFA, VL-T5.

- LXMERT: Trata VQA como clasificación.
- OFA, VL-T5: Generan la respuesta con el decoder.

### 6.6 Visual Grounding

*Usado por:* OFA, X2-VLM, VL-T5, todos los grounded transformers.

Referential resolution — predice $\langle x_1, y_1, x_2, y_2 \rangle$ del bounding box. Grounded captioning da la operación inversa.

**Limitación:** Requiere anotaciones costosas; solo ~1.3M pares disponibles en RefCOCO, Flickr30K Entities, etc.

### 6.7 Image Captioning como Tarea Proxy

*Usado por:* OFA, E2E-VLP, CoCa.

LM causal condicionado en la imagen:

$$\mathcal{L}_{cap} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}, x)$$

### 6.8 Prefix Language Modeling

*Usado por:* DaVinci, mPLUG, SimVLM.

Se trunca la secuencia a longitud aleatoria $T_p$. Al **prefijo** se le aplica atención bidireccional; al **sufijo** se le aplica LM autoregresivo. El prefijo siempre contiene todos los tokens de imagen.

**DaVinci** extiende esto a **prefix image modeling**: prefijo = texto completo + secuencia parcial de imagen; el modelo restaura los tokens de imagen del sufijo.

### 6.9 Otros Objetivos

- **PixelBERT:** Muestreo aleatorio de pixel features como regularización.
- **VLMo:** Preentrenamiento **por etapas** — (1) entrena vision expert con BEiT MIM; (2) entrena language expert con MLM de texto; (3) entrena modelo completo con ITM.
- **UNIMO:** Tareas unimodales (solo lenguaje o solo visión) sin congelar partes del modelo.

---

## 7. Capacidades Downstream

### 7.1 Alineación VL

Retrieval cruzado imagen↔texto. Benchmarks: **MSCOCO** (Lin et al., 2014), **Flickr30K** (Plummer et al., 2015). Los dual encoders son ideales: sus embeddings de imagen pueden cachearse para búsqueda eficiente a gran escala.

### 7.2 Comprensión VL

Benchmarks clave:
- **VQA** (Antol et al., 2015)
- **NLVR2** (Suhr et al., 2018): Verdadero/Falso sobre imágenes

Los fusion encoders son los especialistas. Los encoder-decoder también rinden bien reencuadrando como generación. Los dual encoders (excepto Florence) tienen rendimiento pobre.

### 7.3 Generación de Texto VL

Benchmarks: **MSCOCO Captions**, **NoCaps** (Agrawal et al., 2019). Los encoder-decoder son los más naturales. Algunos fusion encoders (BEiT-3, Oscar, VinVL, UNIMO) también rinden bien en captioning.

### 7.4 Visual Grounding, Grounded Captioning y Object Detection

Benchmark: **RefCOCO** (Kazemzadeh et al., 2014). Los grounded transformers (mDETR, DQ-DETR, UniTAB, Referring Transformer, KD-VLP) están diseñados específicamente para esto.

### 7.5 Generación de Imagen

Solo **OFA** y **DaVinci** pueden generar imágenes, completando el ciclo de verdadera multimodalidad.

### 7.6 Tareas de Video

CoCa, Florence, GIT, OmniVL, X2-VLM entrenados y evaluados en retrieval y comprensión de video.

### 7.7 Tareas Unimodales

Clasificación pura de imágenes: CLIP, ALIGN, BLIP-2, CoCa, Flamingo, GIT, OFA, OmniVL, PaLI.  
Benchmarks NLP puros (GLUE): solo **UNIMO** y **OFA**.

---

## 8. Datos de Preentrenamiento

### 8.1 Fuentes Públicas

#### Anotados por Humanos

| Dataset | Tamaño | Descripción |
|---|---|---|
| **MSCOCO** (Lin et al., 2014) | 328k imágenes; ~567k–1M pares | Objetos en contexto; anotados en Amazon MTurk. Base de casi todos los modelos. |
| **Visual Genome** (Krishna et al., 2017) | 108k imágenes; 5M+ pares | Descripciones de regiones, objetos segmentados, scene graphs. Esencial para grounding. |

#### Obtenidos de la Web

| Dataset | Tamaño | Descripción |
|---|---|---|
| **SBU Captions** (Ordonez et al., 2011) | ~1M pares | Imágenes de Flickr + texto de matching. |
| **CC3M** (Sharma et al., 2018) | 3M pares | Web crawl con alt-texts limpiados. |
| **CC12M** (Changpinyo et al., 2021) | 12M pares | CC3M con criterios de filtrado relajados. |
| **LAION-400M** (Schuhmann et al., 2021) | 400M pares | Alt-texts de la web filtrados con CLIP. |

### 8.2 Datasets Propietarios

| Modelo | Tamaño | Nota |
|---|---|---|
| **ALIGN** | 1.8B pares | Filtrado mínimo — ruido alto, cantidad supera al ruido. |
| **CLIP** | 400M pares | Fuentes diversas de internet; detalles no publicados. |
| **Flamingo** | 312M pares LTIP + 27M video (+ ALIGN) | DeepMind/Google. |
| **LiT** | 4B pares | Proceso similar a ALIGN con filtrado aún más relajado. |

### 8.3 Escala de Datos

| Categoría | Rango | Modelos ejemplo |
|---|---|---|
| **Pequeño** | < 10M pares | BridgeTower, METER, UNITER, VILLA, ViLT; E2E-VLP, KD-VLP, PixelBERT |
| **Mediano** | 10–25M pares | ALBEF, BEiT-3, mPLUG, OFA, OmniVL |
| **Grande** | > 25M pares | CLIP, ALIGN, LiT, Florence, GIT, PaLI, Flamingo, CoCa, DaVinci |

---

## 9. Análisis: Fortalezas, Limitaciones y Futuro

### 9.1 Fortalezas

**Representaciones Generalizadas:**  
Antes se necesitaban modelos ad hoc por tarea. Los VL transformers se transfieren a nuevas tareas con cambios mínimos y fine-tuning accesible. El costo de preentrenamiento se incurre una sola vez.

**Facilidad de Uso:**  
Los transformers son ubicuos — la mayoría de los practicantes tienen familiaridad. El fine-tuning consiste típicamente en reemplazar y reentrenar solo la capa final. Frameworks como **LAVIS** (Li et al., 2022) simplifican el deployment.

**Rendimiento:**  
Los leaderboards de VQA, RefCOCO, etc. están dominados por transformers. Han impulsado mejoras masivas igual que en NLP.

### 9.2 Limitaciones y Preguntas Abiertas

#### 9.2.1 Costo de Datos y Cómputo

Los modelos van de 83M parámetros (UNITER-Base) a 80B (Flamingo). Bugliarello et al. (2021) estimaron que entrenar un VL transformer 10 veces para 4 tareas requirió una máquina de **4 GPUs en AWS por dos meses (~$6,000 en 2021)**. GIT-2 usa 12.9B pares imagen-texto.

Los datos VL son difíciles de producir: el texto debe ser relevante a la imagen. Los datos anotados son caros y escasos; los datos web son muy ruidosos.

#### 9.2.2 Tareas de Preentrenamiento

¿Aprende algo visual el MLM cuando BERT ya es muy bueno sin imagen? El ITM es análogo al NSP de BERT, que RoBERTa demostró prescindible. No existe evidencia empírica sistemática aún. Los modelos con VQA y visual grounding como proxy parecen prometedores, pero requieren datasets costosos.

#### 9.2.3 Embeddings Visuales: Comparativa

| Tipo | Ventajas | Desventajas |
|---|---|---|
| **Region features** | Semánticamente ricos; etiquetas explícitas | Techo de vocabulario; cuello de botella computacional |
| **Grid features** | Sin techo; representación densa | Aún requiere CNN; sigue siendo costoso |
| **Patch embeddings** | Mínimo cómputo; sin CNN separada | ¿Representaciones inferiores? Debate activo |

**No existe un meta-análisis controlado que responda definitivamente qué estrategia es superior y bajo qué condiciones.**

### 9.3 Direcciones Futuras

#### 9.3.1 Generación de Datos y Meta-Análisis

Necesidad urgente de más datasets de preentrenamiento públicos de calidad, más benchmarks diversos, y un meta-análisis amplio controlando: cantidad de datos, hiperparámetros, múltiples repeticiones de entrenamiento, evaluación en amplio rango de tareas. El ritmo de creación de modelos ha superado al suministro de datos y al conocimiento sobre cómo rinden.

#### 9.3.2 Tareas de Preentrenamiento Alternativas

Explorar sistemáticamente nuevas tareas que creen interacciones más explícitas y profundas entre visión y lenguaje. Ejemplo prometedor: **position-guided text prompting** (Wang et al., 2023) — divide la imagen en parches, identifica objetos y crea tareas fill-in-the-blank; compatible teóricamente con cualquier arquitectura VL.

#### 9.3.3 Modalidades Adicionales

- **ONE-PEACE** (Wang et al., 2023) y **VALOR** (Chen et al., 2023): Visión + lenguaje + **audio**.
- **PaLM-E** (Driess et al., 2023): Visión + lenguaje + **control robótico** — modelo VL encarnado.

Estas extensiones se acercan a resolver el **symbol grounding problem** (Harnad, 1990): cómo los símbolos lingüísticos adquieren significado a través de su relación con el mundo perceptual.

---

## 10. Mapa de Modelos Revisados

| Modelo | Arquitectura | Embedding Visual | Escala Datos |
|---|---|---|---|
| ALBEF | Combo Encoder | Patch | Mediano |
| ALIGN | Dual Encoder | Grid | Grande (1.8B) |
| BEiT-3 | Combo Encoder | Patch | Mediano |
| BLIP-2 | Enc-Dec | Patch | Grande |
| BridgeTower | Two-Tower | Patch | Pequeño |
| CLIP | Dual Encoder | Grid | Grande (400M) |
| CoCa | Enc-Dec | Patch | Grande |
| DaVinci | Enc-Dec | Patch | Grande |
| DQ-DETR | Enc-Dec | Grid | Pequeño |
| E2E-VLP | Enc-Dec | Grid | Pequeño |
| Flamingo | Enc-Dec | Grid | Grande (80B params) |
| FLAVA | Combo Encoder | Patch | Grande |
| Florence | Dual Encoder | Grid | Grande |
| GIT | Enc-Dec | Grid | Grande |
| GPV | Enc-Dec | Grid | Pequeño |
| KD-VLP | One-Tower | Grid | Pequeño |
| LEMON | Enc-Dec | Region | Mediano |
| LiT | Dual Encoder | Grid | Grande (4B) |
| LXMERT | Two-Tower | Region | Pequeño |
| mDETR | Enc-Dec | Grid | Pequeño |
| METER | Two-Tower | Patch | Pequeño |
| mPLUG | Enc-Dec | Patch | Mediano |
| OFA | Enc-Dec | Patch | Mediano+ |
| OmniVL | Enc-Dec | Patch | Mediano |
| Oscar | One-Tower | Region | Pequeño |
| PaLI | Enc-Dec | Patch | Grande |
| PixelBERT | One-Tower | Grid | Pequeño |
| Referring Transformer | Enc-Dec | Grid | Pequeño |
| SimVLM | Enc-Dec | Patch | Grande |
| SOHO | One-Tower | Grid | Pequeño |
| Unicoder-VL | One-Tower | Region | Pequeño |
| UNIMO | One-Tower | Region | Pequeño |
| UniTAB | Enc-Dec | Grid | Pequeño |
| UNITER | One-Tower | Region | Pequeño |
| ViLBERT | Two-Tower | Region | Pequeño |
| VILLA | One-Tower | Region | Pequeño |
| ViLT | One-Tower | Patch | Pequeño |
| VinVL | One-Tower | Region | Pequeño |
| VisualBERT | One-Tower | Region | Pequeño |
| VL-BERT | One-Tower | Region | Pequeño |
| VLMo | Combo Encoder | Patch | Mediano |
| VL-T5 | Enc-Dec | Region | Pequeño |
| X2-VLM | Combo Encoder | Patch | Mediano |

---

## 11. Referencias Clave

### Arquitectura Base
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.

### Fundamentos NLP
- Devlin, J. et al. (2018). BERT. *arXiv:1810.04805*.
- Radford, A. et al. (2018). GPT. OpenAI.
- Liu, Y. et al. (2019). RoBERTa. *arXiv:1907.11692*.
- Sanh, V. et al. (2019). DistilBERT. *arXiv:1910.01108*.
- OpenAI (2023). GPT-4 Technical Report. *arXiv:2303.08774*.

### Fundamentos CV
- Dosovitskiy, A. et al. (2020). ViT. *arXiv:2010.11929*.
- Bao, H. et al. (2021). BEiT. *arXiv:2106.08254*.
- Liu, Z. et al. (2021). Swin Transformer. *ICCV 2021*.
- Dong, X. et al. (2021). CSWin Transformer. *arXiv:2107.00652*.

### Dual Encoders
- Radford, A. et al. (2021). CLIP. *ICML 2021*.
- Jia, C. et al. (2021). ALIGN. *ICML 2021*.
- Zhai, X. et al. (2022). LiT. *CVPR 2022*.
- Yuan, L. et al. (2021). Florence. *arXiv:2111.11432*.

### Fusion Encoders
- Lu, J. et al. (2019). ViLBERT. *NeurIPS 2019*.
- Chen, Y.-C. et al. (2019). UNITER. *arXiv*.
- Kim, W. et al. (2021). ViLT. *ICML 2021*.
- Dou, Z.-Y. et al. (2022). METER. *CVPR 2022*.
- Xu, X. et al. (2022). BridgeTower. *arXiv:2206.08657*.

### Combination Encoders
- Bao, H. et al. (2022). VLMo. *NeurIPS 2022*.
- Li, J. et al. (2021). ALBEF. *NeurIPS 2021*.
- Wang, W. et al. (2023). BEiT-3. *CVPR 2023*.
- Singh, A. et al. (2022). FLAVA. *CVPR 2022*.
- Zeng, Y. et al. (2022). X2-VLM. *arXiv:2211.12402*.

### Encoder-Decoder
- Cho, J. et al. (2021). VL-T5. *ICML 2021*.
- Alayrac, J.-B. et al. (2022). Flamingo. *arXiv:2204.14198*.
- Wang, P. et al. (2022). OFA. *arXiv:2202.03052*.
- Li, J. et al. (2023). BLIP-2. *arXiv:2301.12597*.
- Yu, J. et al. (2022). CoCa. *arXiv:2205.01917*.
- Diao, S. et al. (2022). DaVinci. *arXiv:2206.07699*.

### Grounded Transformers
- Kamath, A. et al. (2021). mDETR. *ICCV 2021*.
- Liu, S. et al. (2022). DQ-DETR. *arXiv:2211.15516*.
- Li, M. & Sigal, L. (2021). Referring Transformer. *arXiv:2106.03089*.
- Yang, Z. et al. (2022). UniTAB. *ECCV 2022*.

### Meta-Análisis y Frameworks
- Bugliarello, E. et al. (2021). Multimodal Pretraining Unmasked. *TACL*.
- Li, D. et al. (2022). LAVIS. *arXiv:2209.09019*.

### Datasets
- Lin, T.-Y. et al. (2014). MSCOCO. *ECCV 2014*.
- Krishna, R. et al. (2017). Visual Genome. *IJCV*.
- Ordonez, V. et al. (2011). SBU Captions. *NeurIPS 2011*.
- Sharma, P. et al. (2018). CC3M. *ACL 2018*.
- Changpinyo, S. et al. (2021). CC12M. *CVPR 2021*.
- Schuhmann, C. et al. (2021). LAION-400M. *arXiv:2111.02114*.

### Benchmarks
- Antol, S. et al. (2015). VQA. *ICCV 2015*.
- Suhr, A. et al. (2018). NLVR2. *arXiv:1811.00491*.
- Kazemzadeh, S. et al. (2014). RefCOCO. *EMNLP 2014*.
- Plummer, B. et al. (2015). Flickr30K Entities. *ICCV 2015*.

### Modelos Multimodales Extendidos
- Driess, D. et al. (2023). PaLM-E. *arXiv:2303.03378*.
- Wang, P. et al. (2023). ONE-PEACE. *arXiv:2305.11172*.
- Chen, S. et al. (2023). VALOR. *arXiv:2304.08345*.

### Marco Conceptual
- Harnad, S. (1990). The symbol grounding problem. *Physica D, 42(1):335–346*.

---

## 12. Síntesis Final

Los Vision Language Transformers representan la arquitectura dominante en la intersección de visión y lenguaje. Su éxito descansa en cinco pilares:

1. **El poder del preentrenamiento masivo** — representaciones generales transferibles.
2. **La flexibilidad del mecanismo de atención** — permite interacciones arbitrarias entre tokens visuales y textuales.
3. **La unificación de modalidades** — con la estrategia correcta de embedding, imágenes y texto son secuencias comparables.
4. **La escalabilidad** — más datos y más parámetros generalmente mejoran los resultados.
5. **La versatilidad arquitectónica** — del dual encoder al complejo encoder-decoder, cada diseño ofrece trade-offs distintos.

Las preguntas abiertas más urgentes al momento del paper (julio 2023) son: (i) qué estrategia de embedding visual es óptima, (ii) si las tareas de preentrenamiento actuales realmente fuerzan la alineación visión-lenguaje o solo imitan el NLP, y (iii) cómo escalar la creación de datos de calidad. La tendencia hacia modelos más grandes, multimodales, con LLMs como backbone (Flamingo, BLIP-2, PaLM-E) marcaba ya la dirección hacia lo que hoy conocemos como **Multimodal Large Language Models (MLLMs)**.

---

*Revisión elaborada sobre: Fields, C. & Kennington, C. (2023). Vision Language Transformers: A Survey. arXiv:2307.03254v1. Julio 2023.*
