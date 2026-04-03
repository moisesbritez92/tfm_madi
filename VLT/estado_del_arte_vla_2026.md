# Estado del arte de los modelos Vision-Language-Action (VLA)

**Fecha de corte:** 3 de abril de 2026  

**Propósito:** documento de síntesis técnica sobre evolución, arquitectura, benchmarks, modelos punteros y retos abiertos de los VLA.

## Resumen ejecutivo

- Los VLA surgieron como una extensión de los VLM hacia el control físico: integran percepción visual, comprensión lingüística y generación de acciones dentro de una misma política o de un sistema acoplado.

- Entre 2023 y 2024 el campo pasó de pruebas de concepto semánticas (PaLM-E, RT-2) a políticas abiertas y reutilizables (Octo, OpenVLA) y a nuevas familias de decodificación de acciones basadas en difusión o flow matching (π0, RDT-1B, Diffusion-VLA).

- En 2025 el estado del arte se movió en tres direcciones simultáneas: mayor generalización en entornos abiertos (π0.5), mejor adaptación y velocidad de inferencia (OpenVLA-OFT) y expansión hacia robots humanoides (GR00T N1, Helix).

- En 2026 el cuello de botella ya no es solo la precisión: la robustez lingüística, la dificultad realista de los benchmarks y la inferencia a frecuencia útil para el robot se han convertido en problemas centrales.

- La frontera actual combina co-entrenamiento heterogéneo, datos multi-robot, video humano, simulación y arquitecturas duales o asíncronas para equilibrar generalización, destreza y latencia.


## 1. Introducción

Los modelos Vision-Language-Action (VLA) representan una de las líneas más activas de la robótica contemporánea. Su objetivo es que una misma arquitectura pueda observar el entorno, entender instrucciones en lenguaje natural y producir acciones continuas o discretizadas para controlar un robot. En comparación con pipelines clásicos -percepción, planeamiento y control por separado- los VLA intentan aprender una representación unificada del bucle percepción-razonamiento-acción [3][6][15].

Aunque el término VLA se consolidó con RT-2 en 2023, la genealogía del campo es más amplia. Gato propuso la idea de un agente generalista multimodal capaz de emitir texto, torques y otras salidas desde una misma red [1]. PaLM-E mostró que un modelo multimodal grande podía incorporar sensores continuos y observaciones visuales dentro de una formulación de lenguaje encarnado [2]. RT-2 dio el salto decisivo al expresar acciones robóticas como tokens y coajustar un VLM con datos web y trayectorias de robot, abriendo la puerta a capacidades semánticas emergentes en control robótico [3].

Desde 2024, el campo dejó de ser una curiosidad de laboratorio y pasó a organizarse como ecosistema: datasets abiertos multi-robot, políticas generalistas, recetas de fine-tuning, benchmarks más diagnósticos y primeras propuestas industriales de despliegue. Por eso, hablar hoy del estado del arte de los VLA implica revisar tanto la investigación académica revisada por pares como las propuestas industriales que marcan la frontera práctica.


## 2. ¿Qué es un modelo VLA?

Un VLA puede definirse como una política fundacional encarnada que mapea observaciones visuales y contexto lingüístico a acciones robóticas. En forma compacta:

π(a_t | o_{1:t}, l, h) 

donde o_{1:t} son observaciones visuales o multimodales, l es la instrucción en lenguaje natural y h resume historia, propriocepción o memoria. En la práctica, el campo usa dos grandes diseños:

1. **Monolíticos/autoregresivos.** Un único backbone procesa visión y lenguaje y genera acciones token a token o chunk a chunk. Esta familia incluye RT-2 y OpenVLA [3][6].
2. **Híbridos/difusión-flow/dual-system.** Un VLM o backbone multimodal produce representaciones de alto nivel y un experto de acciones -por difusión, flow matching o sistema rápido separado- produce el control motor. Esta familia incluye π0, RDT-1B, GR00T N1 y Helix [7][8][11][12].

La promesa central de los VLA es la **generalización abierta**: aprender conocimiento semántico a gran escala y reutilizarlo para tareas, objetos, instrucciones y configuraciones robóticas no vistas durante el entrenamiento [3][7][13].


## 3. Evolución del campo

### 2022-2023: precursores y definición del paradigma

- **Gato (2022)** mostró que un mismo transformador podía operar sobre múltiples modalidades y tareas, incluyendo control robótico, bajo un paradigma de agente generalista [1].
- **PaLM-E (2023)** llevó esta idea hacia modelos encarnados multimodales, integrando entradas visuales y de estado continuo en un LLM y demostrando transferencia positiva entre dominios web y tareas de embodied reasoning [2].
- **RT-2 (2023)** acuñó de facto el enfoque VLA moderno: acciones expresadas como tokens, co-fine-tuning con datos web y de robot, y capacidades emergentes derivadas del preentrenamiento multimodal a escala [3].

### 2023-2024: la base de datos y la apertura del ecosistema

- **Open X-Embodiment** consolidó un dataset multi-institucional con 22 robots, 21 instituciones y 527 habilidades, junto con modelos RT-X para estudiar transferencia entre robots [4].
- **Octo (2024)** fue una política generalista abierta entrenada con 800k trayectorias de Open X-Embodiment; aunque suele clasificarse como política generalista más que como VLA pleno, fue un puente crucial hacia la reutilización abierta y la adaptación rápida [5].
- **OpenVLA (2024)** se convirtió en un punto de inflexión: modelo abierto de 7B, entrenado sobre 970k episodios reales, con fuerte grounding lingüístico y resultados superiores a modelos cerrados como RT-2-X en manipulación generalista [6].

### 2024-2025: nueva generación de arquitecturas de acción

- **π0 (2024)** introdujo una familia basada en flow matching sobre un backbone VLM, dirigida explícitamente a control general de robots con tareas dextras y multi-robot [7].
- **RDT-1B (2024/2025)** extendió el uso de difusión a manipulación bimanual, proponiendo un espacio de acción unificado físicamente interpretable y 1.2B parámetros [8].
- **Diffusion-VLA (2024/2025)** combinó razonamiento autoregresivo y generación de acciones por difusión, añadiendo interpretabilidad mediante self-generated reasoning [9].

### 2025-2026: eficiencia, generalización abierta y humanoides

- **OpenVLA-OFT (2025)** mostró que una buena receta de adaptación puede ser tan importante como el modelo base: elevó OpenVLA de 76.5% a 97.1% en LIBERO y multiplicó por 26x el throughput de generación de acciones [10].
- **GR00T N1 (2025)** llevó el paradigma VLA a humanoides con una arquitectura dual (System 2 semántico + System 1 motor) y mezcla de trayectorias reales, videos humanos y datos sintéticos [11].
- **Helix (2025)** trasladó la narrativa del VLA a control de la parte superior completa del cuerpo de un humanoide y colaboración multi-robot; sin embargo, sus resultados pertenecen al frente industrial y no a la literatura revisada por pares [12].
- **π0.5 (2025)** empujó el argumento de generalización abierta: co-entrenamiento con datos heterogéneos, predicción semántica de subtareas y demostraciones en hogares no vistos [13].
- **SmolVLA (2025)** puso el foco en accesibilidad y costo: entrenamiento en una sola GPU y despliegue en hardware de consumo con rendimiento competitivo frente a modelos ~10x mayores [14].
- **VLA-Perf (2026)** no propone una nueva política, pero sí una nueva disciplina: diseñar VLA bajo restricciones explícitas de latencia, throughput y sitio de inferencia [16].
- **Helix 02 (2026)** refuerza la dirección industrial hacia control full-body desde píxeles, pero sigue siendo evidencia de frontera práctica más que académica consolidada [22].


## 4. Taxonomía técnica actual

El estado del arte no está dominado por una sola arquitectura. Más bien, se observan cuatro ejes de diseño.

### 1. Backbone perceptivo-semántico
Los VLA modernos heredan gran parte de su inteligencia semántica de VLM o LLM multimodales preentrenados. OpenVLA combina Llama 2 con DINOv2 y SigLIP [6]. π0 parte de un VLM preentrenado [7]. GR00T N1 separa explícitamente el componente visión-lenguaje de la generación motora [11].

### 2. Decodificación de acciones
Aquí aparecen hoy las familias más importantes:
- **Autoregresiva**: genera acciones como tokens o chunks en secuencia. Ventaja: integración simple con VLM/LLM y fuerte alineación semántica. Desventaja: menor naturalidad para acciones continuas de alta frecuencia [3][6][16].
- **Difusión / flow matching**: modela mejor distribuciones multimodales y trayectorias continuas. Ventaja: más estabilidad en tareas motoras complejas; desventaja: costo de inferencia más alto por pasos iterativos [7][8][9][16].
- **Dual-system**: desacopla razonamiento lento y control rápido. Ventaja: mejor compromiso entre generalidad y frecuencia de control; desventaja: mayor complejidad de entrenamiento y sincronización [11][12][16].

### 3. Estrategia de datos
El consenso actual es que no basta con trayectorias teleoperadas de un solo robot. Los sistemas más competitivos combinan:
- múltiples robots y embodiments [4][13],
- demostraciones reales a escala [4][19],
- video humano o egocéntrico [11],
- datos sintéticos o simulación [11][16],
- etiquetas semánticas intermedias como subtareas u objetos [13].

### 4. Estrategia de despliegue
El diseño del modelo ya no se separa del diseño del sistema. Los trabajos más recientes incorporan **action chunking**, inferencia asíncrona, reducción del número de pasos de difusión, paralelización del decodificado y decisiones explícitas sobre si la inferencia ocurre onboard, en edge o en cloud [10][14][16].


## 5. Datasets y benchmarks

La calidad del estado del arte en VLA depende tanto del modelo como del ecosistema de datos y evaluación.

### Datasets clave
- **Open X-Embodiment** es la infraestructura de datos más influyente para políticas generalistas abiertas: 22 robots, 21 instituciones y 527 habilidades en formato estandarizado [4].
- **DROID** aportó diversidad in-the-wild: 76k trayectorias, 350 horas, 564 escenas y 84 tareas recolectadas por 50 operadores en varios continentes [19].

### Benchmarks más usados
- **CALVIN** sigue siendo una referencia para manipulación de largo horizonte condicionada por lenguaje [18].
- **LIBERO** es uno de los benchmarks más citados para transferencia y evaluación de políticas generalistas; OpenVLA-OFT lo usa como punto de referencia para medir mejoras de adaptación [10][17].
- En bimanualidad y plataformas reales, ALOHA/Mobile ALOHA y configuraciones afines se usan de forma creciente para validar destreza de alto nivel, aunque con menor estandarización universal que CALVIN o LIBERO [10].

### Nuevo giro en evaluación: robustez real
Una de las conclusiones más importantes de 2026 es que varios benchmarks clásicos se están quedando “demasiado fáciles” o demasiado cercanos a la distribución de entrenamiento. **LIBERO-X** argumenta que LIBERO puede saturarse bajo gaps de entrenamiento-prueba reducidos y propone una jerarquía de perturbaciones más duras para evaluar generalización espacial, reconocimiento de objetos e interpretación de instrucciones [20]. **LIBERO-Para** añade una dimensión lingüística crítica: observa caídas de 22-52 puntos porcentuales cuando las instrucciones son parafraseadas, lo que revela dependencia de coincidencias superficiales más que grounding semántico robusto [21].


## 6. Modelos punteros a abril de 2026

A abril de 2026, la frontera técnica puede resumirse así:

1. **OpenVLA / OpenVLA-OFT**  
   Sigue siendo la referencia abierta más importante para manipulación generalista y adaptación eficiente. OpenVLA demostró que un modelo abierto de 7B podía superar a RT-2-X con muchos menos parámetros [6]. OpenVLA-OFT convirtió esa base en una receta práctica de fine-tuning, mejorando tanto rendimiento como latencia [10].

2. **π0 / π0.5**  
   Representan una de las apuestas más fuertes por VLA basados en flow matching y co-entrenamiento heterogéneo. π0 demostró control general y destreza en tareas variadas [7]; π0.5 fue más allá hacia generalización abierta en hogares nuevos y tareas de limpieza o reordenamiento de largo horizonte [13].

3. **RDT-1B y Diffusion-VLA**  
   Son muy relevantes cuando la precisión motora y la multimodalidad de acciones importan más que la simplicidad del pipeline. RDT-1B sobresale en manipulación bimanual y aprendizaje con muy pocas demostraciones [8]. Diffusion-VLA añade un componente de razonamiento explícito e interpretabilidad poco frecuente en políticas robóticas [9].

4. **GR00T N1 y Helix / Helix 02**  
   Definen la frontera humanoide. GR00T N1 es la propuesta más clara en formato académico abierto para humanoides generalistas [11]. Helix y Helix 02 son quizás las demostraciones industriales más ambiciosas en control whole-body o upper-body desde lenguaje natural y píxeles, pero conviene distinguir entre promesa industrial y evidencia reproducible [12][22].

5. **SmolVLA**  
   Es importante porque cambia la economía del campo. Demuestra que el futuro del estado del arte no solo consiste en escalar, sino también en miniaturizar con criterio para abaratar entrenamiento y despliegue sin perder demasiado rendimiento [14].


### Tabla comparativa


| Modelo | Año | Paradigma | Apertura | Aporte clave |
|---|---:|---|---|---|
| Gato | 2022 | Autoregresivo generalista | No robótico específico / no abierto como producto VLA | Primer agente multimodal y multi-embodiment generalista |
| PaLM-E | 2023 | LLM multimodal encarnado | Cerrado | Grounding multimodal y transferencia positiva hacia tareas embodied |
| RT-2 | 2023 | VLM + acciones tokenizadas | Cerrado | Formaliza el paradigma VLA moderno |
| OpenVLA | 2024 | Autoregresivo 7B | Abierto | Referencia abierta en manipulación generalista |
| π0 | 2024 | Flow matching | Parcialmente abierto | Control general y destreza con backbone VLM |
| RDT-1B | 2024/25 | Difusión | Abierto | Fuerte en bimanualidad y pocas demostraciones |
| Diffusion-VLA | 2024/25 | AR + difusión | Abierto | Razona y actúa con interpretabilidad |
| OpenVLA-OFT | 2025 | Receta de adaptación | Abierto | SOTA en LIBERO y gran mejora de velocidad |
| GR00T N1 | 2025 | Dual-system | Abierto | Humanoides generalistas con datos reales, humanos y sintéticos |
| π0.5 | 2025 | Flow + co-training heterogéneo | Parcialmente abierto | Generalización abierta en hogares nuevos |
| SmolVLA | 2025 | Compacto + asíncrono | Abierto | Reduce drásticamente el costo de entrenamiento e inferencia |
| Helix / Helix 02 | 2025/26 | Dual-system / full-body | Industrial | Frontera práctica en humanoides; evidencia no revisada por pares |


## 7. Tendencias del estado del arte

### Tendencia 1: del “más grande” al “mejor compuesto”
El campo se está desplazando de la simple escala paramétrica hacia composiciones más inteligentes: VLM + experto de acciones, dual-system, action chunking e inferencia asíncrona [10][11][16].

### Tendencia 2: más heterogeneidad de datos
La mezcla de robots, hogares, escenas, video humano y etiquetas semánticas parece ser una condición casi obligatoria para generalización abierta [11][13][19].

### Tendencia 3: evaluación más diagnóstica
Ya no alcanza con reportar success rate promedio. Se buscan benchmarks que separen robustez espacial, robustez lingüística, objetos no vistos, perturbaciones compuestas y calidad de ejecución [20][21].

### Tendencia 4: VLA para humanoides
El auge reciente de GR00T N1, Helix y Helix 02 muestra que la comunidad empieza a tratar a los humanoides como plataforma natural para evaluar inteligencia encarnada generalista [11][12][22].

### Tendencia 5: inferencia como restricción de primer orden
VLA-Perf formaliza que la pregunta correcta ya no es solo “qué arquitectura generaliza mejor”, sino “qué arquitectura puede ejecutarse a frecuencia útil, con el hardware disponible y dentro del presupuesto de energía y red” [16].


## 8. Limitaciones actuales

Pese al progreso, el estado del arte dista de estar resuelto.

- **Robustez lingüística insuficiente.** Cambios menores en la redacción pueden romper la ejecución; LIBERO-Para documenta degradaciones muy severas bajo parafraseo [21].
- **Generalización todavía frágil.** Aunque modelos como π0.5 y Helix muestran avances, la transferencia a hogares, objetos y tareas realmente abiertas sigue siendo limitada y costosa [12][13][20].
- **Latencia y costo de inferencia.** Los modelos más capaces no siempre son compatibles con la frecuencia de control que exigen tareas físicas rápidas; esto es especialmente crítico para difusión y contextos largos [16].
- **Dependencia de datos caros.** Recolectar demostraciones reales, multi-robot y de alta calidad sigue siendo uno de los principales costos del campo [4][19].
- **Reproducibilidad desigual.** El área mezcla trabajos abiertos, semiabiertos y anuncios industriales. Por ello, conviene distinguir cuidadosamente entre “estado del arte reproducible” y “frontera demostrativa no revisada por pares”.


## 9. Agenda de investigación

Para investigación y desarrollo, las oportunidades más prometedoras parecen ser las siguientes:

1. **VLA con memoria y estado explícito.** Muchos fallos no son perceptivos sino de consistencia a lo largo de horizontes largos.
2. **Evaluación multimétrica.** Además de success rate, hace falta medir elegancia, seguridad, eficiencia temporal, robustez semántica y recuperación tras error.
3. **Transferencia humano-a-robot.** GR00T N1 y trabajos asociados apuntan a que el video humano puede ser un multiplicador clave de datos [11].
4. **Modelos pequeños pero competentes.** SmolVLA sugiere que habrá una línea fuerte de VLA compactos para laboratorios y empresas sin infraestructura masiva [14].
5. **VLA en dominio específico.** En industria, energía, logística, salud o agricultura, probablemente el mejor rendimiento vendrá de VLA fundacionales especializados, no totalmente generalistas [15].
6. **Sistemas híbridos con planeamiento explícito.** Una dirección probable es combinar VLA con módulos simbólicos, world models o verificación de seguridad para tareas críticas.


## 10. Conclusión

El estado del arte de los VLA, a abril de 2026, puede describirse como un campo que ya superó la etapa de demostración inicial y entró en una fase de ingeniería científica intensiva. La pregunta dejó de ser si un VLA puede funcionar; ahora la discusión gira en torno a qué combinación de arquitectura, datos, benchmark y sistema de inferencia produce una política realmente útil fuera del laboratorio.

En el plano académico abierto, **OpenVLA/OFT**, **π0/π0.5**, **RDT-1B**, **Diffusion-VLA**, **GR00T N1** y **SmolVLA** resumen bastante bien la frontera reproducible actual [6][7][8][9][10][11][13][14]. En el plano industrial, **Helix** y **Helix 02** muestran hacia dónde apunta la carrera por humanoides generalistas [12][22]. El siguiente gran salto probablemente no será un único modelo “más grande”, sino una síntesis mejor entre semántica abierta, control motor robusto, evaluación exigente y despliegue eficiente en tiempo real.


## Referencias

[1] Reed, S. et al. (2022). A Generalist Agent. Transactions on Machine Learning Research. arXiv:2205.06175.

[2] Driess, D. et al. (2023). PaLM-E: An Embodied Multimodal Language Model. arXiv:2303.03378.

[3] Brohan, A. et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.

[4] Open X-Embodiment Collaboration et al. (2023/2025). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.

[5] OM Team et al. (2024). Octo: An Open-Source Generalist Robot Policy. arXiv:2405.12213.

[6] Kim, M. J. et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.

[7] Black, K. et al. (2024/2026). π0: A Vision-Language-Action Flow Model for General Robot Control. RSS 2025. arXiv:2410.24164.

[8] Liu, S. et al. (2024/2025). RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation. arXiv:2410.07864.

[9] Wen, J. et al. (2024/2025). Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning. ICML 2025. arXiv:2412.03293.

[10] Kim, M. J., Finn, C., & Liang, P. (2025). Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. RSS 2025. arXiv:2502.19645.

[11] Bjorck, J. et al. (2025). GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734.

[12] Figure AI (2025). Helix: A Vision-Language-Action Model for Generalist Humanoid Control. Official technical release.

[13] Physical Intelligence et al. (2025). π0.5: a Vision-Language-Action Model with Open-World Generalization. arXiv:2504.16054.

[14] Shukor, M. et al. (2025). SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv:2506.01844.

[15] Kawaharazuka, K. et al. (2025). Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications. IEEE Access / arXiv:2510.07077.

[16] Jiang, W. et al. (2026). How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf. arXiv:2602.18397.

[17] Liu, B. et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.

[18] Mees, O. et al. (2021/2022). CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks. arXiv:2112.03227.

[19] Khazatsky, A. et al. (2024). DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. arXiv:2403.12945.

[20] LIBERO-X authors (2026). LIBERO-X: Robustness Litmus for Vision-Language-Action Models. arXiv:2602.06556.

[21] Kim, C. et al. (2026). LIBERO-Para: A Diagnostic Benchmark and Metrics for Paraphrase Robustness in VLA Models. arXiv:2603.28301.

[22] Figure AI (2026). Helix 02: Full-Body Autonomy. Official technical release.
