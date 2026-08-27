# INFORME DE EVALUACIÓN DEL TFM

## Ficha inicial del trabajo

| Elemento | Información verificable en el PDF |
|---|---|
| Tipo y modalidad | Proyecto Fin de Máster. Estudio experimental comparativo y *benchmarking* en aprendizaje por imitación y aprendizaje profundo. |
| Universidad y centro | Universidad de Navarra, Tecnun Escuela de Ingeniería. |
| Máster | Máster Universitario en Análisis de Datos en Ingeniería. |
| Título | *Influencia del codificador visual y su estrategia de entrenamiento en Diffusion Policy para manipulación robótica: estudio en Push-T*. |
| Autor | Moises Britez. |
| Director | Diego Borro. |
| Fecha y lugar | Donostia-San Sebastián, septiembre de 2026. |
| Problema investigado | Se desconoce qué configuración de codificador visual y estrategia de adaptación resulta más conveniente para una Diffusion Policy en Push-T con pocas demostraciones. |
| Pregunta de investigación | No se formula como pregunta única. El Apartado 1.2 plantea dos cuestiones implícitas: qué codificador conviene y qué estrategia, entrenamiento desde cero, congelación o ajuste fino, conviene en esa configuración. |
| Objetivo general | Evaluar la influencia del codificador visual y de su estrategia de entrenamiento sobre el rendimiento de una Diffusion Policy aplicada a Push-T. |
| Objetivos específicos | Implementar cinco variantes; comparar su puntuación media; analizar parámetros, tiempos, latencia y memoria. |
| Hipótesis | H1: el preentrenamiento mejora al entrenamiento desde cero. H2: el ajuste fino mejora a la congelación. H3: los ViT alcanzan rendimiento comparable a las redes convolucionales, con mayor coste de inferencia. |
| Datos de ajuste | Dataset público declarado de Push-T con 206 demostraciones teleoperadas, 25.650 transiciones y unos 30 MB. Se emplean 90 episodios para entrenamiento, 4 para validación y se descartan 112. |
| Datos de selección de modelos | 50 condiciones iniciales del simulador, correspondientes al intervalo de semillas 100.000 a 100.049 según el Apartado 4.5, evaluadas cada 50 épocas. Esas mismas condiciones seleccionan el punto de control y producen los resultados finales. |
| Procedencia | El PDF atribuye el dataset y el entorno a los autores de Diffusion Policy. El autor declara haber cotejado el primer estado de las 206 demostraciones con las semillas 0 a 205. La ejecución de esa comprobación es `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` |
| Periodo de generación de datos | `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` |
| Población o estimando | No se define formalmente. El objeto observado más preciso es la puntuación de un punto de control seleccionado, en una ejecución de entrenamiento, sobre 50 condiciones concretas del simulador y una realización no documentada del muestreo estocástico de la política. |
| Variables experimentales | Bloque visual, pesos iniciales, estrategia de entrenamiento, arquitectura, resolución, recorte y agregación espacial. Las últimas cuatro no permanecen constantes en todas las comparaciones. |
| Variables de resultado | Puntuación máxima de cobertura por episodio, media, desviación típica, error estándar, pérdidas, error de acción, tiempos, latencia y memoria reservada. |
| Métodos estadísticos | Diferencias pareadas, intervalos de confianza del 95 % cuyo método no se especifica, prueba de rangos con signo de Wilcoxon mediante aproximación normal y corrección de Holm para diez pares. |
| Algoritmos | Diffusion Policy con DDPM, red U-Net unidimensional y cinco bloques visuales: ResNet-18 desde cero, ResNet-18 ImageNet congelada, ResNet-18 ImageNet ajustada, DINOv2 ViT-S/14 congelado y CLIP ViT-B/16 congelado. |
| Código y repositorio | El PDF declara una bifurcación propia y muestra la URL `https://github.com/moisesbritez92/tfm_madi.git` en la Bibliografía, página 43. No se ha accedido ni auditado el repositorio. Su operatividad, contenido y correspondencia con el PDF son `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` |
| Anexos | No aparecen anexos en el índice ni en las 59 páginas PDF. |
| Normas institucionales disponibles | Solo se aplican los requisitos Tecnun incluidos en el agente evaluador. No se ha aportado el reglamento institucional completo ni una plantilla oficial versionada. |
| Tiempo previsto para la defensa | `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` |
| Ética, privacidad y conflictos | El estudio usa simulación y demostraciones publicadas; no se describen datos personales. Licencias del dataset, declaración de conflictos y declaración de uso de IA generativa: `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` |

### Alcance de la evaluación

La evaluación se limita al contenido verificable en `main.pdf`. No se han auditado código, registros, ficheros de configuración, fuentes LaTeX, artefactos ni contenido del repositorio. Cuando una comprobación exige esos materiales se indica literalmente `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` Las páginas citadas corresponden a la numeración impresa de la memoria. En preliminares se usa la numeración romana visible o la página PDF.

## 1. Dictamen preliminar

**Dictamen: Requiere reformulación antes de defensa.**

La memoria delimita el experimento, informa resultados negativos, identifica varios confusores y documenta costes con precisión. Sin embargo, la evidencia principal no procede de una evaluación final independiente. Las mismas 50 condiciones reservadas se consultan de cuatro a diez veces, seleccionan el punto de control y generan los intervalos y contrastes finales. Este procedimiento introduce sesgo de selección y hace que los intervalos y valores p de los puntos elegidos no tengan la interpretación confirmatoria que se les atribuye.

La segunda deficiencia estructural es la existencia de una sola ejecución de entrenamiento por variante. Los 50 episodios no son 50 réplicas de la estrategia de entrenamiento. Solo describen la variación entre condiciones iniciales y, al existir muestreo de difusión estocástico, incorporan además una fuente de variabilidad no separada. No se puede inferir con este diseño que una estrategia o familia de codificadores sea superior de forma estable.

La desigualdad de presupuestos y reglas de parada, la interrupción manual de V3 antes del mínimo declarado y los factores confundidos entre V0 y las restantes variantes reducen aún más la comparabilidad. El trabajo sí permite una conclusión descriptiva más estrecha: entre los cinco artefactos entrenados, V0 obtuvo la mayor puntuación observada en el conjunto usado para selección y el menor tiempo por época en el equipo descrito. Esta conclusión no equivale a demostrar la superioridad de un codificador o una estrategia de entrenamiento.

## 2. Resumen analítico del TFM

El TFM estudia si el preentrenamiento visual beneficia a una política de difusión en un régimen de 90 demostraciones de Push-T. Para ello construye cinco bloques visuales que alimentan una misma red generativa de unos 277 millones de parámetros. Las variantes combinan ResNet-18, DINOv2 y CLIP con entrenamiento desde cero, congelación o ajuste fino. El estudio registra rendimiento en bucle cerrado, pérdidas y cuatro dimensiones de coste computacional.

El diseño usa 90 episodios para ajustar los modelos y 4 para calcular la pérdida de validación. Otros 112 episodios se excluyen siguiendo el límite de la implementación de referencia. Cada 50 épocas se ejecutan 50 condiciones iniciales reservadas del simulador. El mejor punto de control según la media de esas condiciones se conserva y se compara con los mejores puntos de las demás variantes mediante diferencias pareadas, Wilcoxon y Holm.

Los máximos informados son 0,864 para V0, 0,668 para V1, 0,648 para V2, 0,622 para V3 y 0,535 para V4. V0 también consume menos tiempo por época. El análisis de pérdidas muestra divergencia entre entrenamiento y validación, sobre todo en las variantes preentrenadas. La memoria concluye que ningún bloque preentrenado evaluado mejora a V0 en la configuración estudiada, pero restringe la atribución causal al bloque visual completo porque V0 también cambia resolución, recorte y agregación espacial.

La contribución verificable es una comparación descriptiva bien documentada de cinco ejecuciones y de su coste sobre un equipo concreto. La contribución inferencial es limitada: no existe prueba final, no se estima variación entre entrenamientos y las oportunidades de selección no son equivalentes. Por ello, el estudio no establece de forma confirmatoria el efecto de la estrategia ni una ordenación general de codificadores.

## 3. Puntuación global

| Criterio | Máximo | Obtenido | Justificación |
|---|---:|---:|---|
| A. Problema, justificación y relevancia | 8 | 7 | El problema está delimitado a Push-T, 90 demostraciones y cinco configuraciones; el hueco se vincula con antecedentes concretos. Su relevancia científica queda circunscrita a una matriz experimental muy específica. |
| B. Pregunta, objetivos e hipótesis | 10 | 8 | Los objetivos son claros y medibles. La pregunta no se formula expresamente y el objetivo de evaluar la «influencia» excede lo identificable con una ejecución y factores confundidos. |
| C. Estado del arte y marco conceptual | 10 | 8 | Revisión actual, mayoritariamente primaria y con síntesis comparativa. Falta estrategia de búsqueda, evaluación de calidad y una conexión más rigurosa entre el gap y el diseño capaz de cubrirlo. |
| D. Diseño metodológico | 14 | 6 | La comparación comparte tarea y red generativa, pero reutiliza el conjunto de selección, usa una semilla, aplica presupuestos y paradas desiguales, detiene V3 manualmente y no mantiene constante todo salvo el factor de interés. |
| E. Datos y preparación | 12 | 8 | Se cuantifican origen declarado, reparto, ventanas y distribuciones. Cuatro episodios de validación son insuficientes, el descarte se justifica mediante no significación y faltan controles verificables de integridad, duplicados y anomalías. |
| F. Análisis estadístico y modelado | 16 | 6 | Wilcoxon pareado y Holm son razonables en abstracto, pero se aplican después de seleccionar sobre los mismos episodios. El método de los IC no se especifica, no se trata la variabilidad entre entrenamientos ni la estocasticidad de inferencia y no se realiza equivalencia. |
| G. Resultados, validación y discusión | 15 | 6 | Las tablas son claras y se informan resultados adversos. No existe validación final independiente, la comparación no es plenamente justa y varias interpretaciones exceden el estimando observado. |
| H. Conclusiones, aportación y limitaciones | 8 | 6 | La memoria reconoce selección, semilla, presupuestos y confusión. Aun así, algunas formulaciones sobre degradación de representaciones y dominancia de estrategias son más fuertes que la evidencia. |
| I. Reproducibilidad, transparencia y ética | 4 | 2 | Se aportan hardware, versiones, semilla, hiperparámetros y URL. Faltan commit, hashes, comandos, configuraciones concretas, artefactos, parámetros completos del planificador y semillas de inferencia. |
| J. Presentación académica | 3 | 2 | La estructura, tablas y figuras son legibles. Persisten dos marcadores `??`, agradecimientos sin sustituir, una Bibliografía separada no numerada y alguna terminología gráfica en inglés. |
| **TOTAL** | **100** | **59** | **Suma exacta de A a J: 59.** |

**TOTAL: 59/100**

**Calificación según la escala del agente: Deficiente.**

### Techo de puntuación aplicado

Se activa el techo de **máximo 59/100** del Apartado 8 del agente por dos condiciones expresas:

1. El conjunto empleado para seleccionar el modelo se utiliza también para producir los resultados principales. Aunque la memoria lo denomina correctamente «conjunto de selección» y no «prueba», la ausencia de un bloque final independiente cumple materialmente el supuesto «el conjunto de prueba fue utilizado para seleccionar el modelo» cuando esos resultados se interpretan como rendimiento final.
2. Los resultados principales carecen de validación independiente. Los máximos, intervalos y contrastes se calculan sobre los mismos 50 episodios consultados durante la búsqueda del punto de control.

También concurren motivos del techo de 69/100: validación insuficiente, reproducibilidad parcial, objetivos cumplidos solo parcialmente y sesgos relevantes no resueltos. Prevalece el techo más restrictivo de 59/100. La puntuación por criterios ya incorpora estas deficiencias y suma exactamente el techo; no se añade una penalización posterior.

## 4. Perfil de puntuación

### Tres dimensiones más fuertes

1. **Delimitación y transparencia del alcance.** El documento identifica una tarea, cinco variantes, 90 demostraciones y un hardware concreto. Además, reconoce que V0 frente a V1 y V2 no aísla la inicialización, páginas 15 a 16, y que la selección no es una prueba final, páginas 18 y 30.
2. **Estado del arte orientado al problema.** El Capítulo 2 compara Diffusion Policy, extensiones geométricas y representaciones visuales, con una síntesis del hueco en las páginas 8 a 9.
3. **Caracterización del coste.** Las Tablas 8, 11 y 12 separan tiempo por época, tiempo total, latencia del codificador, latencia de política y memoria. Esta separación evita atribuir al codificador el coste dominado por cien pasos de difusión.

### Tres dimensiones más débiles

1. **Validación final.** No hay datos ajenos a la selección del punto de control.
2. **Unidad de replicación e incertidumbre.** Una ejecución por variante impide estimar la variación entre entrenamientos y sustentar inferencias sobre estrategias.
3. **Comparabilidad causal.** Presupuestos, reglas de parada, oportunidades de evaluación y componentes del bloque visual cambian simultáneamente.

### Principal riesgo metodológico

El riesgo principal es confundir el rendimiento máximo seleccionado en 50 condiciones repetidamente consultadas con rendimiento final no sesgado. Este riesgo afecta directamente a las diferencias, los intervalos de confianza, los valores p y la conclusión central.

## 5. Matriz de coherencia

### Cadena de coherencia científica

`problema -> pregunta -> objetivo -> hipótesis -> datos -> método -> resultado -> conclusión`

| Eslabón | Formulación en la memoria | Evaluación y ruptura |
|---|---|---|
| Problema | No se conoce qué bloque visual y estrategia convienen para Diffusion Policy en Push-T con pocas demostraciones, páginas 1 y 8 a 9. | Problema concreto y delimitado. El alcance científico es específico, no universal. |
| Pregunta | Dos cuestiones implícitas: arquitectura y estrategia, página 1. | Ruptura menor: no existe pregunta formal con población, intervención, comparador y resultado. |
| Objetivo | Evaluar la influencia del codificador y la estrategia, página 2. | Ruptura: «influencia» sugiere identificación del efecto, pero varios factores cambian conjuntamente y solo existe una ejecución por variante. |
| Hipótesis | Preentrenamiento superior; ajuste fino superior; ViT comparable con mayor coste, página 11. | Son contrastables en principio. La comparabilidad exige equivalencia o no inferioridad, no ausencia de significación. |
| Datos | 90 episodios de entrenamiento, 4 de validación y 50 condiciones de selección, páginas 12 y 17 a 18. | Ruptura crítica: no existe muestra final independiente y la validación de pérdida tiene cuatro unidades episódicas. |
| Método | Cinco entrenamientos, evaluación periódica, selección del máximo, Wilcoxon y Holm, páginas 16 a 19. | Ruptura crítica: selección e inferencia usan los mismos episodios. Ruptura mayor: presupuestos y paradas desiguales. |
| Resultado | V0 obtiene el mayor máximo y menor tiempo por época, Tablas 8 a 12. | Resultado descriptivo de los artefactos observados. Su incertidumbre no representa la variación entre entrenamientos. |
| Conclusión | Ningún bloque preentrenado mejora a V0 en la configuración; la ventaja corresponde al bloque completo, páginas 31 a 33. | Parcialmente sustentada si se restringe a los cinco artefactos y al conjunto de selección. No sustenta superioridad estable de estrategias ni generalización fuera de Push-T. |

### Rupturas principales

La cadena se rompe entre datos, método y resultado porque no existe separación entre selección y prueba. También se rompe entre objetivo, método y conclusión: el objetivo habla de influencia, pero la comparación V0 frente a las variantes preentrenadas cambia inicialización, resolución, recorte, agregación y, en V3/V4, arquitectura. La comparación V1 frente a V2 sí aísla congelación frente a ajuste fino dentro de una misma inicialización y preprocesado, pero una sola pareja de entrenamientos no estima un efecto estable.

### Matriz por objetivo

| Objetivo | Método | Datos | Resultado | Conclusión | Estado |
|---|---|---|---|---|---|
| OE1. Implementar cinco variantes manteniendo constantes arquitectura de control y configuración experimental. | Configuración de V0 a V4, proyección a 512 y red generativa común, Apartados 3.3 a 3.5. | Cinco registros de entrenamiento y tablas de parámetros. | Se informan resultados para las cinco variantes. Cambian presupuesto, lote físico, resolución, recorte y agregación; V3 se interrumpe manualmente. | La existencia de salidas es coherente con cinco ejecuciones, pero la corrección de la implementación es `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` La constancia experimental solo se cumple parcialmente. | **Parcialmente cumplido.** |
| OE2. Comparar el rendimiento mediante la puntuación media de Push-T. | Evaluación de 50 condiciones, selección del máximo, diferencias pareadas, Wilcoxon y Holm. | Los mismos 50 episodios en cada checkpoint y una ejecución por variante. | Máximos de 0,864; 0,668; 0,648; 0,622; 0,535. | Se compara descriptivamente, pero no se obtiene rendimiento final independiente ni inferencia entre entrenamientos. | **Parcialmente cumplido.** |
| OE3. Analizar coste mediante parámetros, tiempos y memoria. | Conteo de parámetros, cronometría, 50 repeticiones de latencia y 12 pasos para memoria. | Registros de las cinco variantes en los entornos declarados. | Tablas 4, 8, 11 y 12. | Se caracteriza el coste en el equipo y configuraciones descritos. V2 contiene tiempos parcialmente estimados y el total no es comparable por presupuestos. | **Cumplido dentro del alcance del equipo descrito.** |
| Objetivo general. Evaluar la influencia del codificador y su estrategia. | Síntesis de OE1 a OE3. | Una tarea sintética, una ejecución por variante, condiciones de selección reutilizadas. | V0 presenta la mayor puntuación observada y menor coste unitario. | No se identifica de forma robusta la influencia aislada ni su estabilidad. Sí se documenta una comparación exploratoria de cinco bloques. | **Parcialmente cumplido.** |

## 6. Fortalezas demostrables

1. La memoria declara de forma explícita que las 50 condiciones reservadas cumplen función de selección y no de prueba, página 18, Apartado 3.7. Esta precisión evita una denominación engañosa, aunque no corrige el diseño.
2. Se informa que solo existe una semilla de entrenamiento y se distingue expresamente la dispersión entre episodios de la variación entre entrenamientos, páginas 11, 18 y 29.
3. Se documentan los presupuestos desiguales, el momento en que se definió la parada y la interrupción manual de V3, páginas 16 a 17. El documento no oculta la desviación del protocolo.
4. El confusor entre V0 y las variantes preentrenadas se identifica con sus cuatro componentes, inicialización, resolución, recorte y agregación, páginas 15 a 16 y 31 a 32.
5. Se muestran variantes que rinden peor y curvas que retroceden tras su máximo, Figuras 2 y 3. El informe no selecciona únicamente resultados favorables a las hipótesis originales.
6. La métrica de cobertura se define mediante la Ecuación (5), con umbral, agregación temporal y rango, página 18.
7. El coste de inferencia se descompone entre llamada completa y codificador aislado, Tabla 11. Esta decisión permite comprobar que el coste del codificador queda diluido por los cien pasos de difusión.
8. Las limitaciones sobre simulación, robustez, semilla, selección y desigualdad presupuestaria aparecen en los Apartados 1.4 y 4.7, en vez de relegarse a una mención genérica.
9. Hardware, versiones principales, semilla global, split, hiperparámetros y entorno se documentan en las Tablas 5 a 7.
10. La conclusión sobre V0 se restringe en varios pasajes al bloque visual completo y a Push-T, lo que reduce, aunque no elimina, la sobreinterpretación.

## 7. Hallazgos críticos

### Hallazgo C1. Reutilización del conjunto de selección como evaluación final

**Ubicación:** páginas 16 a 18, Apartados 3.5 a 3.7; página 23, Tabla 9; página 25, Tabla 10; página 30, Apartado 4.7.

**Tipo:** error de validación; sesgo de selección; *leakage* de selección; error estadístico.

**Descripción:** las 50 condiciones reservadas se evalúan cada 50 épocas, determinan qué checkpoint se conserva y vuelven a utilizarse para reportar el máximo, construir intervalos y efectuar contrastes. No existe un conjunto final independiente.

**Evidencia:** el Apartado 3.5 indica que se conservan los tres mejores checkpoints según la puntuación de evaluación. El Apartado 3.7 reconoce que esas 50 condiciones «cumplen la función de un conjunto de selección». La Tabla 9 reporta el mejor valor entre 4 y 10 evaluaciones por variante. El Apartado 4.7 reconoce el sesgo optimista.

**Por qué es problemático:** el checkpoint es una función de los mismos 50 resultados sobre los que después se calcula incertidumbre. La selección favorece fluctuaciones positivas y viola la condición de análisis fijado antes de observar los datos. Los valores p e intervalos convencionales condicionados al checkpoint seleccionado no incorporan esa búsqueda.

**Impacto sobre resultados:** las puntuaciones máximas están sesgadas al alza; las diferencias pueden estar afectadas de forma desigual porque V0 y V1 tuvieron diez oportunidades y V3/V4 solo cuatro. La evidencia no valida rendimiento fuera del conjunto de selección ni la significación confirmatoria de la conclusión principal.

**Corrección necesaria:** seleccionar checkpoints con un conjunto de validación y evaluarlos una sola vez en un bloque disjunto de semillas. Si se desean inferencias sobre estrategias de entrenamiento, el procedimiento debe repetirse con varias semillas de entrenamiento y el entrenamiento debe ser la unidad de replicación.

**Evidencia necesaria para considerarlo resuelto:** resultados bloqueados de los checkpoints ya seleccionados sobre semillas nunca consultadas; protocolo de selección y análisis fijado antes de esa evaluación; tabla de resultados finales sin nueva selección; análisis que separe variación entre entrenamientos y entre episodios.

No se detectan otros hallazgos críticos independientes. Los restantes problemas son mayores, aunque varios refuerzan el impacto del Hallazgo C1.

## 8. Observaciones mayores

### Hallazgo M1. Una sola semilla no permite inferir sobre estrategias de entrenamiento

**Ubicación:** página 2, Apartado 1.4; página 11, Apartado 3.1; páginas 18 a 19, Apartado 3.7; página 29, Apartado 4.7.

**Tipo:** limitación reconocida; error de inferencia si se generaliza; incertidumbre no cuantificada.

**Descripción:** cada variante corresponde a un único ajuste con semilla 42. Los 50 episodios son evaluaciones del mismo modelo, no 50 réplicas del algoritmo de entrenamiento.

**Evidencia:** el PDF reconoce que los intervalos y contrastes «no sustituyen» la variabilidad entre semillas. A pesar de ello, las hipótesis y conclusiones se expresan en términos de preentrenamiento, ajuste fino y familias de codificadores.

**Por qué es problemático:** la inicialización, el orden de lotes, operaciones no deterministas y trayectoria de optimización pueden cambiar el checkpoint obtenido. Sin repeticiones no se conoce si las diferencias V0 frente a V1 a V4 son estables ni si la diferencia V1 frente a V2 cambia de signo.

**Impacto sobre resultados:** los IC y valores p de la Tabla 10 no sustentan inferencias sobre el rendimiento esperado de volver a entrenar cada estrategia. Solo describen cinco modelos concretos.

**Corrección necesaria:** repetir cada condición con semillas independientes bajo un protocolo idéntico y analizar el rendimiento final por semilla. Si el cómputo no lo permite, reformular objetivos y conclusiones como estudio exploratorio de ejecuciones únicas.

**Evidencia necesaria para considerarlo resuelto:** distribución de resultados finales por variante y semilla, estimación de variación entre entrenamientos y análisis jerárquico o resumen por réplica que use el entrenamiento como unidad experimental.

### Hallazgo M2. Presupuestos, oportunidades de selección y reglas de parada desiguales

**Ubicación:** páginas 16 a 17, Apartado 3.5 y Tabla 5; páginas 21 a 23, Tablas 8 y 9; página 29, Apartado 4.7.

**Tipo:** error metodológico; comparabilidad insuficiente; decisión no preespecificada; limitación reconocida.

**Descripción:** V0/V1 reciben 500 épocas, V2/V3 300 y V4 200. La parada se define después de observar V0/V1. V2 se detiene por inspección, V3 manualmente en el índice 154 antes del mínimo de 200 y V4 agota su presupuesto. Las variantes tienen entre cuatro y diez evaluaciones.

**Evidencia:** el Apartado 3.5 afirma que la regla «no precedió al experimento» y que las decisiones se tomaron «por inspección». También reconoce menos oportunidades para V3/V4.

**Por qué es problemático:** el máximo depende del número de oportunidades y del horizonte de entrenamiento. Un protocolo adaptado durante la campaña permite que observaciones previas influyan en condiciones posteriores. V3 no satisface ni la regla descrita para la propia campaña.

**Impacto sobre resultados:** V0/V1 pueden beneficiarse de más oportunidades de máximo, mientras V3 puede quedar infradesarrollada. El contrafactual de parada uniforme solo se reconstruye para checkpoints ya observados y no recupera las épocas nunca ejecutadas.

**Corrección necesaria:** fijar antes de entrenar el presupuesto, frecuencia de evaluación, paciencia, mínimo, criterio y recurso computacional comparable. Completar V3 o excluir cualquier afirmación que requiera haber observado su trayectoria prevista.

**Evidencia necesaria para considerarlo resuelto:** protocolo fechado o preregistrado, registros completos con el mismo criterio y tabla de oportunidades de selección equivalentes.

### Hallazgo M3. La comparación principal confunde varios componentes del bloque visual

**Ubicación:** páginas 15 a 16, Apartado 3.4; páginas 31 a 33, Apartado 5.1.

**Tipo:** variable de confusión; limitación reconocida; conclusión causal restringida.

**Descripción:** V0 frente a V1/V2 cambia inicialización, entrada de 96 a 224 píxeles, recorte y agregación spatial softmax frente a descriptor global. Frente a V3/V4 cambia además la arquitectura. Solo V1 frente a V2 aísla la estrategia congelada frente a ajuste fino.

**Evidencia:** el propio Apartado 3.4 enumera los cuatro factores y el Apartado 5.1 limita la atribución al bloque completo.

**Por qué es problemático:** el título, objetivo general e hipótesis se refieren al codificador y a su estrategia. La comparación no identifica qué componente causa la diferencia V0 frente a las alternativas.

**Impacto sobre resultados:** es válida la ordenación observada de bloques completos, pero no afirmar que el preentrenamiento, la arquitectura o la estrategia explica por sí sola la ventaja de V0.

**Corrección necesaria:** diseño factorial o ablaciones que igualen resolución, aumento y agregación, variando un factor cada vez. Mantener además presupuesto y selección comunes.

**Evidencia necesaria para considerarlo resuelto:** resultados de las ablaciones propuestas en la página 33, con evaluación independiente y repeticiones de entrenamiento.

### Hallazgo M4. La estocasticidad de inferencia no está controlada ni cuantificada

**Ubicación:** página 14, Ecuación (4); páginas 17 a 19, Apartados 3.6 y 3.7; página 29, Apartado 4.7.

**Tipo:** información faltante; incertidumbre no cuantificada; problema de reproducibilidad.

**Descripción:** la política inicia la acción con ruido y añade ruido fresco en cada paso de difusión. El PDF no especifica semillas de inferencia, sincronización de ruido entre variantes ni repeticiones por condición y checkpoint.

**Evidencia:** la Ecuación (4) contiene un término aleatorio. El protocolo describe una ejecución por condición, pero no identifica el estado del generador aleatorio. El Apartado 4.7 atribuye la dispersión a condiciones iniciales sin separar esta fuente.

**Por qué es problemático:** dos políticas evaluadas en la misma condición inicial no forman necesariamente un par que difiera solo por la política. Una realización favorable o adversa del muestreo puede alterar el episodio. La incertidumbre atribuida solo a semillas del entorno mezcla dos fuentes.

**Impacto sobre resultados:** diferencias pareadas, IC, Wilcoxon y reproducibilidad exacta quedan insuficientemente caracterizados.

**Corrección necesaria:** fijar y documentar semillas de inferencia comunes o repetir varias trayectorias de difusión por condición. Analizar por separado condición inicial, semilla de inferencia y semilla de entrenamiento.

**Evidencia necesaria para considerarlo resuelto:** protocolo explícito de generación aleatoria, tabla de repeticiones y descomposición de variabilidad o análisis jerárquico.

### Hallazgo M5. La pérdida de validación se estima con cuatro episodios

**Ubicación:** página 12, Tabla 3; página 18, Apartado 3.7; páginas 25 a 26, Figura 3 y Apartado 4.3.

**Tipo:** validación insuficiente; tamaño muestral débilmente justificado.

**Descripción:** la validación contiene 4 episodios y 404 ventanas solapadas. La unidad independiente es el episodio, no cada ventana. La pérdida se usa para interpretar sobreajuste y apoyar la parada.

**Evidencia:** la Tabla 3 informa cuatro episodios. El Apartado 4.3 interpreta ascensos de pérdida como sobreajuste.

**Por qué es problemático:** cuatro trayectorias pueden no representar la heterogeneidad de estados. Las ventanas de una trayectoria comparten contexto y no multiplican por 404 el tamaño efectivo. Una pérdida de ruido sobre cuatro episodios puede fluctuar y tampoco es sustituto directo del rendimiento en bucle cerrado, como el propio documento observa para V0.

**Impacto sobre resultados:** la estabilidad de las curvas de validación y su utilidad para detener entrenamiento son débiles. La explicación de sobreajuste es compatible con los datos, pero no queda demostrada con robustez.

**Corrección necesaria:** ampliar la validación a más episodios, definirla antes del entrenamiento y estimar incertidumbre por episodio. Usar el conjunto solo para selección, manteniendo un test final intacto.

**Evidencia necesaria para considerarlo resuelto:** pérdida y rendimiento de validación sobre una muestra episódica suficiente, con dispersión y estabilidad frente a cambios de split.

### Hallazgo M6. Especificación estadística incompleta y multiplicidad no resuelta

**Ubicación:** páginas 18 a 19, Apartado 3.7; página 25, Tabla 10.

**Tipo:** error estadístico; información faltante; interpretación insuficiente.

**Descripción:** no se indica cómo se construyen los IC de diferencias. La aproximación normal de Wilcoxon no documenta tratamiento de empates, ceros ni corrección de continuidad. Holm controla la familia de diez pares, pero no la búsqueda de checkpoints, la adopción de la parada tras observar curvas ni los IC múltiples. Los intervalos se declaran sin ajuste.

**Evidencia:** la Tabla 10 solo señala «aproximación normal» y «los intervalos no incorporan este ajuste».

**Por qué es problemático:** un IC de la diferencia media y una prueba de Wilcoxon se refieren a parámetros distintos si el intervalo no corresponde al estimando de rangos. Con 50 pares, la aproximación normal puede ser suficiente en condiciones regulares, pero esas condiciones no se documentan. La corrección Holm es válida solo para la familia declarada y un análisis fijado de antemano.

**Impacto sobre resultados:** no puede verificarse la cobertura de los IC. Los valores p ajustados siguen siendo posselección y no controlan el error global del procedimiento experimental.

**Corrección necesaria:** definir estimando, método de IC, algoritmo exacto de Wilcoxon y familia de hipótesis antes del análisis. Aplicar inferencia a resultados finales independientes y usar réplicas de entrenamiento para conclusiones sobre estrategias.

**Evidencia necesaria para considerarlo resuelto:** protocolo estadístico reproducible, salidas completas de cada prueba, control de multiplicidad coherente y análisis final no reutilizado para selección.

### Hallazgo M7. La ausencia de significación se interpreta como ausencia de sesgo o comparabilidad

**Ubicación:** página 14, final del Apartado 3.2; páginas 28 y 31, Apartados 4.6 y 5.1.

**Tipo:** error de interpretación; conclusión que excede la evidencia.

**Descripción:** el PDF afirma que descartar 112 episodios «no introduce un sesgo apreciable» porque U y KS no detectan diferencias. También usa la no significación entre variantes preentrenadas para hablar de rendimiento comparable o ausencia de orden.

**Evidencia:** los p de longitud y puntuación son 0,752/0,999 y 0,302/0,418; cinco KS de condiciones iniciales quedan entre 0,232 y 0,696. V1 frente a V2 produce p = 0,82 y varios IC amplios.

**Por qué es problemático:** no rechazar igualdad no demuestra equivalencia ni ausencia de sesgo. Las comprobaciones son univariantes y no evalúan la distribución conjunta de estado, acciones, contactos, imágenes y dificultad. La comparabilidad requiere un margen de equivalencia y potencia adecuada.

**Impacto sobre resultados:** la aleatorización del sorteo sí evita un descarte deliberado por calidad, pero la equivalencia empírica de subconjuntos no queda demostrada. Tampoco se demuestra que estrategias o arquitecturas tengan rendimiento comparable.

**Corrección necesaria:** reformular como «no se detectaron diferencias en las variables comprobadas» y, si se pretende equivalencia, predefinir márgenes y aplicar pruebas o IC de equivalencia. Analizar balance multivariante y sensibilidad al split.

**Evidencia necesaria para considerarlo resuelto:** lenguaje corregido, márgenes justificados, análisis de equivalencia o balance conjunto y repetición con otros splits.

### Hallazgo M8. Ecuaciones de difusión y planificador insuficientes para reproducción

**Ubicación:** página 14, Ecuaciones (3) y (4); página 17, Tabla 5; páginas 20 a 21, Apartado 3.8.

**Tipo:** problema conceptual; información faltante; reproducibilidad.

**Descripción:** la Ecuación (3) representa la secuencia ruidosa como `A0 + epsilon_k` sin exponer los factores dependientes de la acumulación de alfas. La Ecuación (4) usa coeficientes genéricos alfa, gamma y sigma, pero no los define. «DDPM, varianza fija, predicción de ruido» y 100 pasos no especifican el calendario ni todos los parámetros del planificador.

**Evidencia:** el texto remite los coeficientes al planificador, sin presentar sus fórmulas, valores o configuración exacta.

**Por qué es problemático:** existen varios calendarios beta, tipos de varianza, recortes, espaciados y convenciones de predicción que producen procedimientos distintos. La formulación no basta para reconstruir el entrenamiento ni verificar que las Ecuaciones (3) y (4) coinciden con la implementación.

**Impacto sobre resultados:** un tercero no puede repetir el proceso de ruido y muestreo solo desde el PDF. La correspondencia entre ecuaciones, biblioteca y código es `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.`

**Corrección necesaria:** sustituir las expresiones esquemáticas por la formulación DDPM exacta usada y listar la configuración completa del planificador, incluida la secuencia beta, tipo de varianza, predicción, recorte, timesteps y tratamiento del último paso.

**Evidencia necesaria para considerarlo resuelto:** ecuaciones consistentes, tabla de parámetros exactos y referencia a un fichero de configuración versionado con hash.

### Hallazgo M9. El estimando y la población de generalización no están definidos

**Ubicación:** páginas 1 a 2, Apartados 1.2 a 1.4; páginas 17 a 19, Apartados 3.6 y 3.7; páginas 31 a 33, Conclusiones.

**Tipo:** debilidad conceptual; limitación de validez externa; información faltante.

**Descripción:** no se precisa si el objetivo es la media sobre las 50 semillas concretas, la expectativa sobre el generador de condiciones de Push-T, la ejecución repetida de un checkpoint o el rendimiento esperado de una estrategia vuelta a entrenar.

**Evidencia:** el estudio usa una tarea, un entorno sintético, 50 semillas consecutivas y una ejecución por variante. Las conclusiones alternan entre «en esta ejecución», «en Push-T» y formulaciones sobre bloques o estrategias.

**Por qué es problemático:** cada estimando exige unidades y muestreo distintos. Episodios permiten inferir sobre condiciones para un modelo fijo; entrenamientos permiten inferir sobre algoritmos de entrenamiento; tareas permiten inferir sobre dominios. El estudio solo observa el primer nivel y ni siquiera dispone de test independiente.

**Impacto sobre resultados:** no cabe generalizar a codificadores, control robótico, otras tareas, observaciones reales ni nuevas ejecuciones de entrenamiento. Tampoco se demuestra representatividad probabilística de las 50 semillas para todo el generador.

**Corrección necesaria:** definir población, unidad experimental, estimando y alcance antes del análisis. Restringir la conclusión al estimando realmente observado o ampliar el muestreo en los niveles pertinentes.

**Evidencia necesaria para considerarlo resuelto:** definición formal del estimando y diseño de muestreo que permita estimarlo, con conclusiones redactadas en el mismo nivel.

### Hallazgo M10. Reproducibilidad material incompleta y afirmación de control inconsistente

**Ubicación:** páginas 20 a 21, Apartado 3.8 y Tablas 6 y 7; página 43, Bibliografía; página 20, párrafo «cualquier diferencia».

**Tipo:** problema de reproducibilidad; inconsistencia global; conclusión causal excesiva.

**Descripción:** se aportan versiones y URL, pero faltan commit o etiqueta, hash del dataset y pesos, comandos, nombres y contenido de configuraciones, archivos de entorno, checkpoints y artefactos. El texto afirma que «cualquier diferencia entre variantes es atribuible al codificador», aunque el propio PDF documenta diferencias de presupuesto, parada, lote físico, resolución, aumento y agregación.

**Evidencia:** la Tabla 7 deja `diffusion_policy` como «bifurcación propia» sin versión. El repositorio solo aparece en página 43. Dos remisiones a configuración y versiones son `??`.

**Por qué es problemático:** una URL mutable no identifica el estado ejecutado. La frase causal contradice las salvedades metodológicas del Apartado 3.5 y puede inducir a una lectura más fuerte que la permitida.

**Impacto sobre resultados:** la repetición exacta y la auditoría de correspondencia son `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` La afirmación de control experimental queda formalmente contradicha por el propio documento.

**Corrección necesaria:** fijar versión inmutable, hashes, comandos, configuraciones, dependencias y artefactos; corregir la frase causal para enumerar los factores no controlados.

**Evidencia necesaria para considerarlo resuelto:** paquete reproducible versionado y enlazado desde la memoria, con manifest, checksums, instrucciones y resultados de una repetición limpia.

### Hallazgo M11. La «degradación de la representación» no se demuestra

**Ubicación:** página 31, Apartado 5.1, conclusión sobre V2.

**Tipo:** error de interpretación; conclusión que excede la evidencia.

**Descripción:** la memoria concluye que la actualización de pesos «degrada la representación de partida sin construir otra más útil». Los resultados disponibles son pérdida de ruido, pérdida de validación, error de acción y puntuación de control de una sola ejecución.

**Evidencia:** V2 sobreajusta según las curvas y no supera a V1, pero obtiene menor error cuadrático de acción que todas las variantes, página 26.

**Por qué es problemático:** no se mide calidad representacional, geometría del embedding, transferencia ni información retenida. Un descenso de rendimiento de control no identifica el mecanismo.

**Impacto sobre resultados:** la explicación causal puede ser falsa aunque el rendimiento observado sea correcto.

**Corrección necesaria:** presentar la degradación como hipótesis explicativa o añadir análisis de representaciones y réplicas que la prueben.

**Evidencia necesaria para considerarlo resuelto:** métricas representacionales, sondas o ablaciones que vinculen el cambio de representación con el rendimiento, replicadas entre semillas.

## 9. Observaciones menores

### Hallazgo N1. Marcadores de referencia sin resolver

**Ubicación:** página 17, Apartado 3.5, «se recoge en el ??»; página 20, Apartado 3.8, «se documentan en el ??».

**Tipo:** problema de redacción y estructura académica.

**Descripción:** dos referencias cruzadas aparecen como `??`.

**Evidencia:** texto visible en ambas páginas.

**Por qué es problemático:** impide localizar la configuración efectiva y las versiones justo en pasajes de reproducibilidad.

**Impacto sobre resultados:** no invalida métricas, pero reduce trazabilidad y presenta un documento no finalizado.

**Corrección necesaria:** reparar las referencias cruzadas y comprobar todas las remisiones.

**Evidencia necesaria para considerarlo resuelto:** PDF recompilado sin apariciones de `??` y enlaces dirigidos al apartado o tabla correctos.

### Hallazgo N2. Agradecimientos sin sustituir

**Ubicación:** página iii, página PDF 5, Agradecimientos.

**Tipo:** problema formal y de acabado.

**Descripción:** permanece el texto de plantilla entre corchetes en lugar de agradecimientos o de una página omitida.

**Evidencia:** «[Texto de agradecimientos. Esta pagina es opcional...]».

**Por qué es problemático:** la memoria se presenta como versión final, pero conserva una instrucción editorial.

**Impacto sobre resultados:** ninguno científico; afecta la presentación académica.

**Corrección necesaria:** redactar el apartado o eliminarlo junto con su entrada de índice.

**Evidencia necesaria para considerarlo resuelto:** preliminares sin texto de plantilla.

### Hallazgo N3. Referencias y Bibliografía siguen sistemas distintos

**Ubicación:** páginas 39 a 41, Referencias; página 43, Bibliografía; índice, páginas iv a v.

**Tipo:** problema de citación y estructura académica.

**Descripción:** las 29 fuentes académicas aparecen numeradas en IEEE, mientras una monografía del autor y el repositorio se separan en una Bibliografía no numerada. No se observa una cita numérica a esas dos entradas.

**Evidencia:** separación visible en el índice y páginas finales.

**Por qué es problemático:** mezcla dos lógicas bibliográficas y dificulta la correspondencia cita-entrada.

**Impacto sobre resultados:** no altera los experimentos, pero reduce consistencia formal y trazabilidad del repositorio.

**Corrección necesaria:** integrar todo recurso citado en una lista coherente o explicar la función documental de una sección separada según la norma institucional.

**Evidencia necesaria para considerarlo resuelto:** lista final uniforme y cada entrada citada desde el cuerpo.

### Hallazgo N4. La pregunta de investigación queda implícita

**Ubicación:** página 1, Apartado 1.2; página 11, Apartado 3.1.

**Tipo:** problema de estructura académica.

**Descripción:** se enuncian dos cuestiones y tres hipótesis, pero no una pregunta formal que defina comparadores, resultado y alcance.

**Evidencia:** el Apartado 1.2 usa formulación narrativa y el 3.1 introduce directamente las hipótesis.

**Por qué es problemático:** dificulta comprobar si el diseño responde exactamente a una pregunta confirmatoria o a una exploración.

**Impacto sobre resultados:** menor, aunque favorece el desplazamiento entre «bloque», «codificador», «estrategia» y «familia».

**Corrección necesaria:** formular una pregunta principal y, si procede, subpreguntas separadas para rendimiento y coste.

**Evidencia necesaria para considerarlo resuelto:** pregunta explícita alineada con estimando, unidad experimental y conclusiones.

### Hallazgo N5. Terminología no homogeneizada en la Figura 4

**Ubicación:** página 29, Figura 4.

**Tipo:** problema de visualización y redacción.

**Descripción:** los paneles conservan `Start`, `End`, `Reward curve`, `Reward` y `Environment step` en inglés, mientras el cuerpo y pie están en español y la métrica se denomina puntuación de cobertura.

**Evidencia:** rótulos visibles en ambos paneles.

**Por qué es problemático:** introduce una denominación distinta para la misma métrica y reduce uniformidad.

**Impacto sobre resultados:** ninguno metodológico.

**Corrección necesaria:** traducir rótulos y emplear «puntuación» o «cobertura» de forma consistente.

**Evidencia necesaria para considerarlo resuelto:** figura regenerada con terminología uniforme.

No se identifican otras observaciones menores que puedan sostenerse exclusivamente con el PDF sin convertir preferencias editoriales en requisitos.

## 10. Auditoría metodológica

### Diseño

El estudio es comparativo y experimental respecto a la configuración ejecutada, pero no constituye un experimento de un solo factor en toda la matriz. V1 frente a V2 ofrece el contraste mejor controlado: misma ResNet-18, mismos pesos iniciales, resolución, normalización y agregación, con congelación frente a ajuste fino. V0 frente a V1/V2 es una comparación de bloques completos. V0 frente a V3/V4 añade arquitectura. Esta distinción aparece en el PDF y debe gobernar toda interpretación.

El procedimiento de selección es adaptativo. Se observan curvas, se define una parada durante la campaña, se aplican decisiones por inspección y se interrumpe V3 fuera de la regla. No existe un protocolo único aplicado prospectivamente. El análisis contrafactual de la página 22 muestra que V0 conservaría casi el mismo valor con otra parada, pero no elimina la asimetría ni permite conocer el comportamiento no observado de V3.

### Muestra y unidades experimentales

Hay tres unidades distintas:

1. Episodio de demostración para construir ventanas de entrenamiento.
2. Condición inicial y trayectoria de inferencia para evaluar un checkpoint fijo.
3. Ejecución completa de entrenamiento para comparar estrategias.

El TFM dispone de 90, 50 y 1 unidades por variante en esos niveles, respectivamente. La inferencia principal se formula en el tercer nivel, pero la replicación solo existe en el segundo. Tratar 50 episodios como soporte de una afirmación sobre estrategias constituye pseudorreplicación respecto al proceso de entrenamiento.

### Variables y confusores

La variable declarada es el bloque visual. Si esa definición se mantiene, la comparación de bloques es descriptivamente coherente. Si se pretende atribuir el efecto al preentrenamiento, a la familia o a la estrategia, resolución, recorte, agregación y arquitectura actúan como confusores. El lote efectivo se mantiene en 64, pero el lote físico y número de pasadas cambian; esto importa para el coste, aunque no implica necesariamente un sesgo de rendimiento.

### Procedimiento y sesgos

El sesgo dominante es la selección repetida sobre 50 condiciones. También existe una ventaja por oportunidades de evaluación para V0/V1 y una desventaja potencial por truncamiento para V3. La selección de 90 episodios con semilla 42 es común a variantes y evita que el split sea un confusor entre ellas. Sin embargo, un único split no permite saber si la ordenación depende de la composición del entrenamiento.

### Validez interna

La validez interna es insuficiente para conclusiones confirmatorias. La diferencia V0 frente a las demás es grande y estable en las últimas evaluaciones según la Figura 2, lo cual aporta evidencia descriptiva moderada. No corrige la reutilización del conjunto, la ausencia de réplicas ni el confusor del bloque. La afirmación válida es sobre los artefactos observados, no sobre el efecto aislado de una decisión.

### Validez externa

La evidencia cubre una tarea sintética, una cámara cenital, una dinámica bidimensional, una cantidad de demostraciones, una red generativa y un equipo. No cubre robots físicos, otras tareas, otras distribuciones visuales, robustez, diferentes cantidades de datos ni otros generadores. La memoria reconoce buena parte de este límite. Cualquier generalización a «codificadores visuales para control robótico» excede la evidencia.

## 11. Auditoría de datos

### Procedencia y trazabilidad

El PDF identifica el dataset como el publicado por los autores de Diffusion Policy, en formato Zarr, con 206 demostraciones y 25.650 transiciones. Describe variables y tamaño aproximado. No muestra URL directa, versión, checksum, licencia ni fecha de descarga. La correspondencia entre el archivo usado y el original es `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.`

El autor declara reproducir el primer instante mediante semillas 0 a 205. Este control aporta una afirmación de trazabilidad, pero no permite verificar imágenes, acciones, contactos, orden temporal o integridad completa desde el PDF.

### Calidad, limpieza e integridad

Se informan longitudes, puntuaciones, posiciones y orientaciones. No se documentan comprobaciones de valores ausentes, episodios duplicados, transiciones corruptas, rangos imposibles, continuidad de acciones o contactos anómalos. Su presencia o ausencia es `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` No debe asumirse que un dataset de referencia carece de estos problemas.

### Partición

La partición se realiza a nivel de episodio, lo cual evita mezclar ventanas de un mismo episodio entre entrenamiento y validación según la descripción. La semilla 42 es común. Este diseño es mejor que un split aleatorio de ventanas. No obstante, 4 episodios de validación proporcionan una muestra inestable y los 112 descartados reducen innecesariamente la información disponible para explorar el régimen de pocas demostraciones.

### Representatividad

La Figura 1 compara algunas distribuciones marginales. Los contrastes U y KS no demuestran equivalencia. La selección aleatoria uniforme hace razonable afirmar que no se aplicó un filtro deliberado por calidad; no permite afirmar que el subconjunto representa en todas las dimensiones a las 206 demostraciones. Las 50 semillas de evaluación parecen proceder del mismo generador, pero su representatividad de toda la población de condiciones no se justifica mediante un esquema de muestreo formal.

### *Leakage*

No se observa solapamiento de semillas entre demostraciones 0 a 205 y evaluación 100.000 a 100.049 según el PDF. Tampoco se describe normalización calculada sobre datos agregados del dataset; las normalizaciones de ImageNet son externas. En cambio, sí existe *leakage* de selección: la evaluación reservada guía la elección de checkpoint y después se usa como resultado final. La ausencia de otras formas de *leakage* en el código es `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.`

## 12. Auditoría estadística

### Estadística descriptiva

La memoria presenta medias, desviaciones, errores estándar, medianas e intervalos según el indicador. La métrica principal está acotada entre 0 y 1 y toma el máximo temporal. La media responde a cobertura promedio de condiciones, mientras la distribución puede acumular masa en 1 por el umbral de éxito. No se informa cuántos episodios alcanzan 1 por variante, dato útil para interpretar la forma de la distribución.

### Wilcoxon pareado

El emparejamiento por condición inicial es adecuado en principio porque cada política afronta las mismas semillas. La prueba de rangos con signo evita exigir normalidad de diferencias, pero supone pares independientes y una distribución de diferencias compatible con su interpretación. Con 50 pares, la aproximación normal puede ser suficiente si se tratan correctamente empates y ceros. El PDF no especifica estos detalles ni la corrección de continuidad.

El problema principal no es la elección nominal de Wilcoxon, sino aplicarlo a checkpoints seleccionados con esos mismos pares. Los valores p no incorporan la búsqueda del máximo. Además, si el ruido de difusión no se sincroniza, el par combina la diferencia de política con dos realizaciones aleatorias distintas.

### Intervalos de confianza

La Tabla 10 no identifica si los IC corresponden a la media mediante aproximación normal, *bootstrap* pareado, percentiles u otro método. Tampoco se explica por qué el intervalo estima una diferencia media mientras el contraste usa rangos. La cobertura nominal del 95 % es `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` Los IC se calculan después de selección y no son simultáneos.

### Corrección de Holm y multiplicidad

Holm controla el error familiar de los diez contrastes por pares, condicionado a una familia fijada y datos no usados para elegir modelos. La familia explícita son las diez comparaciones de cinco checkpoints. No controla:

1. Las 4 a 10 evaluaciones temporales consideradas antes de elegir cada checkpoint.
2. Las decisiones de parada adoptadas tras inspeccionar curvas.
3. Los intervalos múltiples, que la tabla reconoce no ajustados.
4. Las comparaciones descriptivas adicionales de pérdidas, latencia, memoria y error de acción.
5. La selección de hipótesis o explicaciones después de observar resultados.

Por tanto, Holm está bien aplicado a una familia limitada, pero no resuelve la multiplicidad del procedimiento completo.

### Tamaño de efecto, potencia y equivalencia

Las diferencias absolutas de puntuación son tamaños de efecto interpretables y se reportan. No se define un umbral de relevancia práctica ni se informa potencia. En V1 frente a V2, el IC `[-0,093; 0,133]` admite diferencias relevantes en ambas direcciones. La no significación no demuestra igualdad. H3 usa la palabra «comparable», que requeriría un margen de equivalencia o no inferioridad fijado antes de analizar.

### Alcance válido de la inferencia

Incluso si no existiera selección, los 50 pares permitirían inferir sobre condiciones iniciales para esos modelos fijos bajo un esquema de muestreo defendible. No permitirían inferir sobre el algoritmo vuelto a entrenar. Esa inferencia necesita varias ejecuciones por variante.

## 13. Auditoría de Machine Learning

### Dataset y split

El split por episodio es apropiado para evitar ventanas relacionadas entre entrenamiento y validación. La evaluación en simulación usa semillas distintas de las demostraciones. Sin embargo, el bloque de evaluación se convierte en validación de checkpoint y no queda test final. Cuatro episodios no bastan para una validación estable de la pérdida.

### Baseline

V0 es una línea base pertinente porque reproduce la configuración visual de referencia declarada para Diffusion Policy. El estudio también contextualiza resultados previos de ResNet y CLIP. No se requiere una política más simple para responder a la ablación concreta, aunque una referencia de comportamiento no aprendido o de clonación simple habría ayudado a calibrar la dificultad, no a resolver la pregunta principal.

### Arquitecturas e hiperparámetros

Se especifican horizonte, pasos de observación y acción, DDPM, optimizador, tasa, calentamiento, EMA, lote y semilla. Faltan detalles completos del planificador, regularización efectiva, *weight decay*, transformaciones exactas, identificadores de modelos timm, configuración de capas de proyección y parámetros de parada en un formato reproducible. La correspondencia entre descripción y ejecución es `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.`

### Selección y sobreajuste

Las curvas muestran disminución de pérdida de entrenamiento y aumento de validación, junto con descenso de puntuación en V1 a V4. Esto es compatible con sobreajuste. V0 demuestra que la pérdida de ruido de cuatro episodios no sigue necesariamente el rendimiento de control. Seleccionar por la puntuación de las 50 condiciones es razonable como validación, pero invalida usarlas después como test.

### Métricas

La cobertura máxima es pertinente para Push-T y se define con claridad. El uso del máximo temporal coincide con la posibilidad de desalineación posterior. La métrica no caracteriza éxito sostenido, estabilidad de trayectoria, colisiones, eficiencia de acción ni robustez. Estas dimensiones quedan fuera del objetivo declarado y no son errores por sí mismas. La tasa de éxito por variante habría facilitado la interpretación de la masa en 1.

### Robustez e interpretabilidad

No se evalúan perturbaciones visuales, cambio de dinámica, otras cámaras, otras tareas ni condiciones fuera de distribución. El Apartado 1.4 lo reconoce. La explicación sobre descriptores globales y spatial softmax es plausible, pero no se prueba mediante análisis de atención, mapas de activación o ablaciones.

### Complejidad frente a utilidad

Los modelos preentrenados no aportan mejora observada y aumentan coste por época. El estudio responde bien a la prueba de simplicidad del agente dentro de los artefactos evaluados: la línea base más simple es también la mejor observada. La superioridad esperada de V0 al volver a entrenar o en otros dominios sigue sin demostrarse.

## 14. Auditoría de validación

| Función | Datos usados | Evaluación |
|---|---|---|
| Entrenamiento | 90 episodios, 10.726 ventanas | Descrito; corrección de la ejecución no verificable desde el PDF. |
| Validación de pérdida | 4 episodios, 404 ventanas | Insuficiente para estabilidad episódica. Las ventanas no son independientes. |
| Selección de checkpoint | 50 condiciones reservadas, repetidas cada 50 épocas | Adecuado como validación si se usa solo para seleccionar. |
| Prueba final | Ninguna | Fallo crítico. |
| Validación entre semillas de entrenamiento | Una semilla por variante | Ausente. |
| Repeticiones de inferencia | No especificadas | Ausentes o no documentadas. |
| Validación externa | Ninguna tarea adicional ni robot físico | Ausente y reconocida. |

La estrategia no responde satisfactoriamente a «¿cómo sabemos que el modelo funcionará en datos no observados?». Las semillas no vistas durante el ajuste de pesos sí son datos nuevos respecto al entrenamiento, pero dejan de ser nuevos respecto a la selección del checkpoint después de consultarlas repetidamente. La media de las tres últimas evaluaciones reduce la dependencia de un máximo puntual y muestra que V0 permanece por encima, pero sigue usando las mismas condiciones y no constituye test.

La validación mínima necesaria debe incluir tres niveles: validación para selección de checkpoint, test final de condiciones no consultadas y varias ejecuciones de entrenamiento para estimar estabilidad de cada estrategia. Si se desea generalizar más allá de Push-T sintético, se necesitan tareas o dominios externos.

## 15. Auditoría de reproducibilidad

**Clasificación: Parcial.**

### Información presente

1. Hardware y sistema operativo, Tabla 6.
2. Versiones de las bibliotecas principales, Tabla 7.
3. Semilla global 42 y semilla de split 42.
4. Cantidades de episodios, ventanas y actualizaciones.
5. Arquitecturas, estrategias y conteos de parámetros.
6. Hiperparámetros principales, Tabla 5.
7. Frecuencia de checkpoints, uso de EMA y criterios adoptados.
8. Intervalo declarado de semillas oficiales de evaluación.
9. URL de un repositorio en la página 43.
10. Declaración expresa de que no se activó determinismo algorítmico.

### Información ausente o insuficiente

1. Commit, etiqueta o hash inmutable del código ejecutado.
2. Hash, versión y URL directa del dataset.
3. Identificadores exactos y checksums de pesos preentrenados.
4. Comandos de instalación, entrenamiento, reanudación, evaluación y análisis.
5. Nombres y contenido de ficheros Hydra usados en cada variante.
6. Archivo completo de dependencias o entorno bloqueado.
7. Configuración completa del planificador DDPM.
8. Semillas y repeticiones de inferencia.
9. Checkpoints seleccionados y artefactos de resultados.
10. Procedimiento exacto de reconstrucción de V2 tras la reanudación.
11. Código o especificación del cálculo de IC y pruebas.
12. Evidencia de una ejecución desde un entorno limpio.

### Prueba de reproducibilidad mental

Un tercero podría reconstruir una implementación aproximada y un experimento conceptualmente similar. No podría identificar de forma inequívoca los bytes de datos, pesos, código, configuraciones y checkpoints usados. Tampoco podría reproducir el muestreo estocástico ni el análisis estadístico exacto. El repositorio se declara disponible, pero su contenido y concordancia son `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` Por ello, la reproducibilidad no alcanza la categoría «Adecuada».

## 16. Evaluación del estado del arte

### Calidad y cobertura

El Capítulo 2 cubre aprendizaje por imitación, modelos de difusión, Diffusion Policy, extensiones, ResNet, ViT, DINOv2, CLIP y representaciones robóticas. Las 29 referencias numeradas incluyen artículos de revista, conferencias, documentación de software y varios preprints. La mayoría son fuentes primarias y se vinculan a afirmaciones concretas.

### Actualidad

El corpus combina fundamentos de 2011 a 2021 con trabajos de 2022 a 2025. La presencia de referencias clásicas está justificada por su función conceptual. Se incorporan antecedentes recientes como DINOv3-DP de 2025. La exactitud bibliográfica y vigencia externa de cada entrada son `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.`

### Comparabilidad y síntesis

La Tabla 1 compara estrategias en Square y la Tabla 2 clasifica extensiones por componente. Los Apartados 2.5.4 a 2.6 no se limitan a enumerar autores: contrastan tarea, modalidad, adaptación y dominio. Esta síntesis es una fortaleza.

### Gap

El hueco está formulado con precisión: no se identifica una comparación conjunta de cinco configuraciones concretas bajo una Diffusion Policy común en Push-T. Es un gap experimental verificable dentro del corpus revisado, no una necesidad universal. Su relevancia es incremental y depende de que el diseño realmente aísle factores. Como el experimento cambia preprocesado y aplica protocolos desiguales, el trabajo cubre una comparación de bloques, pero no cierra el gap causal sobre inicialización o estrategia.

### Limitaciones del estado del arte

No se describe búsqueda, bases consultadas, criterios de inclusión, fecha de corte ni evaluación de calidad. No era obligatorio realizar una revisión sistemática para este TFM, pero sin estrategia no puede asegurarse exhaustividad. La monografía del autor y el repositorio aparecen en una Bibliografía separada, sin integración clara en el sistema IEEE.

## 17. Resultados y discusión

### Lo que muestran los datos

1. Entre los checkpoints seleccionados, V0 obtiene 0,864 y las otras cuatro variantes entre 0,535 y 0,668, Tabla 9.
2. V0 mantiene puntuaciones próximas a 0,86 durante gran parte de la curva observada, mientras V1 a V4 retroceden después de sus máximos, Figura 2.
3. En los mismos 50 episodios usados para selección, las diferencias V0 frente a V1 a V4 son positivas y los p de Wilcoxon ajustados por Holm son iguales o inferiores a 0,0030, Tabla 10.
4. Los IC entre variantes preentrenadas son amplios y atraviesan cero salvo V1 frente a V4 antes de considerar simultaneidad; su p Holm es 0,0501.
5. La pérdida de entrenamiento disminuye y la de validación, calculada sobre cuatro episodios, aumenta en fases posteriores, Figura 3.
6. V0 requiere 1,6 min por época; las restantes, entre 2,3 y 14,7 min, Tabla 8.
7. El codificador aislado varía de 2,7 a 16,1 ms con lote unitario, pero la llamada completa varía alrededor de 1,5 %, Tabla 11.
8. V2 reserva más memoria que la capacidad física declarada; la inferencia de todas las variantes queda por debajo del 23 %, Tabla 12.

### Lo que afirma el autor

El autor afirma que ninguno de los bloques preentrenados mejora a V0, que el ajuste fino no supera a la congelación, que los ViT no muestran ventaja propia y que V0 domina en rendimiento y coste. También propone que V2 degrada la representación, que el descarte no introduce sesgo apreciable y que el sobrecoste de los transformadores existe en el codificador pero no altera apreciablemente la política.

### Lo que puede concluirse legítimamente

1. Para las cinco ejecuciones concretas y el conjunto de selección observado, V0 produjo la mayor puntuación media y el menor tiempo por época.
2. No se detectó una diferencia V1 frente a V2 en esos checkpoints y episodios; el intervalo es compatible con efectos relevantes en ambas direcciones.
3. La diferencia de latencia entre codificadores queda dominada por los 100 pasos de la red generativa en este hardware y configuración.
4. Las variantes preentrenadas no ofrecieron una ventaja observada que compensara su mayor coste en esta campaña.
5. Las curvas son compatibles con sobreajuste, especialmente V1 a V4, pero la pérdida de cuatro episodios no identifica por sí sola el mecanismo.
6. La ventaja observada de V0 corresponde a su bloque completo, no al preentrenamiento aislado.

### Lo que todavía no está demostrado

1. El rendimiento final de los checkpoints en condiciones ajenas a su selección.
2. La estabilidad de la ordenación frente a nuevas semillas de entrenamiento.
3. La superioridad de entrenar desde cero como estrategia.
4. La equivalencia de V1 y V2 o de CNN y ViT.
5. Que la representación de V2 se haya degradado.
6. Que el descarte de 112 episodios no produzca diferencias relevantes.
7. Que las 50 semillas representen la población de Push-T.
8. Robustez, generalización a otras tareas, imágenes reales o robot físico.
9. Correspondencia exacta entre métodos descritos y código. `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.`

## 18. Cumplimiento de objetivos

### Objetivo específico 1

**Evaluación:** parcialmente cumplido.

Se documentan cinco variantes y todas producen resultados. La salida de 512 componentes y la red generativa común están descritas. No se mantiene constante la configuración experimental en sentido estricto: cambian presupuesto, parada, lote físico y componentes de preprocesado. La corrección funcional de la implementación es `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.`

### Objetivo específico 2

**Evaluación:** parcialmente cumplido.

La comparación descriptiva está realizada y las métricas responden a la tarea. No es una comparación final válida porque las condiciones seleccionan checkpoints y después se tratan como resultado. Tampoco estima la variación entre entrenamientos. El objetivo se cumple como actividad, pero no como demostración robusta de diferencias entre estrategias.

### Objetivo específico 3

**Evaluación:** cumplido dentro del alcance declarado.

Parámetros, tiempo por época, tiempo total, latencia y memoria se presentan para las cinco variantes. Se distinguen correctamente indicadores no comparables, como tiempo total con presupuestos distintos. El coste se generaliza solo al hardware, versiones y configuración descritos.

### Objetivo general

**Evaluación:** parcialmente cumplido.

Se observa cómo cinco bloques concretos se comportaron en una campaña. No se identifica la influencia aislada del codificador ni de la estrategia y no se estima su estabilidad. El término «influencia» debe reemplazarse por «comparación exploratoria» o respaldarse con ablaciones, réplicas y test independiente.

## 19. Evaluación de conclusiones

### Conclusiones individualizadas

| Conclusión formulada o implícita | Clasificación | Evaluación |
|---|---|---|
| Se implementaron y evaluaron cinco variantes dentro de una misma política. | **NO VERIFICABLE** en su corrección; coherente con el PDF. | Las tablas y curvas acreditan cinco ejecuciones declaradas. La equivalencia real de infraestructura y código es `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` |
| Ningún bloque preentrenado mejora a V0 en Push-T con 90 demostraciones. | **PARCIALMENTE SUSTENTADA.** | Es cierta para los máximos observados de cinco ejecuciones en el conjunto de selección. No es una estimación final ni estable entre entrenamientos. |
| V0 supera significativamente a V1 a V4. | **NO SUSTENTADA** como inferencia confirmatoria; descriptivamente observada. | Wilcoxon y Holm se calculan después de seleccionar los checkpoints con esos mismos episodios. |
| El ajuste fino no supera a la congelación. | **PARCIALMENTE SUSTENTADA.** | No se detecta ventaja en una pareja de ejecuciones. No demuestra ausencia de ventaja esperada ni equivalencia. |
| La congelación tampoco supera al ajuste fino. | **PARCIALMENTE SUSTENTADA.** | El intervalo contiene cero, pero es amplio. La formulación válida es «no se detectó diferencia». |
| Los transformadores no ofrecen ventaja de rendimiento propia. | **PARCIALMENTE SUSTENTADA.** | No mejoran al artefacto V0 observado, pero arquitectura, agregación y preprocesado están confundidos; una sola ejecución no identifica efecto de familia. |
| V2 degrada su representación de partida. | **NO SUSTENTADA.** | No se analiza la representación. Las curvas solo muestran desempeño y pérdidas de una ejecución. |
| V0 es menos costosa por época que las alternativas. | **SUSTENTADA** para el equipo y protocolo descritos. | Tabla 8. El coste total no es comparable, pero el tiempo unitario sí se mide. |
| Congelar el codificador no garantiza entrenamiento más barato. | **SUSTENTADA** en las configuraciones evaluadas; **EXCEDE EL ALCANCE** si se universaliza. | V1, V3 y V4 son congeladas y más lentas por época que V0. Otros factores también cambian. |
| El codificador apenas modifica la latencia completa de la política con 100 pasos. | **SUSTENTADA** para el hardware y software de la Tabla 11. | La llamada completa varía 1,5 % y el componente generativo domina. |
| Ninguna variante alcanza tiempo real en el equipo. | **SUSTENTADA** bajo la definición usada. | Ocho acciones tardan alrededor de 1,7 s frente a 0,8 s de ejecución simulada. No se generaliza a otro hardware o muestreador. |
| La ventaja de V0 debe atribuirse al bloque visual junto con su preprocesado. | **PARCIALMENTE SUSTENTADA.** | Es una restricción causal correcta frente a atribuirla a la inicialización. Aun así, presupuestos, parada y selección también difieren. |
| La familia del codificador no explica por sí sola las diferencias. | **PARCIALMENTE SUSTENTADA.** | El diseño no permite atribución a la familia; no demuestra que la familia carezca de efecto. |

### Matriz de trazabilidad de conclusiones

| Conclusión | Resultado que la sustenta | Método | Evidencia | Validez |
|---|---|---|---|---|
| V0 obtiene la mayor puntuación observada. | Tabla 9 y Figura 2. | Máximo entre evaluaciones periódicas de 50 condiciones. | **Moderada** como descripción. | Válida para los artefactos y conjunto de selección; sesgada como estimación final. |
| V0 mantiene ventaja al final. | Media de tres últimas: 0,862 frente a 0,483 a 0,585, página 23. | Promedio descriptivo de evaluaciones finales. | **Moderada.** | Reduce dependencia de un pico, pero reutiliza las mismas condiciones. |
| V0 difiere de V1 a V4. | Tabla 10. | Diferencias pareadas, Wilcoxon y Holm. | **Débil** como inferencia confirmatoria. | Selección e inferencia usan los mismos datos; no hay réplicas de entrenamiento. |
| V1 y V2 no muestran diferencia detectada. | Diferencia 0,020, IC `[-0,093; 0,133]`, p = 0,82. | Un contraste pareado de checkpoints seleccionados. | **Débil.** | No demuestra equivalencia y el IC admite efectos relevantes. |
| Las variantes preentrenadas sobreajustan. | Figura 3 y descensos de Figura 2. | Seguimiento de pérdidas y puntuación. | **Débil a moderada.** | Compatible con sobreajuste; validación de cuatro episodios y selección adaptativa. |
| V0 tiene menor coste por época. | Tabla 8. | Cronometría de ejecuciones. | **Moderada.** | Válida para la plataforma; V2 incluye estimación parcial. |
| ViT aumenta coste del codificador, no de la llamada completa. | Tabla 11. | 50 repeticiones tras 20 calentamientos. | **Moderada a fuerte** para ese equipo. | Protocolo claro; IQR completos no se muestran por variante, pero se acota su máximo. |
| El ajuste fino de V2 agota memoria física. | Tabla 12. | 12 pasos y memoria reservada. | **Moderada.** | Describe el asignador y entorno indicados; no equivale necesariamente a memoria residente física. |
| La ventaja observada pertenece al bloque completo. | Cambios enumerados en Apartado 3.4. | Análisis del diseño. | **Fuerte** como límite interpretativo. | Impide atribuir causalidad a inicialización, pero no identifica cuál factor domina. |
| Los resultados no se generalizan a otros dominios. | Una tarea sintética y ausencia de validación externa. | Prueba de generalización. | **Fuerte.** | Restricción necesaria y coherente con el alcance. |

## 20. Limitaciones

### Reconocidas por el autor

1. Una sola semilla de entrenamiento, páginas 2, 11, 18 y 29.
2. Presupuestos desiguales y distinto número de evaluaciones, páginas 16 a 17 y 29.
3. V3 interrumpida antes de su presupuesto y del mínimo de parada, páginas 17 y 29.
4. Reutilización de las 50 condiciones para selección y ausencia de prueba final, páginas 18 y 30.
5. Evaluación temporal discreta cada 50 épocas, página 30.
6. Reanudación de V2 y extrapolación parcial de tiempos, páginas 21 y 23.
7. Factores confundidos entre V0 y variantes preentrenadas, páginas 15 a 16 y 31 a 32.
8. Ausencia de robot físico y robustez visual, página 2.
9. No determinismo de bibliotecas, página 21.
10. Limitación de memoria que impide ajustar los ViT, páginas 16 y 19 a 20.

El reconocimiento es detallado y constituye una fortaleza de transparencia. No basta, sin embargo, para convertir resultados sesgados en evidencia válida.

### Detectadas por el tribunal

1. Los IC y p de los checkpoints seleccionados no corrigen la selección sobre esos mismos episodios.
2. Cuatro episodios de validación son insuficientes y sus ventanas no son unidades independientes.
3. La estocasticidad de inferencia no se documenta ni se separa de la variación de condiciones.
4. La no significación de U/KS no demuestra ausencia de sesgo ni equivalencia de subconjuntos.
5. La hipótesis de comparabilidad de arquitecturas no se prueba mediante equivalencia o no inferioridad.
6. El estimando y la población de generalización no se definen.
7. Las Ecuaciones (3) y (4) y la configuración DDPM son insuficientes para reproducir el muestreo.
8. Faltan identificadores inmutables, comandos, hashes, configuraciones y artefactos.
9. La explicación de degradación de representación no está medida.
10. No se informa sensibilidad al split de 90 episodios.
11. No se separa rendimiento del checkpoint fijo de rendimiento esperado del algoritmo entrenado de nuevo.
12. No se analiza la representatividad conjunta de las 50 semillas ni se justifica que un intervalo consecutivo sea muestra aleatoria del generador.

## 21. Cumplimiento formal

### Cumplimiento formal institucional verificable en el PDF

| Requisito | Estado | Evidencia |
|---|---|---|
| Portada | Cumple. | Página PDF 1: centro, tipo, máster, título, autor, director, lugar y fecha. |
| Primera hoja | Cumple con una salvedad. | Página PDF 3: título, propósito, autor y director; los campos de firma aparecen sin firma visible. Si la firma es obligatoria, `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.` |
| Agradecimientos | No cumple como versión final. | Página iii mantiene texto de plantilla. El apartado es opcional, por lo que puede eliminarse. |
| Índice de contenidos | Cumple. | Páginas iv y v. |
| Índice de figuras | Cumple. | Página vii. |
| Índice de tablas | Cumple. | Página ix. |
| Resumen | Cumple en idioma y extensión aparente. | Página xi; claramente inferior a 500 palabras. |
| Abstract | Cumple en idioma y extensión aparente. | Página xiii; inglés y menos de 500 palabras. |
| Cuerpo | Cumple. | Seis capítulos numerados. |
| Presupuesto | Cumple. | Capítulo 6, páginas 35 a 38. |
| Referencias | Cumple parcialmente. | Referencias IEEE numeradas, páginas 39 a 41; Bibliografía separada en página 43. |
| Anexos | No incluidos. | Los requisitos enumeran anexos, pero no se puede afirmar que sean obligatorios cuando no hay contenido anexo. No se penaliza. |
| Jerarquía | Cumple. | Numeración hasta tercer nivel en 2.5.1 a 2.5.5, sin niveles huérfanos evidentes. |
| Figuras y tablas | Cumple en numeración, pies, fuente e índices. | Cuatro figuras y diecisiete tablas. Figura 4 conserva rótulos ingleses. |
| Remisiones internas | No cumple plenamente. | Marcadores `??` en páginas 17 y 20. |

### Normas no verificables directamente

El PDF muestra proporción vertical compatible con A4 y texto visualmente justificado, pero no se dispone de medición técnica de tamaño de página, márgenes, fuente, cuerpo de 11 puntos e interlineado. Por tanto: `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.`

Tampoco puede determinarse solo por apariencia si todas las tablas se insertaron como estructura textual o como imagen, aunque el texto extraíble indica que gran parte del contenido es seleccionable. El cumplimiento exacto de la plantilla Tecnun, firmas, depósito y requisitos administrativos es `NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.`

### Separación entre forma y ciencia

Los defectos formales son corregibles y no determinan el dictamen metodológico. A la inversa, una maquetación cuidada no compensa la ausencia de test independiente ni de réplicas de entrenamiento.

## 22. Correcciones prioritarias

### PRIORIDAD 1: imprescindibles antes de defensa

1. Congelar los checkpoints ya seleccionados y evaluarlos una sola vez sobre un bloque disjunto de semillas nunca consultadas. No volver a seleccionar después de ver ese resultado.
2. Repetir los entrenamientos con varias semillas independientes bajo un protocolo idéntico. Si no es viable, reducir el objetivo y todas las conclusiones a una comparación exploratoria de cinco ejecuciones concretas, sin inferencia sobre estrategias.
3. Predefinir y aplicar el mismo presupuesto, frecuencia de evaluación y regla de parada. Completar V3 conforme al protocolo o excluir afirmaciones que dependan de su entrenamiento completo.
4. Recalcular IC y contrastes sobre la evaluación final. Separar la variación entre entrenamiento, condición inicial e inferencia estocástica.
5. Reformular la conclusión principal: no afirmar superioridad significativa de estrategias con los valores p actuales.
6. Corregir Ecuaciones (3) y (4) y documentar todos los parámetros del planificador DDPM.
7. Eliminar la afirmación de que cualquier diferencia es atribuible al codificador y la explicación no demostrada sobre degradación de representaciones.

### PRIORIDAD 2: necesarias para mejorar validez

1. Ejecutar ablaciones que igualen resolución, recorte y agregación para separar inicialización, preentrenamiento y arquitectura.
2. Ampliar la validación más allá de cuatro episodios y estimar su incertidumbre por episodio.
3. Definir formalmente el estimando, la unidad experimental y la población de generalización.
4. Documentar o repetir el muestreo estocástico de difusión por condición; usar semillas comunes cuando proceda.
5. Especificar el método de IC, tratamiento de empates y ceros en Wilcoxon, corrección de continuidad y familia exacta de Holm.
6. Para afirmar comparabilidad, fijar un margen de equivalencia o no inferioridad y diseñar el estudio con potencia suficiente.
7. Sustituir «no introduce un sesgo apreciable» por una formulación de ausencia de diferencias detectadas y añadir balance multivariante o sensibilidad a varios splits.
8. Publicar una versión inmutable del paquete experimental con commit, hashes, configuraciones, comandos, entorno y checkpoints.

### PRIORIDAD 3: recomendadas para elevar calidad

1. Sustituir o eliminar el texto de agradecimientos.
2. Resolver los dos marcadores `??` y comprobar referencias cruzadas.
3. Integrar Referencias y Bibliografía en un sistema coherente y citar el repositorio desde el cuerpo.
4. Formular explícitamente la pregunta de investigación.
5. Homogeneizar al español los rótulos de la Figura 4.
6. Añadir tasas de éxito y distribuciones por variante, no solo medias.
7. Incluir un diagrama del flujo entrenamiento, validación, selección y prueba para impedir ambigüedad.
8. Diferenciar en la defensa «resultado observado», «estimación» e «inferencia sobre el algoritmo».

## 23. Preguntas previsibles del tribunal

| Nivel | Pregunta | Por qué la formularía el tribunal | Elementos mínimos de una buena respuesta | Repregunta | Dificultad | Parte del TFM |
|---:|---|---|---|---|---:|---|
| 1 | ¿Cuál es la contribución demostrada, expresada sin generalizar a otros entrenamientos ni tareas? | Comprueba si se distingue la campaña observada de una conclusión general. | Cinco bloques, una tarea, una semilla por variante, V0 mayor puntuación observada y menor coste por época, sin prueba final. | ¿Qué palabra del título o del objetivo cambiaría para reflejar ese alcance? | 1 | Apartados 1.2 a 1.4 y 5.1. |
| 1 | ¿Cuál es exactamente el gap y por qué no queda cubierto por la ablación original ni por DINOv3-DP? | Evalúa dominio del estado del arte y novedad incremental. | Diferencias de tarea, matriz de modelos, estrategias y protocolo; reconocimiento de que el gap es específico. | ¿Qué parte del gap sigue abierta después de su experimento? | 2 | Apartados 2.3, 2.4 y 2.6. |
| 2 | ¿Por qué se usan 90 episodios para entrenamiento, 4 para validación y se descartan 112? | Examina justificación de datos y estabilidad de validación. | Referencia al protocolo base, régimen de pocos datos, independencia por episodio, debilidad de cuatro episodios y necesidad de sensibilidad a split/tamaño. | ¿Qué cambiaría si entrenara con los 202 episodios disponibles? | 2 | Apartado 3.2 y Tabla 3. |
| 2 | ¿Cómo justifica que V3 se detenga antes del mínimo mientras V0 y V1 disponen de diez evaluaciones? | Ataca la comparabilidad y las decisiones posobservación. | Reconocer desviación, oportunidades desiguales, imposibilidad de descartar recuperación y necesidad de protocolo prospectivo común. | ¿Puede seguir afirmando que V3 es peor que V0? | 3 | Apartado 3.5, Tablas 5, 8 y 9. |
| 3 | ¿Qué parámetro estima el IC de la Tabla 10 y cómo se calculó? | El método de IC no aparece en el PDF. | Definir diferencia media, fórmula o *bootstrap*, unidad de muestreo, emparejamiento y efecto de selección. | ¿Por qué ese IC y Wilcoxon no estiman exactamente el mismo parámetro? | 4 | Apartado 3.7 y Tabla 10. |
| 3 | ¿Qué controla Holm y qué multiplicidad queda fuera? | Comprueba comprensión de error familiar. | Diez comparaciones por pares; no controla búsqueda entre checkpoints, decisiones adaptativas, IC ni otras familias. | ¿Cómo diseñaría una inferencia válida después de seleccionar checkpoint? | 4 | Apartado 3.7 y Tabla 10. |
| 4 | ¿Por qué 50 episodios no equivalen a 50 semillas de entrenamiento? | Evalúa unidad experimental y pseudorreplicación. | Episodios condicionan un modelo fijo; semilla de entrenamiento produce otro modelo; necesidad de jerarquía y réplicas. | Si solo pudiera ejecutar tres nuevas campañas, ¿cómo las asignaría y qué podría concluir? | 4 | Apartados 3.1, 3.7 y 4.7. |
| 4 | ¿Qué afirmaciones sobreviven al confusor entre V0 y V1/V2? | Exige separar observación y causalidad. | Ordenación de bloques completos y coste observado; V1 frente a V2 aísla mejor estrategia; no atribución a inicialización, resolución, recorte o spatial softmax. | Diseñe la ablación mínima que separe esos cuatro factores. | 4 | Apartado 3.4 y Conclusiones. |
| 5 | Las mismas 50 condiciones seleccionan el checkpoint y producen p e IC. ¿Por qué eso invalida su interpretación y cómo lo corregiría sin reentrenar? | Es el hallazgo crítico. | Sesgo del máximo, dependencia posselección, número desigual de oportunidades; congelar checkpoints y probar en semillas disjuntas. | Si la ordenación cambia en el test final, ¿qué resultado debe considerarse principal? | 5 | Apartados 3.5 a 3.7, Tabla 9 y Apartado 4.7. |
| 5 | La política es estocástica. ¿Qué semillas de difusión se usaron y qué parte de la varianza de la Tabla 10 procede de ese muestreo? | Detecta una fuente de incertidumbre no documentada. | Reconocer información ausente, distinguir semilla de entorno, inferencia y entrenamiento; proponer repeticiones o números aleatorios comunes. | ¿Sigue siendo válida la prueba pareada si cada variante recibe ruido no sincronizado? | 5 | Ecuación (4), Apartados 3.6 y 3.7. |
| 5 | Escriba la forma exacta de la muestra ruidosa en entrenamiento y de la transición inversa DDPM que ejecutó. | Comprueba consistencia entre Ecuaciones (3), (4) y planificador. | Factores de alfa acumulada, predicción de ruido, media posterior, varianza y calendario beta exacto. | ¿Qué parámetros de `diffusers` faltan en la memoria para reproducirla? | 5 | Ecuaciones (3) y (4), Tabla 5 y Tabla 7. |
| 5 | ¿Por qué p = 0,82 no demuestra que ajuste fino y congelación sean equivalentes? | Examina diferencia entre ausencia de evidencia y equivalencia. | IC amplio, potencia, margen de equivalencia, una sola réplica de entrenamiento y selección previa. | Proponga un margen y explique cómo lo justificaría sin elegirlo después de ver los datos. | 5 | Tabla 10 y Apartados 4.6 y 5.1. |

El conjunto contiene doce preguntas, cubre los cinco niveles y se concentra en los puntos que previsiblemente determinarían la defensa.

## 24. Veredicto final

**¿Está listo para defensa? No todavía.**

La memoria presenta un trabajo técnicamente desarrollado, transparente en varias limitaciones y útil como estudio exploratorio. No está lista como demostración confirmatoria porque el resultado central carece de evaluación final independiente, la inferencia trata episodios como soporte de conclusiones sobre estrategias entrenadas una sola vez y la comparación no aplica un protocolo uniforme. El defecto crítico puede corregirse parcialmente sin reentrenar mediante un test disjunto de los checkpoints congelados. Para sostener conclusiones sobre estrategias, además se requieren réplicas de entrenamiento y comparaciones controladas.

**Respuesta a la pregunta final del tribunal:** **No. El TFM no demuestra todavía, mediante una metodología adecuada, datos suficientemente confiables, análisis válidos y un procedimiento reproducible, que sus resultados respalden las conclusiones formuladas dentro del alcance declarado.** Demuestra una ordenación descriptiva de cinco ejecuciones y varios costes en una configuración concreta. No demuestra rendimiento final no sesgado, estabilidad entre entrenamientos ni efectos aislados del codificador o de su estrategia.
