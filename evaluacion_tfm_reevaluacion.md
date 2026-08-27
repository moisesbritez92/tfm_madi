# INFORME DE EVALUACIÓN DEL TFM

**Ficha inicial**

| Elemento | Identificación a partir del PDF |
|---|---|
| Documento evaluado | `/Users/moises/Documents/001_TFM/tfm_madi/memoria/main.pdf` |
| Versión documental | Memoria fechada en septiembre de 2026, 63 páginas PDF y 47 páginas impresas de cuerpo y referencias |
| Tipo de trabajo | Estudio experimental comparativo y caso de estudio en simulación sobre una política visuomotora |
| Universidad | Tecnun, Universidad de Navarra |
| Máster | Máster Universitario en Análisis de Datos en Ingeniería |
| Título | *Influencia del codificador visual y su estrategia de entrenamiento en Diffusion Policy para manipulación robótica: estudio en Push-T* |
| Autor | Moises Britez |
| Director | Diego Borro |
| Problema investigado | Efecto del bloque visual y de su estrategia de adaptación sobre el rendimiento y el coste de una Diffusion Policy en Push-T con pocas demostraciones |
| Pregunta de investigación | Se formulan dos cuestiones en el apartado 1.2: qué codificador conviene y qué estrategia de entrenamiento resulta adecuada. No se presenta una pregunta interrogativa única ni un estimando de estrategia en la introducción |
| Objetivo general | Evaluar la influencia del codificador visual y de su estrategia de entrenamiento sobre el rendimiento de una Diffusion Policy aplicada a Push-T |
| Objetivos específicos | Implementar cinco variantes; comparar su puntuación media; analizar parámetros, tiempo, latencia y memoria |
| Hipótesis | Ventaja del preentrenamiento; ventaja del ajuste fino frente a la congelación; rendimiento comparable de ViT con mayor coste de inferencia |
| Datos de ajuste | 206 demostraciones teleoperadas, 25.650 transiciones; 90 episodios para entrenamiento, 4 para validación y 112 no utilizados |
| Datos de selección y prueba | 50 condiciones de selección con semillas 100.000 a 100.049 y 200 condiciones finales con semillas 200.000 a 200.199, generadas en el simulador |
| Población y alcance inferencial | Generador de condiciones iniciales de Push-T bajo la configuración concreta del simulador. La población de ejecuciones de entrenamiento no está replicada |
| Variable experimental | Bloque visual, que comprende arquitectura, pesos, estrategia de adaptación y cadena de preprocesado; el presupuesto y el tamaño de lote también presentan desviaciones entre variantes |
| Variables de resultado | Puntuación de cobertura, tasa de éxito, pérdidas, error de acción, tiempo por época, latencia y memoria gráfica |
| Métodos principales | Diffusion Policy con DDPM; ResNet-18, DINOv2 ViT-S/14 y CLIP ViT-B/16; bootstrap BCa; Wilcoxon pareado; corrección de Holm |
| Código y repositorio | El PDF informa de un repositorio en la bibliografía, pero no identifica una revisión, etiqueta, archivo permanente ni huella de integridad. Su contenido no forma parte de esta evaluación |
| Anexos | No se incluyen anexos en el PDF |
| Normas institucionales | Se comprueban únicamente los requisitos Tecnun incorporados al agente de evaluación y los elementos visibles en el PDF |
| Tiempo de defensa | NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA. |

**Alcance de esta reevaluación.** La única evidencia utilizada es el PDF indicado. No se ha inspeccionado el repositorio, el código, los registros de ejecución, los archivos de configuración ni la fuente LaTeX. Toda afirmación sobre su existencia o contenido se trata como una declaración documental del autor, no como una comprobación independiente.

## 1. Dictamen preliminar

**Dictamen: requiere reformulación antes de defensa.**

La memoria presenta un experimento completo, una comparación negativa informativa y una prueba final de 200 condiciones que, según el protocolo descrito, no intervino ni en el ajuste ni en la selección de los puntos de control. Esta separación corrige adecuadamente el riesgo de utilizar el conjunto de selección como prueba final. Las semillas 200.000 a 200.199 son disjuntas de las demostraciones y de las 50 semillas de selección; los cinco puntos de control se describen como congelados antes de consultar ese bloque (p. 19-20, apartado 3.8; p. 27-28, apartado 4.3; Tablas 11 y 12). No procede penalizar el trabajo por ausencia de prueba final ni afirmar que el test se utilizó para seleccionar el modelo.

La limitación principal aparece en otro nivel. Cada estrategia se entrena una sola vez. Las 200 condiciones replican la evaluación de cinco artefactos fijos, pero no replican el proceso de entrenamiento. Por ello, los intervalos y contrastes permiten comparar esos cinco puntos de control bajo las condiciones ensayadas, no estimar el rendimiento esperado de volver a entrenar cada estrategia. La memoria lo reconoce con precisión en las páginas 20, 28, 33 y 35, pero conserva formulaciones de objetivo, hipótesis y conclusión que atribuyen resultados a estrategias de entrenamiento. Esa atribución no queda demostrada con una unidad de entrenamiento por variante.

La comparación también está condicionada por presupuestos desiguales y decisiones adoptadas durante la campaña. V0 y V1 agotaron 500 épocas, V2 registró 266, V3 se detuvo manualmente en 155 pese a un presupuesto de 300 y V4 registró 200. La regla de parada se formuló después de observar V0 y V1 y no se aplicó de modo uniforme (p. 16-17, apartado 3.5; p. 33, apartado 4.8). Los análisis a época 150 y con cuatro oportunidades de selección reducen esta amenaza, pero no convierten el protocolo en preespecificado ni descartan una evolución posterior de V3.

Se añaden cuatro problemas relevantes. Primero, el supuesto contraste aislado entre V1 y V2 convive con una contradicción documental: se afirma que ambas comparten pesos iniciales y preprocesado, pero después se indica que V2 procede de `torchvision` y que los demás codificadores preentrenados se cargan mediante `timm` (p. 15-16, apartado 3.4). Segundo, la Ecuación (3) no representa correctamente el estado perturbado de un DDPM y contradice la Ecuación (1). Tercero, la inferencia estocástica se evalúa con una sola realización de ruido por condición, común a las variantes, de modo que la incertidumbre queda condicionada a ese esquema aleatorio. Cuarto, la reproducibilidad documental es parcial: faltan identificadores exactos de modelos y datos, un identificador de revisión del código, configuraciones completas y se mantienen dos referencias cruzadas sin resolver (pp. 16 y 22).

No se identifica evidencia documental de *data leakage* entre demostraciones, selección y prueba final. Tampoco se identifica un fallo que invalide por completo el resultado descriptivo central: entre los cinco puntos de control concretos, V0 obtiene mayor puntuación que V1 a V4 sobre las 200 condiciones finales. Sí existe insuficiencia para elevar ese resultado a una comparación robusta entre estrategias repetibles. Esta diferencia determina el dictamen.

## 2. Resumen analítico del TFM

El trabajo estudia una cuestión delimitada dentro de Diffusion Policy: si un bloque visual preentrenado ofrece una ventaja frente a una ResNet-18 ajustada desde cero cuando solo se utilizan 90 demostraciones de Push-T. El estado del arte conecta aprendizaje por imitación, modelos de difusión, representaciones visuales y adaptación de codificadores. El hueco se formula como la ausencia, dentro del corpus revisado, de una comparación conjunta de ResNet-18 desde cero, ResNet-18 preentrenada congelada y ajustada, DINOv2 congelado y CLIP congelado bajo un protocolo común de Push-T (pp. 8-9, apartado 2.6).

El diseño compara cinco variantes, V0 a V4. Todas entregan un vector de 512 componentes y comparten la red generativa. Sin embargo, no solo cambia la inicialización: V0 opera sobre la imagen de 96 píxeles, aplica recorte y `spatial softmax`, mientras V1 y V2 trabajan a 224 píxeles, normalizan según ImageNet y producen un descriptor global. V3 y V4 sustituyen además la arquitectura por un ViT y añaden una proyección lineal. La memoria reconoce que las comparaciones de V0 con las variantes preentrenadas estiman el efecto del bloque visual completo, no el efecto puro del preentrenamiento (pp. 15-16, apartado 3.4; p. 36, apartado 5.1).

El conjunto de demostraciones contiene 206 episodios. El ajuste usa 90, la pérdida de validación se calcula sobre 4 y 112 quedan descartados por sorteo. Los puntos de control se evalúan periódicamente sobre 50 condiciones reservadas. Una vez elegido uno por variante, los cinco se miden una vez sobre un bloque final de 200 semillas disjuntas. Las diferencias se emparejan por condición y se sincroniza el ruido de difusión mediante números aleatorios comunes. Se presentan intervalos BCa de las diferencias, pruebas de Wilcoxon y ajuste de Holm para las diez comparaciones principales (pp. 19-21, apartado 3.8).

En la prueba final, V0 obtiene 0,872; V1, 0,649; V2, 0,586; V3, 0,578; y V4, 0,490. Las cuatro diferencias entre V0 y las demás variantes mantienen valores ajustados muy inferiores a 0,05. Solo V1 frente a V4 resulta significativo entre las variantes preentrenadas (pp. 27-28, Tablas 11 y 12). La línea base también registra el menor tiempo por época. En latencia total, las variantes se separan poco porque cien pasos de difusión dominan la llamada; el codificador aislado sí muestra diferencias amplias (pp. 30-31, Tabla 13).

La contribución demostrable es un estudio de caso transparente sobre cinco entrenamientos concretos, con separación entre selección y prueba, resultados negativos y caracterización conjunta de rendimiento y coste. La contribución no alcanza una estimación general de las estrategias, pues falta replicación del entrenamiento, igualdad experimental completa y una medida de la variabilidad de inferencia. El trabajo puede defenderse si se reformula su alcance inferencial o, preferiblemente, si se repite el entrenamiento bajo un protocolo uniforme y se reserva un nuevo bloque final.

## 3. Puntuación global

| Criterio | Máximo | Obtenido antes del techo | Justificación |
|---|---:|---:|---|
| A. Problema, justificación y relevancia | 8 | 7,0 | Problema concreto, alcance estrecho y hueco respaldado por trabajos comparables. La relevancia se limita a un banco simulado y el gap depende del corpus revisado, sin estrategia de búsqueda documentada |
| B. Pregunta, objetivos e hipótesis | 10 | 7,0 | Objetivos claros y trazables. Las hipótesis sobre preentrenamiento y comparabilidad no están plenamente operacionalizadas: existe confusión de factores y no se fija margen de equivalencia |
| C. Estado del arte y marco conceptual | 10 | 8,5 | Cobertura actual, fuentes primarias y síntesis comparativa. Falta método de revisión y la formulación matemática de DDPM presenta un error posterior que afecta al marco técnico |
| D. Diseño metodológico | 14 | 8,5 | Separación correcta de ajuste, selección y prueba; variables y procedimiento bien descritos. Penalizan una sola semilla de entrenamiento, presupuestos desiguales, parada post hoc y detención manual de V3 |
| E. Datos y preparación | 12 | 9,5 | Procedencia, episodios, transiciones, partición y variables bien descritos. Faltan identificador y huella del conjunto, controles explícitos de integridad y una validación suficientemente grande |
| F. Análisis estadístico y modelado | 16 | 11,0 | Comparación pareada, BCa y Holm son decisiones valiosas. El Wilcoxon no contrasta directamente el estimando medio, falta variabilidad entre entrenamientos y el muestreo estocástico queda condicionado |
| G. Resultados, validación y discusión | 15 | 11,0 | Resultados claros, prueba final disjunta, análisis de resultados negativos y limitaciones. La validez se restringe a cinco artefactos y la discusión propone mecanismos no medidos |
| H. Conclusiones, aportación y limitaciones | 8 | 6,0 | Conclusiones vinculadas a objetivos y limitaciones explícitas. Algunas formulaciones siguen atribuyendo efectos a estrategias y una degradación de representación no demostrada |
| I. Reproducibilidad, transparencia y ética | 4 | 2,5 | Se informan hardware, versiones, semillas y bastantes hiperparámetros. Faltan identificador de revisión del código, identificadores exactos, configuración completa y determinismo; hay referencias cruzadas sin resolver |
| J. Presentación académica y comunicación | 3 | 2,0 | Estructura y tablas legibles. Persisten texto de plantilla, dos `??`, una ambigüedad de resolución y una bibliografía separada no integrada en las citas numeradas |
| **Suma aritmética** | **100** | **73,0** | **7,0 + 7,0 + 8,5 + 8,5 + 9,5 + 11,0 + 11,0 + 6,0 + 2,5 + 2,0 = 73,0** |

**Techo aplicado: 69/100.** Se activa la regla de máximo 69 porque la validación no permite estimar el rendimiento de las estrategias ante nuevos entrenamientos, los objetivos se cumplen solo de forma parcial en ese nivel y existen sesgos relevantes derivados del presupuesto desigual y de decisiones de parada no preespecificadas. No se activa el techo de 59: el PDF sí documenta una prueba final disjunta, la procedencia principal de los datos es trazable y no hay evidencia de que el test final seleccionara el modelo. No se activa el techo de 49: el resultado descriptivo central conserva evidencia directa para los cinco artefactos evaluados.

**TOTAL: 69/100.**

**Calificación: metodológicamente débil.** La puntuación bruta sería 73/100, «aprobable con correcciones importantes», pero el techo metodológico reduce la calificación final.

## 4. Perfil de puntuación

**Tres dimensiones más fuertes**

1. **Datos y preparación, 9,5/12.** Se informan origen, formato, número de episodios y transiciones, variables, reparto y semillas (pp. 11-14, apartado 3.2; Tabla 3; Figura 1).
2. **Estado del arte, 8,5/10.** La revisión conecta métodos de difusión, codificadores y antecedentes comparables, y termina en un hueco explícito (pp. 3-9, capítulo 2; Tablas 1 y 2).
3. **Resultados y validación descriptiva, 11/15.** El bloque final disjunto, las 200 condiciones comunes, la presentación de diferencias y el ajuste de multiplicidad sostienen la ordenación observada de los cinco artefactos (pp. 27-29, apartado 4.3; Tablas 11 y 12).

**Tres dimensiones más débiles**

1. **Diseño metodológico, 8,5/14.** Una ejecución de entrenamiento por variante no replica la intervención experimental; el presupuesto y la parada tampoco son uniformes (pp. 11, 16-17 y 33).
2. **Reproducibilidad, 2,5/4.** La documentación es amplia, pero no basta para fijar exactamente los artefactos, los datos y todas las decisiones; dos referencias internas están rotas (pp. 16, 21-22 y p. 47).
3. **Pregunta e hipótesis, 7/10.** «Comparable» no tiene margen ni diseño de equivalencia, y la hipótesis sobre preentrenamiento no se separa del preprocesado y la agregación espacial (pp. 11 y 15-16).

**Principal riesgo metodológico:** confundir 200 réplicas de condiciones iniciales para un modelo fijo con 200 réplicas de la estrategia de entrenamiento. La unidad experimental pertinente para afirmar que una estrategia supera a otra es la ejecución completa de entrenamiento, de la que existe una por variante (p. 20, apartado 3.8; p. 33, apartado 4.8).

## 5. Matriz de coherencia

### Cadena de coherencia científica

**Problema:** no se ha comparado el conjunto concreto de codificadores y estrategias bajo una Diffusion Policy común en Push-T (pp. 1 y 8-9).

**Pregunta:** qué codificador y qué estrategia convienen con pocas demostraciones (p. 1). La pregunta no define si el interés recae en artefactos entrenados una vez o en el rendimiento esperado de estrategias repetibles.

**Objetivo:** evaluar la influencia del codificador y de su estrategia sobre rendimiento y coste (p. 2).

**Hipótesis:** ventaja del preentrenamiento, ventaja del ajuste fino y comparabilidad de ViT con mayor coste (p. 11).

**Datos:** 90 episodios para ajuste, 4 para validación, 50 condiciones para selección y 200 para prueba de los puntos de control fijos (pp. 11-13 y 19-20).

**Método:** cinco entrenamientos, uno por variante; selección periódica; prueba final pareada; medición de coste (pp. 14-23).

**Resultado:** V0 supera en la muestra final a V1-V4; las variantes preentrenadas muestran orden interno incierto; V0 tiene menor coste por época (pp. 27-31).

**Conclusión:** V0 domina en la configuración estudiada; no se demuestra una ventaja del preentrenamiento ni del ajuste fino; la atribución se restringe al bloque visual completo (pp. 35-37).

**Rupturas identificadas**

1. **Hipótesis a método:** la hipótesis sobre preentrenamiento requiere aislar inicialización o estrategia. V0 frente a V1/V2 cambia resolución, recorte y agregación espacial, además de los pesos (pp. 15-16).
2. **Método a inferencia:** una ejecución por variante permite evaluar artefactos, no la distribución de resultados de las estrategias. La propia memoria fija una unidad por variante en ese nivel (p. 20).
3. **Método a comparación justa:** los presupuestos, las oportunidades de selección y la parada difieren; V3 se interrumpe antes de la regla declarada (pp. 16-17 y 33).
4. **Hipótesis a análisis:** «rendimiento comparable» no dispone de margen de equivalencia ni prueba de equivalencia. La no significación de algunos pares no demuestra comparabilidad (pp. 11, 28 y 33).
5. **Marco conceptual a método:** la Ecuación (3) omite los coeficientes del proceso directo que sí aparecen en la Ecuación (1), lo que rompe la consistencia matemática (pp. 3 y 14).
6. **Resultado a mecanismo:** la frase que atribuye a V2 una degradación de la representación no se apoya en un análisis de representaciones, sino en pérdidas y rendimiento (p. 36).

### Matriz por objetivo

| Objetivo | Método | Datos | Resultado | Conclusión | Estado |
|---|---|---|---|---|---|
| General: evaluar la influencia del codificador y de su estrategia | Comparación V0-V4 con prueba final pareada | 90 episodios de ajuste, 50 condiciones de selección, 200 de prueba; una ejecución por variante | Orden V0 > V1 > V2 > V3 > V4 en la muestra final | V0 domina en la configuración estudiada | **Parcialmente cumplido.** Se demuestra una diferencia entre artefactos, no un efecto reproducible de las estrategias |
| OE1: implementar cinco variantes manteniendo la política y la configuración | Sustitución del bloque visual y proyección común a 512 componentes | Mismos episodios y red generativa | Las cinco variantes generan curvas y resultados | Se declara cumplido | **Cumplido en el plano documental, con salvedades.** Presupuesto, lote y preprocesado no son constantes; la implementación no se comprueba fuera del PDF |
| OE2: comparar el rendimiento por puntuación media | Selección sobre 50 condiciones y prueba sobre 200; comparación pareada | 200 condiciones comunes por variante | 0,872; 0,649; 0,586; 0,578; 0,490 | V0 supera a las alternativas evaluadas | **Parcialmente cumplido.** La comparación es válida para puntos de control fijos, pero no incorpora variabilidad entre entrenamientos ni entre realizaciones de inferencia |
| OE3: analizar coste computacional | Parámetros, tiempo por época y total, latencia, memoria | Mediciones sobre hardware y entornos declarados | Tablas 8, 13 y 14 | V0 tiene menor coste por época; el generador domina la latencia total | **Cumplido dentro de la plataforma declarada.** No se generaliza a otro hardware ni a otras versiones |

## 6. Fortalezas demostrables

1. **Separación de selección y prueba final.** Las 50 semillas 100.000 a 100.049 seleccionan el punto de control y las 200 semillas 200.000 a 200.199 se consultan después sobre puntos congelados (pp. 19-20, Figura 2). Esta estructura evita estimar el rendimiento final con el mismo bloque que maximizó la puntuación.
2. **Definición explícita de unidades.** La memoria distingue episodio de demostración, condición inicial y ejecución completa de entrenamiento, y declara 90, 200 y 1 unidades por variante respectivamente (p. 20). Esta precisión permite identificar el alcance real de los contrastes.
3. **Emparejamiento pertinente.** Las cinco variantes resuelven las mismas condiciones y comparten el ruido de difusión mediante números aleatorios comunes (p. 20). El emparejamiento reduce ruido ajeno a la diferencia de interés entre los puntos de control fijos.
4. **Control de multiplicidad.** Las diez comparaciones de la variable principal se tratan como una familia y se ajustan con Holm (p. 21; Tabla 12). La tabla diferencia los valores sin ajustar y ajustados.
5. **Reconocimiento del carácter descriptivo de los intervalos no ajustados.** El texto aclara que los intervalos BCa no incorporan el ajuste familiar (p. 21). Esta declaración evita presentar esos intervalos como simultáneos.
6. **Resultado negativo informado sin ocultación.** Las cuatro alternativas preentrenadas rinden peor que V0 y se muestran sus curvas, costes y pérdidas (pp. 25-33). No se selecciona únicamente una variante favorable.
7. **Confusión codificador-preprocesado reconocida.** El apartado 3.4 enumera resolución, recorte y agregación espacial como factores que cambian junto con los pesos, y las conclusiones atribuyen el resultado al bloque visual completo (pp. 15-16 y 36).
8. **Limitaciones centrales declaradas.** Una sola semilla de entrenamiento, presupuestos desiguales, candidatos filtrados, frecuencia de evaluación y reanudación de V2 se describen en el apartado 4.8 (pp. 33-34).
9. **Caracterización de coste multidimensional.** Se separan tiempo por época, tiempo total, latencia del codificador, latencia de la política y memoria en entrenamiento e inferencia (pp. 22-23 y 30-31; Tablas 8, 13 y 14).
10. **Estado del arte conectado con la decisión experimental.** Las Tablas 1 y 2 comparan antecedentes por codificador, estrategia y componente modificado, y el apartado 2.6 deriva un hueco específico (pp. 5-9).

## 7. Hallazgos críticos

No se identifica en el PDF un hallazgo crítico que invalide por completo el resultado principal entendido de forma estricta como comparación de cinco puntos de control fijos sobre las 200 condiciones finales.

La independencia entre selección y prueba final, uno de los riesgos que debía revisarse expresamente, está documentada de forma adecuada en las páginas 19-20 y 27-28. Tampoco se observa en el PDF normalización, imputación, selección de variables, aumento de datos o selección de puntos de control que utilice las 200 condiciones finales. La correspondencia entre lo descrito y la ejecución efectiva es **NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.**

La ausencia de hallazgos críticos no implica que el trabajo esté listo para defensa. Los hallazgos mayores M1 a M9 reducen la validez de las conclusiones sobre estrategias y activan el techo de 69/100.

## 8. Observaciones mayores

### Hallazgo M1

**Ubicación:** p. 11, apartado 3.1; p. 20, «Estimando y unidades experimentales»; p. 28, último párrafo del apartado 4.3; p. 33, apartado 4.8; pp. 35-37, apartado 5.1.

**Tipo:** error de validación e información faltante sobre incertidumbre de entrenamiento.

**Descripción:** cada variante se entrena con una sola semilla. Las 200 condiciones finales son réplicas de evaluación de un punto de control fijo, no réplicas de la estrategia que lo produjo.

**Evidencia:** el PDF declara «1 unidad por variante» en el nivel de ejecución completa y reconoce que los intervalos no cuantifican la variabilidad entre entrenamientos.

**Por qué es problemático:** las hipótesis y el objetivo general se refieren a codificadores y estrategias. El efecto de la inicialización, el orden de lotes, operaciones no deterministas y dinámica de optimización puede desplazar el rendimiento de cada entrenamiento. Con una ejecución no se separa el efecto de estrategia del resultado particular de la semilla 42.

**Impacto sobre resultados:** no invalida que V0=0,872 supere a los otros cuatro puntos de control en las condiciones evaluadas. Sí impide afirmar que entrenar desde cero sea, en expectativa, superior a congelar o ajustar un codificador preentrenado.

**Corrección necesaria:** repetir cada condición con varias semillas de entrenamiento bajo un protocolo idéntico. La ejecución completa debe ser la unidad primaria para inferir sobre estrategias. Si no se añaden réplicas, reformular título, objetivo, hipótesis y conclusiones como estudio de caso de cinco entrenamientos concretos.

**Evidencia necesaria para considerarlo resuelto:** resultados por semilla, dispersión entre entrenamientos, protocolo uniforme preespecificado y análisis jerárquico que separe variabilidad entre entrenamientos, condiciones iniciales e inferencia.

### Hallazgo M2

**Ubicación:** pp. 16-17, apartado 3.5; pp. 22-23, Tabla 8; pp. 25-26, Tablas 9 y 10; p. 33, apartado 4.8.

**Tipo:** error metodológico de comparación y decisión metodológica post hoc.

**Descripción:** las variantes no reciben el mismo presupuesto ni el mismo criterio de parada. V0 y V1 llegan a 500 épocas; V2, a 266; V3 se detiene manualmente en 155 pese a un presupuesto de 300 y al mínimo declarado de 200; V4 registra 200. La regla de parada se adopta después de observar V0 y V1.

**Evidencia:** el apartado 3.5 indica de forma expresa que el criterio «no precedió al experimento», que V3 se interrumpió manualmente y que las decisiones se tomaron por inspección.

**Por qué es problemático:** el rendimiento final depende de la oportunidad de converger, sobreajustar y ser evaluado. La intervención experimental deja de ser solo el bloque visual porque el régimen de optimización también cambia.

**Impacto sobre resultados:** la lectura a época 150 y la igualación a cuatro evaluaciones mantienen a V0 por encima, lo que mitiga el riesgo para la ordenación observada. No descartan una recuperación posterior de V3 ni permiten comparar el mejor rendimiento alcanzable por cada estrategia bajo igual presupuesto.

**Corrección necesaria:** fijar antes de entrenar un presupuesto común en actualizaciones, no solo en épocas, y una regla automática de parada aplicada a todas las variantes. Guardar puntos de control en momentos comunes e independientes de la puntuación.

**Evidencia necesaria para considerarlo resuelto:** protocolo fechado, curvas completas bajo igual número de actualizaciones, registro de paradas automáticas y resultados de todas las variantes con las mismas oportunidades de selección.

### Hallazgo M3

**Ubicación:** pp. 15-16, apartado 3.4, párrafos de V1 y V2 y último párrafo de «V3 y V4».

**Tipo:** contradicción metodológica y problema de reproducibilidad.

**Descripción:** se afirma que V1 y V2 comparten arquitectura, pesos iniciales y preprocesado, por lo que aislarían congelación frente a ajuste fino. Poco después se indica que los codificadores preentrenados se cargan mediante `timm`, salvo la ResNet-18 de V2, que procede de `torchvision`. El texto no identifica el modelo exacto de `timm` ni demuestra que sus pesos y transformaciones sean idénticos a los de `torchvision`.

**Evidencia:** ambas afirmaciones aparecen en páginas consecutivas del mismo apartado.

**Por qué es problemático:** el contraste V1-V2 solo identifica la estrategia de entrenamiento si la inicialización y el preprocesado son exactamente iguales. Dos proveedores o recetas de pesos pueden introducir diferencias en pesos, normalización, interpolación o recorte.

**Impacto sobre resultados:** compromete la interpretación de la diferencia 0,063 como efecto aislado del ajuste fino. No afecta directamente a la observación de que los dos puntos de control obtienen 0,649 y 0,586.

**Corrección necesaria:** indicar identificador exacto, fuente, suma de comprobación y transformaciones de ambos modelos. Lo preferible es inicializar V1 y V2 desde el mismo archivo de pesos y duplicar el estado antes de congelar una copia.

**Evidencia necesaria para considerarlo resuelto:** tabla con identificadores y hashes, confirmación de igualdad tensor a tensor en la inicialización y definición idéntica de la cadena de preprocesado.

### Hallazgo M4

**Ubicación:** p. 2, objetivo general; p. 11, primera hipótesis; pp. 15-16, apartado 3.4; pp. 35-37, apartado 5.1.

**Tipo:** confusión de factores y conclusión que excede el contraste causal disponible.

**Descripción:** V0 frente a V1/V2 cambia al menos inicialización, resolución, recorte y agregación espacial. Frente a V3/V4 cambia también la arquitectura. El trabajo reconoce esta confusión y redefine el tratamiento como bloque visual, pero la hipótesis inicial sigue preguntando por la utilidad del preentrenamiento y el título destaca el codificador y su estrategia.

**Evidencia:** el propio texto enumera cuatro factores que varían conjuntamente y afirma que la comparación debe leerse sobre el bloque completo.

**Por qué es problemático:** no puede atribuirse la diferencia observada al preentrenamiento, a la arquitectura ni a la estrategia por separado. El `spatial softmax` de V0 puede conservar información geométrica que los descriptores globales pierden.

**Impacto sobre resultados:** la conclusión «ningún bloque visual preentrenado evaluado mejora V0» es válida para las configuraciones completas. La conclusión «el preentrenamiento no mejora» no lo es como efecto aislado.

**Corrección necesaria:** ejecutar una ablación factorial que iguale resolución, aumento y agregación, o reformular de manera uniforme todas las preguntas y conclusiones en términos de configuraciones completas.

**Evidencia necesaria para considerarlo resuelto:** comparaciones donde solo cambie un factor por vez, o una reformulación explícita que elimine toda atribución causal al preentrenamiento aislado.

### Hallazgo M5

**Ubicación:** p. 20, «Ruido de difusión sincronizado»; p. 21, último párrafo de «Especificación estadística»; p. 33, apartado 4.8.

**Tipo:** información faltante sobre incertidumbre y limitación no plenamente incorporada al estimando.

**Descripción:** cada condición inicial se resuelve con una sola trayectoria de difusión, compartida entre variantes. La sincronización es útil para un contraste pareado, pero no estima la variabilidad propia del muestreo estocástico ni el rendimiento esperado sobre ese ruido.

**Evidencia:** el texto declara una sola trayectoria por condición y afirma que las medidas no mezclan la variabilidad del muestreo de difusión.

**Por qué es problemático:** una Diffusion Policy es estocástica durante la inferencia. El rendimiento de un punto de control depende tanto de la condición inicial como de la realización del ruido. El estimando se presenta primero como media sobre el generador de condiciones, pero queda condicionado a una secuencia concreta de ruido que no se integra ni se replica.

**Impacto sobre resultados:** la comparación pareada puede ser más precisa, pero sus intervalos no cubren la variación que aparecería al repetir la inferencia con otras semillas. Las diferencias cercanas a cero entre variantes preentrenadas son especialmente sensibles.

**Corrección necesaria:** fijar varias semillas de inferencia por condición y variante, manteniendo números aleatorios comunes dentro de cada réplica. Definir si el estimando integra condición inicial, ruido de difusión o ambos.

**Evidencia necesaria para considerarlo resuelto:** resultados desagregados por semilla de inferencia, componente de varianza correspondiente y análisis jerárquico o remuestreo que respete ambos niveles.

### Hallazgo M6

**Ubicación:** pp. 20-21, «Especificación estadística»; pp. 27-28, Tablas 11 y 12; p. 33, apartado 4.7.

**Tipo:** error estadístico de correspondencia entre estimando y contraste, y especificación incompleta de intervalos.

**Descripción:** el estimando primario es la diferencia de medias, pero la prueba de rangos con signo de Wilcoxon no contrasta directamente una diferencia media. Bajo condiciones adicionales puede interpretarse como contraste de localización; sin ellas, responde a la distribución de diferencias. Además, el método de los intervalos por variante de la Tabla 11 no se identifica, mientras los intervalos BCa de la Tabla 12 sí se documentan.

**Evidencia:** las diferencias se rotulan como «diferencia media» y los valores p proceden de Wilcoxon. La Tabla 11 informa IC 95 % sin método explícito.

**Por qué es problemático:** una conclusión sobre diferencia media debe apoyarse en un procedimiento inferencial dirigido a esa media o en una justificación clara del contraste alternativo. La discrepancia es visible en V1-V2: el IC no ajustado de la media apenas excluye cero, mientras Wilcoxon produce p=0,066 y Holm p=0,132.

**Impacto sobre resultados:** las grandes diferencias V0 frente a V1-V4 también poseen intervalos BCa alejados de cero, por lo que la conclusión descriptiva central es robusta a esta objeción. Las comparaciones limítrofes entre preentrenadas no admiten una lectura concluyente.

**Corrección necesaria:** usar un contraste por permutación o bootstrap dirigido a la diferencia media, justificar sus supuestos y definir todos los intervalos. Mantener Holm o construir intervalos simultáneos si se desean afirmaciones familiares.

**Evidencia necesaria para considerarlo resuelto:** especificación completa, diagnóstico de supuestos, código o pseudocódigo del procedimiento y resultados coherentes para el mismo estimando.

### Hallazgo M7

**Ubicación:** p. 3, Ecuación (1); p. 4, Ecuación (2); p. 14, Ecuaciones (3) y (4), apartado 3.3.

**Tipo:** error conceptual y problema de reproducibilidad matemática.

**Descripción:** la Ecuación (3) introduce el estado perturbado como `A_t^0 + epsilon^k`. En un DDPM, la muestra en el paso k requiere los coeficientes dependientes del plan de ruido que escalan la señal limpia y el ruido, tal como el propio documento expresa en la Ecuación (1). La Ecuación (4) utiliza coeficientes genéricos alfa, gamma y sigma sin definirlos ni relacionarlos con alfa con barra, beta o el planificador.

**Evidencia:** la Ecuación (1) contiene la forma con raíz de alfa acumulada y raíz de uno menos alfa acumulada; la Ecuación (3) omite ambas. El texto de la Ecuación (4) solo indica que los coeficientes «proceden del planificador».

**Por qué es problemático:** la metodología matemática debe describir el algoritmo que supuestamente se ejecuta. La forma actual es incompatible con el proceso directo presentado en el estado del arte y no permite reproducir la perturbación ni la transición inversa.

**Impacto sobre resultados:** el PDF no permite determinar si el error es solo de redacción o si refleja el procedimiento ejecutado. No se infiere que los resultados sean incorrectos, pero la corrección técnica y la reproducibilidad del método quedan debilitadas. La ejecución efectiva es **NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.**

**Corrección necesaria:** sustituir las ecuaciones por la formulación DDPM exacta, definir cada coeficiente y vincularla con la parametrización del planificador utilizado.

**Evidencia necesaria para considerarlo resuelto:** ecuaciones coherentes entre capítulos, valores o reglas del plan de ruido, definición del paso k y correspondencia explícita con la configuración de `diffusers`.

### Hallazgo M8

**Ubicación:** p. 12, Tabla 3; pp. 17-18, apartados 3.6 y 3.7; p. 29, apartado 4.4.

**Tipo:** debilidad de validación e insuficiencia de muestra de diagnóstico.

**Descripción:** la pérdida de validación se estima con solo cuatro episodios y 404 ventanas. Las ventanas de un mismo episodio son dependientes, por lo que 404 no equivale a 404 unidades independientes. El texto utiliza la divergencia de la pérdida de validación como evidencia de sobreajuste.

**Evidencia:** Tabla 3: validación=4 episodios. Figura 4 y apartado 4.4 interpretan el aumento de esa pérdida en V1-V4.

**Por qué es problemático:** cuatro trayectorias ofrecen una estimación inestable de generalización y pueden no cubrir la diversidad de estados. La dependencia temporal reduce todavía más la información efectiva.

**Impacto sobre resultados:** no invalida la prueba final en bucle cerrado, que usa 200 condiciones. Sí debilita la interpretación de las curvas de validación y el diagnóstico de sobreajuste.

**Corrección necesaria:** reservar un conjunto de validación mayor a nivel de episodio o utilizar particiones repetidas por episodio, sin mezclar ventanas del mismo episodio entre subconjuntos.

**Evidencia necesaria para considerarlo resuelto:** justificación del tamaño, incertidumbre de la pérdida por episodio y estabilidad de la selección bajo varias particiones.

### Hallazgo M9

**Ubicación:** p. 11, tercera hipótesis; p. 21, apartado 3.9; p. 22, Tabla 7; p. 47, Bibliografía.

**Tipo:** hipótesis no operacionalizada y problema de reproducibilidad documental.

**Descripción:** la tercera hipótesis utiliza «comparable» sin margen de equivalencia. A la vez, la reproducción exacta depende de una bifurcación propia, configuraciones y pesos sin revisión identificada. El repositorio se cita como entrada bibliográfica general, no como versión inmutable del experimento.

**Evidencia:** no se fija un delta de equivalencia ni se aplica TOST u otro procedimiento equivalente. La Tabla 7 solo nombra «bifurcación propia» y la bibliografía ofrece una URL móvil.

**Por qué es problemático:** no rechazar diferencias entre V1, V2, V3 y V4 no demuestra rendimiento comparable. Sin una versión fija de los artefactos tampoco puede repetirse con precisión el procedimiento que produciría esos puntos de control.

**Impacto sobre resultados:** la memoria evita en general concluir equivalencia y habla de orden no resuelto, lo que es correcto. La hipótesis, sin embargo, no puede considerarse contrastada como fue formulada. La reproducibilidad queda en nivel parcial.

**Corrección necesaria:** definir un margen práctico antes de observar los datos y aplicar un análisis de equivalencia. Identificar una versión archivada del código, configuraciones, pesos y protocolo.

**Evidencia necesaria para considerarlo resuelto:** margen justificado, intervalos compatibles con equivalencia y DOI, commit o etiqueta inmutable con manifiesto de archivos.

## 9. Observaciones menores

### Hallazgo m1

**Ubicación:** p. iii, Agradecimientos, página PDF 5.

**Tipo:** problema de estructura académica y texto de plantilla.

**Descripción:** la página contiene la instrucción «[Texto de agradecimientos...]» en lugar de agradecimientos o de la omisión de la página opcional.

**Evidencia:** texto visible íntegramente en la página iii.

**Por qué es problemático:** revela una plantilla no depurada en la versión evaluada.

**Impacto sobre resultados:** ninguno; reduce la calidad formal y la preparación de entrega.

**Corrección necesaria:** redactar el apartado o eliminarlo y regenerar índices y paginación.

**Evidencia necesaria para considerarlo resuelto:** PDF final sin instrucciones editoriales.

### Hallazgo m2

**Ubicación:** p. 16, final del apartado 3.5; p. 22, apartado 3.9.

**Tipo:** problema de referencias cruzadas y reproducibilidad.

**Descripción:** aparecen dos referencias sin resolver: «se recoge en el ??» y «se documentan en el ??».

**Evidencia:** texto visible en ambas páginas.

**Por qué es problemático:** impide localizar la configuración efectiva y la tabla de versiones a la que remite la metodología.

**Impacto sobre resultados:** no modifica las cifras, pero dificulta verificar el protocolo descrito.

**Corrección necesaria:** enlazar con Tabla 5, Tabla 7 o el apartado correcto, según corresponda.

**Evidencia necesaria para considerarlo resuelto:** ausencia de `??` y referencias navegables correctas.

### Hallazgo m3

**Ubicación:** p. 15, descripción de V0; p. 30, Tabla 13 y párrafo posterior.

**Tipo:** inconsistencia terminológica y de preprocesado.

**Descripción:** V0 se describe como operando a resolución nativa de 96 píxeles, pero la Tabla 13 informa 76 píxeles y el texto compara 76 con 224. No se especifica de forma directa que 76 sea el tamaño posterior al recorte ni el parámetro del recorte.

**Evidencia:** valores distintos para la resolución de entrada en dos apartados.

**Por qué es problemático:** el tamaño efectivo es un factor del experimento y afecta a coste, información espacial y reproducibilidad.

**Impacto sobre resultados:** pequeño para las cifras ya reportadas; relevante para repetir el preprocesado.

**Corrección necesaria:** distinguir resolución renderizada, tamaño antes del recorte y tamaño efectivo del tensor.

**Evidencia necesaria para considerarlo resuelto:** tabla única de transformaciones con tamaños en cada etapa.

### Hallazgo m4

**Ubicación:** pp. 43-45, Referencias; p. 47, Bibliografía.

**Tipo:** problema de citación y consistencia bibliográfica.

**Descripción:** las referencias científicas [1]-[29] siguen numeración, pero la monografía del autor y el repositorio aparecen en una sección «Bibliografía» separada y sin número. La monografía no se integra en el texto y el repositorio se menciona sin cita formal numerada.

**Evidencia:** separación visible entre «Referencias» y «Bibliografía».

**Por qué es problemático:** el sistema IEEE requiere correspondencia unívoca entre citas y entradas. Una obra no citada no debe figurar como referencia bibliográfica del trabajo.

**Impacto sobre resultados:** ninguno; afecta a consistencia y trazabilidad.

**Corrección necesaria:** integrar solo las fuentes citadas en una lista IEEE y citar formalmente el repositorio cuando se describa la reproducibilidad.

**Evidencia necesaria para considerarlo resuelto:** una sola lista numerada y correspondencia completa cita-entrada.

### Hallazgo m5

**Ubicación:** p. 20, definición de tasa de éxito; p. 28, Tabla 11.

**Tipo:** información estadística faltante.

**Descripción:** la tasa de éxito se define como variable secundaria y se informa como conteo, pero no presenta intervalo, contraste ni análisis de multiplicidad.

**Evidencia:** Tabla 11 contiene 101/200, 37/200, 32/200, 32/200 y 16/200 sin incertidumbre.

**Por qué es problemático:** un resultado secundario también requiere indicar su incertidumbre si se interpreta comparativamente.

**Impacto sobre resultados:** limitado, porque las conclusiones principales utilizan la puntuación continua.

**Corrección necesaria:** presentar intervalos binomiales y declarar el análisis como descriptivo o incorporarlo a una familia secundaria preespecificada.

**Evidencia necesaria para considerarlo resuelto:** intervalos y regla de interpretación explícita.

### Hallazgo m6

**Ubicación:** pp. 11-14, apartado 3.2.

**Tipo:** información faltante sobre calidad de datos.

**Descripción:** se describen episodios, transiciones, longitudes, puntuaciones y condiciones iniciales, pero no se documentan controles de valores ausentes, episodios duplicados, transiciones corruptas, rangos de acciones ni duplicidad entre subconjuntos más allá de las semillas.

**Evidencia:** el apartado de datos no contiene un inventario de estas comprobaciones.

**Por qué es problemático:** la procedencia pública no sustituye una comprobación de integridad del archivo concreto empleado.

**Impacto sobre resultados:** no existe evidencia de corrupción, por lo que no se infiere un defecto de los datos. La calidad completa es **NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.**

**Corrección necesaria:** añadir una tabla de controles de integridad y sus resultados.

**Evidencia necesaria para considerarlo resuelto:** conteos, reglas, hashes y resultados de validación del conjunto.

### Hallazgo m7

**Ubicación:** p. 14, último párrafo del apartado 3.2; Figura 1.

**Tipo:** error de interpretación de no significación y multiplicidad exploratoria.

**Descripción:** se afirma que no existe sesgo apreciable y que no se aprecian diferencias a partir de valores p no significativos en varias pruebas U y Kolmogorov-Smirnov. No se define margen de equivalencia y no se corrigen esas comparaciones exploratorias.

**Evidencia:** se informan p=0,752, 0,999, 0,302, 0,418 y cinco valores entre 0,232 y 0,696.

**Por qué es problemático:** no rechazar igualdad no demuestra equivalencia ni ausencia de sesgo. La selección aleatoria uniforme, si se ejecutó como se afirma, es el argumento principal contra un sesgo sistemático de selección.

**Impacto sobre resultados:** menor, pues las pruebas no sostienen el resultado central y el reparto aleatorio está documentado.

**Corrección necesaria:** sustituir la inferencia de equivalencia por comparaciones descriptivas con tamaños de efecto o pruebas de equivalencia con márgenes justificados.

**Evidencia necesaria para considerarlo resuelto:** estimaciones de diferencia e intervalos compatibles con márgenes definidos.

### Hallazgo m8

**Ubicación:** p. 20, «Bloque de prueba disjunto».

**Tipo:** debilidad de justificación documental.

**Descripción:** se afirma que el protocolo final se fijó por escrito y se registró en el repositorio antes de evaluar, pero el PDF no aporta fecha, ruta, commit ni reproducción del protocolo.

**Evidencia:** declaración sin identificador verificable.

**Por qué es problemático:** la preespecificación solo puede evaluarse si existe una marca temporal o una versión inmutable anterior al resultado.

**Impacto sobre resultados:** el orden metodológico descrito es correcto, pero su preespecificación es **NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.**

**Corrección necesaria:** citar el commit o registro fechado que contiene el protocolo.

**Evidencia necesaria para considerarlo resuelto:** identificador inmutable anterior a la ejecución final.

## 10. Auditoría metodológica

### Diseño

El diseño es comparativo y se aproxima a una ablación de configuraciones visuales. La red generativa, los episodios de ajuste, la dimensión de salida y el protocolo de evaluación se declaran comunes (p. 11). V0 funciona como línea base pertinente porque reproduce la opción de referencia de Diffusion Policy (p. 15). La prueba final se separa de la selección y mejora de forma sustancial la validez interna del resultado sobre artefactos fijos.

No es, sin embargo, un diseño factorial ni una comparación equilibrada de estrategias. Arquitectura, inicialización, resolución, aumento y agregación cambian en bloques. El presupuesto y la parada también varían. En consecuencia, la variable manipulada real es una configuración compuesta, no «el codificador» en sentido aislado. La memoria reconoce buena parte de esta limitación, pero no adapta de manera completa las hipótesis.

### Muestra y unidades

Existen tres niveles. Los 90 episodios entrenan cada modelo; las 200 condiciones finales evalúan cada punto de control; una ejecución completa produce cada punto de control. La muestra de 200 es razonable para estimar el rendimiento medio condicionado al modelo y al esquema de ruido, como muestran errores estándar entre 0,018 y 0,029 (Tabla 11). No existe justificación de potencia previa, aunque las diferencias V0-V1/V4 son amplias.

Para comparar estrategias, n=1 por variante. El tamaño de 200 no corrige esta carencia porque esas condiciones no vuelven a entrenar el modelo. Este es el principal límite de validez.

### Variables

La puntuación de cobertura está definida mediante la Ecuación (5), tiene rango [0,1] y se vincula con el umbral de éxito. La media responde al objetivo de rendimiento. La tasa de éxito complementa la interpretación. Los indicadores de coste cubren parámetros, tiempo, latencia y memoria. La definición de «píxeles» debe aclararse entre 96 y 76.

La variable experimental se define finalmente como bloque visual completo. Esta definición es metodológicamente honesta, pero reduce la capacidad de responder a hipótesis separadas sobre preentrenamiento y arquitectura.

### Procedimiento

El PDF documenta horizontes, pasos de difusión, optimizador, tasa de aprendizaje, calentamiento, media móvil, lotes, acumulación, presupuestos y semilla (Tabla 5). Faltan parámetros completos de AdamW, detalles del plan de ruido, transformaciones exactas, identificadores de pesos y automatización reproducible de la parada. La interrupción manual de V3 contradice la regla declarada.

### Sesgos

El sesgo de selección de puntos de control se trata correctamente mediante un conjunto de selección separado. El hecho de guardar los tres mejores según selección no contamina por sí mismo la prueba final; esa es la función normal de un conjunto de validación. El problema residual no es un test «parcialmente contaminado», sino que el conjunto de candidatos es limitado, desigual y generado bajo decisiones post hoc.

El sesgo por presupuesto favorece potencialmente a las variantes con más entrenamiento y evaluaciones. Los análisis de Tabla 10 muestran que V0 ya lidera a época común, lo que reduce pero no elimina la amenaza. El sesgo por semilla de entrenamiento no puede cuantificarse. El sesgo por dominio limita la extrapolación de imágenes sintéticas a percepción real.

### Validez

**Validez interna para cinco artefactos:** moderada. La prueba final disjunta, el emparejamiento y las diferencias amplias apoyan la ordenación observada. La estocasticidad de inferencia, la contradicción V1-V2 y el protocolo desigual reducen la certeza.

**Validez interna para estrategias:** débil. Falta replicación del entrenamiento y aislamiento factorial.

**Validez externa:** baja. Solo se estudia Push-T, una implementación, un equipo, 90 demostraciones, una semilla de entrenamiento y simulación. No hay robot físico, otras tareas, perturbaciones ni dominios visuales reales.

## 11. Auditoría de datos

### Procedencia y trazabilidad

El PDF atribuye el conjunto a los autores de Diffusion Policy, indica formato zarr, tamaño aproximado de 30 MB, 206 demostraciones teleoperadas, 25.650 transiciones y semillas 0 a 205 (pp. 11-12). También declara haber reproducido el primer instante de cada demostración. Estas descripciones ofrecen trazabilidad conceptual.

Faltan URL o identificador específico del artefacto de datos, versión, fecha de descarga, hash, licencia y manifiesto de archivos. La identidad exacta del fichero utilizado es **NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.**

### Calidad e integridad

La memoria caracteriza longitudes, puntuación del demostrador, posiciones y orientaciones iniciales (Tabla 3 y Figura 1). Informa mediana y rango de puntuación humana, y señala que ninguna demostración alcanza el umbral de éxito. Esto es relevante para interpretar el aprendizaje.

No se informan valores ausentes, duplicados, rangos imposibles, episodios truncados, imágenes corruptas ni distribución completa de acciones y contactos. Al tratarse de un conjunto simulado, algunos problemas pueden ser improbables, pero deben comprobarse. No se puede afirmar su presencia ni ausencia.

### Partición

La partición se realiza por episodio y no por ventana, lo que evita que ventanas solapadas del mismo episodio se repartan entre ajuste y validación. La semilla 42 y los conteos se documentan. Los 112 episodios no usados se seleccionan aleatoriamente, no por rendimiento.

El conjunto de validación de cuatro episodios es demasiado pequeño para un diagnóstico estable. Usar solo 90 de 206 reduce cobertura sin que se estudie el efecto del tamaño. La motivación de reproducir la implementación de referencia es válida para comparabilidad, pero no demuestra que 90 sea el tamaño óptimo.

### Representatividad

Las 90 demostraciones representan una submuestra aleatoria del fichero. Las pruebas no significativas frente a los descartados no demuestran equivalencia. La prueba final procede del mismo generador simulado, por lo que evalúa generalización interna a otras condiciones iniciales, no cambio de dominio.

La población válida es: condiciones del generador Push-T bajo la versión y configuración concretas, con la realización de ruido definida por el protocolo. No se justifica extrapolar a otras tareas, cámaras, motores físicos, operadores o imágenes reales.

### Leakage

No se observa leakage documental entre ajuste, selección y prueba. Las demostraciones usan semillas 0 a 205; selección, 100.000 a 100.049; prueba, 200.000 a 200.199. El aumento se aplica en entrenamiento y V0 usa recorte central en evaluación. No se describen normalización o imputación ajustadas con información de prueba.

La ejecución efectiva de estas garantías y la inexistencia de solapamientos internos son **NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.**

### Privacidad y ética

El contenido principal son imágenes y estados de simulación. No se describen datos personales ni información sensible. La teleoperación humana se representa como trayectorias del simulador, sin identificación del operador. No parece necesario un tratamiento de privacidad personal a partir de la evidencia del PDF. Licencia y condiciones de uso secundario del conjunto no se documentan de forma suficiente.

## 12. Auditoría estadística

### Estadística descriptiva

El trabajo informa medias, errores estándar, intervalos, medianas e IQR donde corresponde. La Tabla 11 presenta puntuación y éxito; la Tabla 12 presenta diferencias medias pareadas. El coste usa mediana e IQR de 50 repeticiones de latencia, decisión adecuada para medidas temporales asimétricas (p. 18 y Tabla 13).

Los intervalos individuales de la Tabla 11 carecen de método explícito. Debe aclararse si son normales, bootstrap percentil, BCa u otra construcción. La notación «media ± error estándar» no es un intervalo y no debe confundirse con el IC de la columna siguiente.

### Estimando y unidad experimental

El estimando se define como puntuación media de un punto de control fijo sobre el generador de condiciones. Esta definición es mejor que una referencia genérica a «rendimiento». Aun así, la realización de ruido de difusión queda fija por condición, por lo que el estimando efectivo es condicional. Si el objetivo es el rendimiento operativo de una política estocástica, debe integrarse también esa fuente de azar.

La condición inicial es unidad válida para comparar los cinco puntos de control fijos. La ejecución de entrenamiento es la unidad válida para comparar estrategias. Los valores p de la Tabla 12 no deben interpretarse como inferencia sobre futuras ejecuciones de entrenamiento.

### Bootstrap BCa

El remuestreo de 10.000 pares respeta el emparejamiento y BCa corrige sesgo y asimetría. La semilla fija favorece reproducibilidad. El procedimiento supone que las 200 condiciones son intercambiables o una muestra adecuada del generador. El uso de semillas consecutivas no es problemático por sí mismo si el generador produce muestras equivalentes, pero esa propiedad se asume.

Los intervalos de Tabla 12 no están ajustados por multiplicidad y el texto lo declara. Por ello, un intervalo que excluya cero no equivale a significación familiar. La interpretación de V1-V2 reconoce esta diferencia.

### Wilcoxon

La prueba pareada es coherente con las mismas condiciones. La aproximación normal con corrección de continuidad y el tratamiento de ceros se especifican. Sin embargo, Wilcoxon no es una prueba directa de diferencia media. La memoria debe decidir si el objetivo inferencial es media, mediana, distribución o probabilidad de superioridad, y seleccionar el procedimiento correspondiente.

La sincronización del ruido mejora el emparejamiento, pero una única realización puede hacer que la distribución de diferencias dependa del esquema de siembra. Se requieren varias realizaciones para evaluar robustez.

### Holm y multiplicidad

La familia de diez comparaciones de la variable principal se define de antemano en el texto y Holm controla el error familiar sin exigir independencia. Los valores ajustados de la Tabla 12 son aritméticamente coherentes con el procedimiento secuencial. Este componente es correcto.

La tasa de éxito, los contrastes de representatividad y las múltiples lecturas exploratorias no forman parte de esa familia. Deben rotularse como análisis descriptivos o recibir familias separadas. No se debe usar la ausencia de significación en las pruebas de distribución como prueba de equivalencia.

### Significación, efecto y relevancia

Las diferencias V0-V1/V4, entre 0,223 y 0,382 en una escala [0,1], son grandes y sus IC BCa se alejan de cero. El resultado posee relevancia práctica para los cinco artefactos. Las diferencias entre preentrenadas son menores y varias quedan alrededor del umbral tras Holm. No se demuestra equivalencia entre ellas.

No se presenta un cálculo de potencia. Con una sola semilla de entrenamiento, ningún aumento de condiciones iniciales proporciona potencia para inferir el efecto medio de estrategia entre entrenamientos.

## 13. Auditoría de Machine Learning

### Baseline y alternativas

V0 es un baseline pertinente: ResNet-18 desde cero y configuración visual de la implementación de referencia (p. 15). Para la pregunta concreta sobre el bloque visual, no es imprescindible añadir una política no difusiva. Sí sería útil una ResNet-18 desde cero con el mismo preprocesado de V1/V2, porque separaría inicialización y representación.

### División de datos

La división se realiza por episodio. Las condiciones de evaluación se generan fuera del fichero y la prueba final utiliza semillas disjuntas. No se aprecia fuga entre entrenamiento y prueba en el diseño descrito. El conjunto de validación de cuatro episodios es insuficiente para estabilizar la pérdida.

### Arquitecturas y selección

Las arquitecturas se justifican mediante antecedentes: ResNet, DINOv2 y CLIP cubren supervisión desde cero, supervisada, autosupervisada y multimodal. Todas entregan 512 componentes. Esta igualdad de dimensión controla la interfaz con la red generativa, pero no iguala la información representada, el número de parámetros, la agregación espacial ni la resolución.

V3 y V4 quedan congeladas por restricción de memoria. Por tanto, la matriz no cruza todas las arquitecturas con todas las estrategias. No puede separarse «ViT» de «ViT congelado» ni compararse ajuste fino entre familias.

### Hiperparámetros y entrenamiento

La Tabla 5 ofrece los principales hiperparámetros. Faltan `weight decay`, betas y epsilon de AdamW, detalles de recorte e interpolación, plan de ruido completo, criterio exacto de guardado cuando hay empates y parámetros de determinismo. Los presupuestos desiguales y la regla post hoc impiden una comparación estrictamente controlada.

La acumulación mantiene lote efectivo 64 y 168 actualizaciones por época, lo que es una compensación razonable. No iguala el número de pasadas por el codificador ni el coste, pero esas diferencias forman parte del indicador computacional y se reconocen.

### Sobreajuste

Las curvas muestran pérdida de entrenamiento decreciente y validación creciente, junto con retroceso de la puntuación en V1-V4 (Figura 4). Este patrón es compatible con sobreajuste. No basta para afirmar que el ajuste fino «degrada la representación»; también puede reflejar optimización, tamaño del conjunto, regularización o pérdida desalineada con el control.

### Métricas

La puntuación de cobertura es adecuada para Push-T y la tasa de éxito aporta interpretación. El error de acción se usa como diagnóstico y el texto muestra correctamente que no sustituye el rendimiento en bucle cerrado. La métrica principal toma el máximo temporal, lo que es coherente con la definición del entorno, aunque debe recordarse que no mide estabilidad final.

### Robustez e interpretabilidad

No se evalúan perturbaciones visuales, cambios de dominio, ruido sensorial ni distribución fuera de muestra. El trabajo lo excluye en el alcance. Tampoco se analiza qué características visuales explican la diferencia, por lo que las explicaciones sobre descriptor global, geometría o distancia de dominio son hipótesis plausibles, no resultados demostrados.

### Formulación DDPM

La Ecuación (1) presenta el proceso directo con coeficientes acumulados. La Ecuación (3) lo reemplaza por una suma no escalada y la Ecuación (4) deja coeficientes sin definir. Debe corregirse antes de defensa, pues un tribunal técnico puede cuestionar si la implementación corresponde al DDPM descrito.

## 14. Auditoría de validación

La validación se organiza en cuatro capas:

1. **Pérdida de validación:** cuatro episodios de demostración. Sirve como señal exploratoria, pero su tamaño es débil.
2. **Selección de punto de control:** 50 condiciones de simulación reservadas, consultadas cada 50 épocas. Es un conjunto de selección, no una prueba final. La memoria lo denomina correctamente en el apartado 3.7.
3. **Prueba final:** 200 semillas disjuntas, evaluadas una vez tras congelar el punto de control. Esta capa proporciona independencia respecto de la selección según el procedimiento descrito.
4. **Validación externa:** inexistente. No hay otra tarea, robot, cámara, entorno o conjunto de datos.

La independencia de la tercera capa está correctamente planteada. El filtrado previo de candidatos por la métrica de selección no contamina el test; seleccionar es precisamente la función de ese bloque. La afirmación de la memoria de que el bloque disjunto «no elimina por completo» el sesgo debe matizarse: lo que permanece es una búsqueda limitada y desigual de candidatos, no una pérdida de independencia del test si este no se consultó.

El test final no resuelve dos niveles de incertidumbre. No repite el entrenamiento y no repite el ruido de inferencia. Por ello, valida cinco puntos de control en condiciones internas del simulador, no las estrategias ni el despliegue general.

La preespecificación del bloque final se afirma, pero no se prueba con un identificador fechado. La evaluación una sola vez y la ausencia de decisiones posteriores se aceptan como descripción del documento. Su verificación externa es **NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.**

## 15. Auditoría de reproducibilidad

**Clasificación: parcial.**

### Información disponible

- Origen general, formato y tamaño del conjunto (pp. 11-12).
- Reparto por episodios y semilla de partición (Tabla 3).
- Arquitectura generativa y dimensiones principales (pp. 14-15).
- Variantes, pesos nominales y parámetros totales y entrenables (Tabla 4).
- Hiperparámetros principales (Tabla 5).
- Semillas de selección y prueba (pp. 19-20).
- Hardware, sistema y versiones de bibliotecas (Tablas 6 y 7).
- Método estadístico, remuestreos y ajuste de multiplicidad (pp. 20-21).

### Información insuficiente o ausente

- Identificador exacto, versión, URL y hash del fichero de datos.
- Commit o etiqueta de la bifurcación propia.
- Identificadores y hashes exactos de pesos V1-V4.
- Resolución completa de la contradicción `timm`/`torchvision` para V1/V2.
- Transformaciones exactas, incluido el paso 96 a 76, interpolación y normalización.
- Parámetros completos de AdamW y planificador DDPM.
- Configuraciones efectivas que las referencias `??` debían localizar.
- Regla de parada uniforme y automática; V3 se detuvo por inspección.
- Activación de algoritmos deterministas. El documento declara que no están activados.
- Artefactos finales, hashes de puntos de control y manifiesto de resultados.
- Registro verificable de preespecificación del test.

### Prueba de reproducibilidad mental

Con el PDF, un investigador podría reconstruir una aproximación razonable al experimento y entender el análisis. No podría garantizar los mismos pesos iniciales, transformaciones, puntos de control ni decisiones de parada. Tampoco podría comprobar que el repositorio mencionado corresponde exactamente a esta versión de la memoria. Obtener resultados comparables es plausible; obtener el mismo experimento de forma auditable no lo es.

La no determinación reconocida y la única semilla agravan el problema. La reproducción exacta no es exigible siempre en GPU, pero deben preservarse artefactos, configuraciones e identificadores para distinguir variación numérica de cambios de procedimiento.

## 16. Evaluación del estado del arte

### Cobertura

El capítulo 2 cubre aprendizaje por imitación, multimodalidad, DDPM, modelos basados en puntuación, DDIM, Diffusion Policy, extensiones geométricas, aceleración y representaciones visuales. Para la pregunta del TFM, la cobertura es adecuada. Las Tablas 1 y 2 permiten comparar arquitecturas, estrategias y componentes modificados.

### Actualidad y calidad

Las 29 referencias numeradas abarcan trabajos fundacionales y publicaciones recientes hasta 2025. Se emplean artículos de revista, conferencias principales, TMLR, JMLR y preprints. Los clásicos antiguos se usan con función fundacional y no deben penalizarse. Varias extensiones recientes solo figuran como arXiv; el texto identifica ese estatus.

No se han consultado fuentes externas para confirmar metadatos. La exactitud bibliográfica individual más allá de lo visible en el PDF es **NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.**

### Comparabilidad y síntesis crítica

La revisión no se limita a enumerar autores. Compara modalidad, tarea, estrategia de adaptación, preservación de geometría y coste. Explica por qué DP3, HDP o ET-SEED no aíslan un codificador 2D y por qué R3M, MVP, Voltron y CortexBench no cierran el hueco concreto. Esta síntesis es una fortaleza.

### Gap

El hueco se formula de manera prudente: «en el corpus revisado no se ha identificado» la comparación conjunta (p. 9). Esta redacción evita una afirmación universal. Falta documentar cómo se construyó el corpus, con bases consultadas, términos, fechas y criterios de inclusión. Por ello, el gap está razonado, pero no posee garantía de exhaustividad sistemática.

### Marco conceptual

Las definiciones de codificador congelado, ajuste fino, ViT, DINOv2 y CLIP son suficientes. La formulación matemática pierde coherencia en el capítulo metodológico: la Ecuación (3) no coincide con el proceso directo presentado en la Ecuación (1). Este defecto debe corregirse.

### Citas y bibliografía

Las referencias [1]-[29] parecen emplearse en el cuerpo y mantienen orden numérico. La sección separada «Bibliografía» contiene dos obras propias sin integración en el sistema IEEE. Los dos `??` constituyen errores de referencia interna, no de fuente externa.

## 17. Resultados y discusión

### Lo que muestran los datos

Sobre las 200 condiciones finales y una realización sincronizada de ruido por condición, los puntos de control obtienen:

| Variante | Puntuación | IC 95 % informado | Éxitos |
|---|---:|---:|---:|
| V0 | 0,872 | [0,838; 0,906] | 101/200 |
| V1 | 0,649 | [0,597; 0,700] | 37/200 |
| V2 | 0,586 | [0,529; 0,642] | 32/200 |
| V3 | 0,578 | [0,523; 0,633] | 32/200 |
| V4 | 0,490 | [0,434; 0,547] | 16/200 |

Las diferencias V0-V1/V4 oscilan entre 0,223 y 0,382 y sus IC BCa no incluyen cero (Tabla 12). Los valores p ajustados son inferiores a 2,2 por 10 a la menos 16 para V0-V2/V4 y 1,8 por 10 a la menos 12 para V0-V1. Entre preentrenadas, solo V1-V4 supera claramente el umbral familiar.

Las curvas muestran sobreajuste en V1-V4 y estabilidad relativa de V0. El error de acción no conserva la ordenación de la puntuación, lo que demuestra que la pérdida supervisada no sustituye la evaluación en bucle cerrado (p. 29).

V0 requiere 1,6 min por época; V1, 5,3; V2, 14,7; V3, 2,3; V4, 4,0 (Tabla 8). La latencia total de la política cambia poco, mientras el codificador aislado de V4 es más lento. V2 excede la memoria física informada durante entrenamiento (Tablas 13 y 14).

### Lo que afirma el autor

El autor afirma que ninguno de los bloques preentrenados mejora V0, que el ajuste fino no supera a la congelación, que el orden de las variantes preentrenadas no queda resuelto, que los ViT encarecen el codificador pero no la llamada completa y que V0 domina en rendimiento y coste. También propone que la distancia de dominio y la pérdida de localización espacial pueden explicar el resultado, y sostiene que en V2 la actualización degrada la representación inicial (pp. 35-37).

### Lo que puede concluirse legítimamente

1. Entre los cinco puntos de control concretos, V0 obtiene una puntuación media mayor en la muestra de 200 condiciones finales bajo el ruido sincronizado definido.
2. Las diferencias V0-V1/V4 son grandes respecto de la escala y no dependen del máximo del conjunto de selección.
3. V1-V4 no ofrecen una ordenación estadística familiar completa; la ausencia de significación no implica equivalencia.
4. V0 registra menor tiempo por época en el equipo y entorno declarados.
5. Con 100 pasos DDPM, la red generativa domina la latencia total, de modo que las diferencias del codificador quedan diluidas en esa configuración.
6. Las comparaciones con V0 corresponden al bloque visual completo, no al preentrenamiento aislado.

### Lo que todavía no está demostrado

1. Que volver a entrenar V0 produzca en promedio mejores modelos que volver a entrenar V1-V4.
2. Que el preentrenamiento, por sí mismo, cause el menor rendimiento.
3. Que congelar sea superior a ajustar, porque existe una sola ejecución y una contradicción sobre la fuente de pesos V1/V2.
4. Que V1, V2 y V3 sean equivalentes.
5. Que el ajuste fino degrade internamente la representación; no se analizan características ni geometría aprendida.
6. Que el resultado se mantenga con otra semilla de inferencia, otras tareas, más demostraciones, robot físico o imágenes reales.
7. Que el protocolo final estuviera preespecificado en una versión anterior verificable.

La discusión es más que una repetición de resultados: contrasta hipótesis, presenta explicaciones, compara con R3M y DINOv3-DP y reconoce factores confundidos. Su principal debilidad es que algunas explicaciones mecanísticas se redactan como conclusión sin análisis específico de representaciones o errores.

## 18. Cumplimiento de objetivos

### Objetivo general

**Estado: parcialmente cumplido.** La influencia de cinco configuraciones visuales sobre cinco puntos de control se evalúa con una prueba final adecuada. No se estima la influencia esperada de las estrategias de entrenamiento porque solo existe una ejecución por variante. La palabra «influencia» debe limitarse a asociación experimental entre configuración y artefacto obtenido.

### Objetivo específico 1

**Enunciado:** implementar cinco variantes dentro de la misma Diffusion Policy, manteniendo constantes arquitectura de control y configuración.

**Verificabilidad:** las Tablas 4, 9 y 11 y las curvas muestran cinco variantes operativas. La red generativa y la dimensión de condicionamiento se declaran comunes.

**Resultado:** cumplido en cuanto a existencia documental de cinco resultados. No se mantiene toda la configuración: cambian presupuesto, lote, resolución, preprocesado y agregación. Algunas diferencias son inherentes a la variante, pero deben formar parte del enunciado.

**Clasificación:** cumplido con salvedades. La correspondencia exacta con una implementación funcional es **NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA.**

### Objetivo específico 2

**Enunciado:** comparar rendimiento mediante puntuación media.

**Verificabilidad:** Tabla 11 ofrece la puntuación final y Tabla 12 las diferencias pareadas.

**Resultado:** cumplido para los puntos de control fijos. La separación selección-prueba y el emparejamiento son adecuados.

**Clasificación:** parcialmente cumplido para las estrategias. Faltan réplicas de entrenamiento e inferencia, y la comparación no es equilibrada en presupuesto.

### Objetivo específico 3

**Enunciado:** analizar coste mediante parámetros, tiempos y memoria.

**Verificabilidad:** Tablas 4, 8, 13 y 14 cubren parámetros, tiempo por época, latencia y memoria.

**Resultado:** se distingue coste por época de tiempo total y entrenamiento de inferencia. La conclusión se limita al hardware declarado.

**Clasificación:** cumplido.

## 19. Evaluación de conclusiones

### Conclusiones individualizadas

1. **«Las cinco variantes funcionan dentro de la misma Diffusion Policy». SUSTENTADA en el plano documental.** Existen métricas, curvas y costes para V0-V4 (Tablas 8-14). La ejecución no se comprueba fuera del PDF.
2. **«El primer objetivo se cumple manteniendo la configuración». PARCIALMENTE SUSTENTADA.** La red generativa y los datos son comunes, pero presupuesto, lote y bloque de preprocesado difieren (pp. 15-17).
3. **«V0 supera a las cuatro variantes preentrenadas en la prueba final». SUSTENTADA para los cinco puntos de control.** Tabla 11 y diferencias de Tabla 12.
4. **«Ningún bloque preentrenado mejora el rendimiento». PARCIALMENTE SUSTENTADA.** Es correcta para estas cuatro configuraciones y esta ejecución; no estima futuras ejecuciones de las estrategias.
5. **«La prueba final es independiente de la selección». SUSTENTADA por el diseño descrito.** Semillas y secuencia de decisiones están separadas (pp. 19-20). La ejecución real y el registro previo son no verificables con el PDF.
6. **«El ajuste fino no supera a la congelación». SUSTENTADA solo para V1 y V2 concretas.** 0,586 frente a 0,649. No demuestra superioridad general de congelar y existe ambigüedad sobre pesos iniciales.
7. **«La actualización de los pesos degrada la representación de partida». NO SUSTENTADA.** No se mide calidad de representación; rendimiento y pérdidas admiten otras explicaciones (p. 36).
8. **«El orden interno de las preentrenadas sigue sin resolverse». SUSTENTADA.** La mayoría de comparaciones no supera Holm y no hay diseño de equivalencia (Tabla 12).
9. **«La distancia de dominio explica el resultado». EVIDENCIA DÉBIL.** Se presenta como explicación plausible y se conecta con antecedentes, pero no se manipula el dominio ni se analizan características.
10. **«La atribución corresponde al bloque visual completo, no a la inicialización». SUSTENTADA.** Los factores confundidos se enumeran de forma explícita (pp. 15-16 y 36).
11. **«V0 tiene el menor coste por época». SUSTENTADA en la plataforma declarada.** Tabla 8.
12. **«Congelar no garantiza entrenar más barato». SUSTENTADA para estas configuraciones.** V1, V3 y V4 tienen menos parámetros entrenables nominales y mayor tiempo por época que V0.
13. **«El codificador apenas influye en la latencia de la política». SUSTENTADA dentro de 100 pasos DDPM y el hardware medido.** Tabla 13; no debe generalizarse a muestreadores acelerados.
14. **«Ninguna variante alcanza tiempo real en este equipo». SUSTENTADA bajo la definición de 10 Hz y ocho acciones usada.** El coste por llamada excede el tiempo de consumo de las acciones (pp. 30-31).
15. **«El ajuste fino es lo que agota la memoria». PARCIALMENTE SUSTENTADA.** V2 es la única variante ajustada comparable y supera 8 GiB, pero no existe un diseño que varíe ajuste fino manteniendo todas las demás condiciones para cada arquitectura.
16. **«Con 90 demostraciones, V0 domina en rendimiento y coste». PARCIALMENTE SUSTENTADA.** Es válida para los cinco artefactos y la plataforma; excede la evidencia si se interpreta como propiedad esperada de las estrategias.

### Matriz de trazabilidad de conclusiones

| Conclusión | Resultado que la sustenta | Método | Evidencia | Validez |
|---|---|---|---|---|
| V0 obtiene mayor puntuación que V1-V4 | Tabla 11 y diferencias V0-V1/V4 de Tabla 12 | Prueba final pareada sobre 200 condiciones | **Fuerte para artefactos fijos** | Válida dentro del simulador, el bloque de ruido y los puntos de control evaluados |
| El test no intervino en selección | Semillas disjuntas y flujo de Figura 2 | Congelación previa y evaluación única declarada | **Moderada** | Diseño correcto; registro temporal no verificable en el PDF |
| El preentrenamiento no aporta ventaja | V1-V4 por debajo de V0 | Comparación de configuraciones completas | **Moderada** | Válida para los bloques; no aísla preentrenamiento ni variabilidad de entrenamiento |
| Ajuste fino no mejora congelación | V2=0,586; V1=0,649; diferencia 0,063 | Comparación V1-V2 | **Débil a moderada** | Una semilla y fuente de pesos ambigua; no demuestra superioridad de congelar |
| Variantes preentrenadas no tienen orden resuelto | Cinco de seis comparaciones sin significación familiar | Wilcoxon y Holm | **Moderada** | Ausencia de diferencia no demuestra equivalencia |
| V0 es más barata por época | Tabla 8 | Cronometraje en plataforma común | **Fuerte en la plataforma** | No generalizable a otro hardware o implementación |
| ViT encarece el codificador, no la llamada completa | Tabla 13 | Mediana de 50 repeticiones alternadas | **Fuerte en la configuración** | Condicionada a 100 pasos DDPM y entorno Windows declarado |
| V2 presiona la memoria de entrenamiento | Tabla 14, 10,293 GiB | Doce pasos completos en WSL | **Moderada a fuerte** | Medición local; extrapolación a otras GPU no válida |
| El ajuste fino degrada la representación | Pérdidas y puntuación de V2 | Inferencia mecanística sin análisis de representación | **Sin evidencia directa** | No válida como mecanismo demostrado |
| El dominio sintético explica la desventaja | Comparación narrativa con antecedentes | Discusión teórica | **Débil** | Hipótesis para trabajo futuro, no conclusión experimental |
| El resultado se generaliza a estrategias | No hay resultados entre semillas de entrenamiento | Una ejecución por variante | **Sin evidencia suficiente** | No válida |
| El resultado se generaliza a robótica real | No hay robot físico ni imágenes reales | Ninguno | **Sin evidencia** | No válida y correctamente excluida del alcance |

## 20. Limitaciones

### Reconocidas por el autor

1. **Una semilla de entrenamiento por variante** (pp. 2, 11, 20, 28 y 33). Se reconoce como la limitación más restrictiva.
2. **Presupuestos de épocas y oportunidades de evaluación desiguales** (pp. 16-17 y 33).
3. **Parada post hoc y detención manual de V3** (p. 17).
4. **Filtrado de candidatos por el conjunto de selección** (p. 20 y p. 33). El efecto sobre la independencia del test está descrito de forma excesivamente cauta, pero la limitación de candidatos sí existe.
5. **Evaluación discreta cada 50 épocas** (p. 34).
6. **Reanudación y extrapolación parcial del tiempo de V2** (pp. 23, 26 y 34).
7. **Confusión entre codificador y preprocesado** (pp. 15-16 y 36).
8. **Simulación, ausencia de robot físico y de perturbaciones visuales** (p. 2 y p. 38).
9. **Ausencia de determinismo exacto en GPU** (p. 22).
10. **Uso de 90 demostraciones y descarte de 112** (pp. 12 y 37).

El reconocimiento es amplio y específico. No debe penalizarse la mera existencia de estas limitaciones. Se penaliza que algunas afectan directamente al objetivo y no se corrigen en el diseño, y que ciertas conclusiones no siempre mantienen la restricción a artefactos fijos.

### Detectadas por el tribunal

1. **Una sola realización de inferencia por condición.** La incertidumbre del muestreo DDPM no se estima (p. 20).
2. **Contradicción sobre V1/V2.** `timm` y `torchvision` cuestionan la igualdad de pesos iniciales (pp. 15-16).
3. **Error en las ecuaciones DDPM.** Ecuación (3) incompatible con Ecuación (1); Ecuación (4) insuficientemente definida (pp. 3 y 14).
4. **Validación con cuatro episodios.** La pérdida de validación y el diagnóstico de sobreajuste son inestables (Tabla 3 y Figura 4).
5. **Desajuste Wilcoxon-media.** El contraste no corresponde de forma directa al estimando de diferencia media (pp. 20-21).
6. **Hipótesis de comparabilidad sin margen.** No se puede demostrar equivalencia por no significación (p. 11).
7. **Controles de integridad de datos incompletos.** Missing, duplicados y rangos no se documentan (pp. 11-14).
8. **Reproducibilidad sin versiones inmutables.** No hay commit, hash de datos ni identificadores exactos de pesos (pp. 21-22 y 47).
9. **Referencias cruzadas rotas y texto de plantilla.** Páginas iii, 16 y 22.
10. **Generalización del estimando.** Las 200 semillas estiman condiciones internas del mismo simulador; no tareas, equipos o dominios nuevos.

## 21. Cumplimiento formal

Esta sección se separa de la calidad científica. Solo se evalúan elementos visibles en el PDF frente a los requisitos Tecnun incorporados al agente.

| Requisito | Estado | Evidencia y observación |
|---|---|---|
| Portada | Cumple | Página PDF 1: título, autor, director, programa, lugar y fecha |
| Primera hoja de firma | Cumple | Página PDF 3 |
| Agradecimientos opcionales | No cumple en su forma actual | Página iii contiene texto de plantilla. Debe redactarse o eliminarse |
| Índice de contenidos | Cumple | pp. iv-v |
| Índice de figuras | Cumple | p. vii |
| Índice de tablas | Cumple | p. ix |
| Resumen | Cumple en estructura | p. xi, español, palabras clave y extensión visual inferior a 500 palabras |
| Abstract | Cumple en estructura | p. xiii, inglés, keywords y extensión visual inferior a 500 palabras |
| Cuerpo numerado | Cumple | Seis capítulos con jerarquía consistente, principalmente hasta tercer nivel |
| Presupuesto | Cumple | Capítulo 6, pp. 39-42 |
| Referencias | Cumple parcialmente | Lista IEEE [1]-[29], seguida de Bibliografía separada no integrada |
| Anexos | No aplicable | No se identifican anexos necesarios en el alcance declarado |
| DIN A4 y márgenes reglamentarios | No verificable con precisión métrica | NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA. |
| Tipografía Arial o equivalente y cuerpo 11 | Apariencia coherente, medida exacta no verificable | NO VERIFICABLE CON LA INFORMACIÓN PROPORCIONADA. |
| Texto justificado e interlineado sencillo | Cumple visualmente | Cuerpo principal |
| Figuras y tablas con número, título y fuente | Cumple en general | Índices, Tablas 1-19 y Figuras 1-5 |
| Legibilidad | Cumple en general | Tablas y figuras legibles. Figura 2 contiene texto compacto, pero interpretable |
| Referencias cruzadas | No cumple | Dos apariciones de `??` en pp. 16 y 22 |
| Ausencia de texto de plantilla | No cumple | Agradecimientos, p. iii |

La estructura general es sobria, ordenada y consistente. Las tablas se citan en el texto y la mayoría aporta información necesaria. Los defectos formales determinantes son fáciles de localizar y corregir: página de agradecimientos, dos referencias sin resolver, integración de la bibliografía y aclaración 96/76 píxeles.

## 22. Correcciones prioritarias

### PRIORIDAD 1: imprescindibles antes de defensa

1. **Resolver la unidad experimental de estrategia.** Repetir cada variante con varias semillas de entrenamiento bajo el mismo protocolo. Si no es viable, reformular título, objetivo general, hipótesis y conclusiones como comparación de cinco entrenamientos o puntos de control concretos.
2. **Uniformar el protocolo.** Fijar antes de nuevas ejecuciones el mismo número de actualizaciones, evaluaciones y regla de parada. V3 no debe quedar truncada de forma manual respecto de las demás.
3. **No reutilizar como prueba confirmatoria el bloque ya consultado para decidir cambios.** Si se realizan nuevos entrenamientos o se modifican hipótesis tras leer las semillas 200.000 a 200.199, reservar un nuevo bloque final completamente ajeno a esas decisiones.
4. **Corregir las Ecuaciones (3) y (4).** Definir el proceso directo e inverso DDPM con coeficientes consistentes con la Ecuación (1) y el planificador empleado.
5. **Resolver la identidad V1/V2.** Demostrar que parten de los mismos pesos y transformaciones o retirar la afirmación de que el contraste aísla exclusivamente congelación frente a ajuste fino.
6. **Ajustar las conclusiones.** Eliminar «degrada la representación» o presentarlo como hipótesis. Restringir la superioridad de V0 a los artefactos y condiciones evaluados hasta disponer de réplicas.
7. **Corregir el PDF final.** Eliminar texto de plantilla y resolver los dos `??`.

### PRIORIDAD 2: necesarias para mejorar validez

1. Replicar cada condición inicial con varias semillas de difusión y definir el estimando conjunto de condición y ruido.
2. Aumentar la validación por episodio y cuantificar la incertidumbre de la pérdida de validación.
3. Sustituir o complementar Wilcoxon con inferencia dirigida a la diferencia media. Definir el método de todos los IC.
4. Definir un margen de equivalencia si se mantiene la hipótesis de «rendimiento comparable».
5. Ejecutar ablaciones que igualen resolución, recorte y agregación para separar preentrenamiento de preprocesado.
6. Documentar controles de integridad del conjunto y caracterizar también las 200 condiciones finales, no solo las 50 de selección.
7. Archivar código, configuración, datos, pesos y protocolo con identificadores inmutables y hashes.

### PRIORIDAD 3: recomendadas para elevar calidad

1. Integrar repositorio y monografía en una única lista de referencias solo si se citan en el cuerpo.
2. Aclarar la secuencia 96 a 76 píxeles y todas las transformaciones por variante.
3. Añadir intervalos de la tasa de éxito y declarar su carácter secundario.
4. Presentar análisis de errores por regiones de posición, orientación y dificultad de condición.
5. Evaluar otra tarea y, si el alcance lo permite, imágenes reales o perturbaciones visuales.
6. Añadir una tabla final que separe de forma explícita conclusiones sobre artefactos, estrategias y generalización externa.

## 23. Preguntas previsibles del tribunal

### Nivel 1: comprensión

**Pregunta 1**

**Pregunta:** ¿Cuál es la aportación exacta del trabajo si no puede aislarse el preentrenamiento del preprocesado?

**Por qué la formularía el tribunal:** el título y la primera hipótesis destacan el codificador y el preentrenamiento, mientras el apartado 3.4 reconoce cuatro factores confundidos.

**Elementos mínimos de una buena respuesta:** definir la aportación como comparación de cinco bloques visuales completos; distinguir resultado descriptivo de efecto causal; citar V0=0,872 y el rango 0,490-0,649 de las alternativas; reconocer qué ablaciones faltan.

**Repregunta:** ¿Qué único entrenamiento adicional separaría mejor la inicialización del preprocesado?

**Dificultad:** 1/5.

**Parte del TFM relacionada:** pp. 1-2, 15-16 y 35-37.

**Pregunta 2**

**Pregunta:** ¿Cuál es el resultado principal y cuál es su población válida?

**Por qué la formularía el tribunal:** permite comprobar que no se generaliza de Push-T a manipulación robótica en general.

**Elementos mínimos de una buena respuesta:** cinco puntos de control fijos; 200 condiciones del generador Push-T; una realización sincronizada de ruido; V0 superior en esa muestra; exclusión de otras tareas, robots y reentrenamientos.

**Repregunta:** ¿Por qué 200 condiciones no equivalen a 200 repeticiones del experimento?

**Dificultad:** 1/5.

**Parte del TFM relacionada:** p. 20, Tabla 11 y apartado 4.8.

### Nivel 2: metodología

**Pregunta 3**

**Pregunta:** ¿Por qué una sola semilla de entrenamiento impide concluir que una estrategia es superior aunque haya 200 condiciones de prueba?

**Por qué la formularía el tribunal:** es la principal amenaza a la inferencia sobre estrategias.

**Elementos mínimos de una buena respuesta:** distinguir variabilidad entre condiciones de evaluación y entre entrenamientos; identificar la ejecución como unidad experimental de estrategia; explicar operaciones no deterministas e inicialización; proponer varias semillas y análisis jerárquico.

**Repregunta:** ¿Cuántas semillas usaría y cómo asignaría el presupuesto si solo pudiera añadir diez entrenamientos?

**Dificultad:** 2/5.

**Parte del TFM relacionada:** pp. 11, 20, 28 y 33.

**Pregunta 4**

**Pregunta:** ¿Cómo afecta a la comparación que V3 se detuviera en 155 épocas y V0 llegara a 500?

**Por qué la formularía el tribunal:** el criterio de parada fue post hoc y no se aplicó de modo uniforme.

**Elementos mínimos de una buena respuesta:** reconocer menor oportunidad de convergencia y selección; citar la lectura común a época 150 y K=4 como análisis de sensibilidad; explicar por qué mitiga pero no resuelve; proponer igual número de actualizaciones y parada automática.

**Repregunta:** ¿Debe igualarse el número de épocas, de actualizaciones o el coste de cómputo? Justifique el estimando elegido.

**Dificultad:** 2/5.

**Parte del TFM relacionada:** pp. 16-17, Tablas 8-10 y p. 33.

### Nivel 3: análisis de datos

**Pregunta 5**

**Pregunta:** Si el estimando es una diferencia media, ¿por qué se utilizó Wilcoxon y qué contrasta exactamente esa prueba?

**Por qué la formularía el tribunal:** existe un desajuste entre la cantidad reportada y el contraste.

**Elementos mínimos de una buena respuesta:** Wilcoxon opera sobre rangos de diferencias pareadas y no es una prueba directa de la media; requiere precisión sobre hipótesis de localización; los IC BCa sí estiman la diferencia media; proponer permutación de la media o bootstrap y mantener corrección de multiplicidad.

**Repregunta:** ¿Cómo construiría intervalos simultáneos para las diez diferencias?

**Dificultad:** 3/5.

**Parte del TFM relacionada:** pp. 20-21 y Tabla 12.

**Pregunta 6**

**Pregunta:** Escriba correctamente el estado perturbado de un DDPM en el paso k y explique qué falta en la Ecuación (3).

**Por qué la formularía el tribunal:** la ecuación metodológica contradice el fundamento presentado en la Ecuación (1).

**Elementos mínimos de una buena respuesta:** expresar `A^k = sqrt(alpha_bar_k) A^0 + sqrt(1-alpha_bar_k) epsilon`; explicar la función del plan de ruido; distinguir entrenamiento de transición inversa; reconocer que `A^0 + epsilon^k` es insuficiente.

**Repregunta:** ¿Cómo se relacionan los coeficientes de la Ecuación (4) con beta, alfa y alfa acumulada?

**Dificultad:** 3/5.

**Parte del TFM relacionada:** pp. 3-4 y p. 14, Ecuaciones (1)-(4).

### Nivel 4: pensamiento crítico

**Pregunta 7**

**Pregunta:** ¿Qué conclusión cambiaría más probablemente al repetir el estudio con otras semillas de entrenamiento?

**Por qué la formularía el tribunal:** obliga a jerarquizar incertidumbres y no repetir la lista de limitaciones.

**Elementos mínimos de una buena respuesta:** la ordenación interna V1-V3 y el contraste congelación-ajuste son más frágiles; las diferencias V0-V1/V4 son grandes pero tampoco están protegidas frente a variabilidad de entrenamiento; diferenciar magnitud observada de robustez no medida.

**Repregunta:** ¿Qué patrón entre réplicas bastaría para retirar la conclusión principal?

**Dificultad:** 4/5.

**Parte del TFM relacionada:** Tablas 11-12 y apartado 4.8.

**Pregunta 8**

**Pregunta:** ¿Qué se gana y qué se pierde al sincronizar el ruido de difusión entre variantes?

**Por qué la formularía el tribunal:** el uso de números aleatorios comunes es correcto, pero puede confundirse con una caracterización completa de incertidumbre.

**Elementos mínimos de una buena respuesta:** se reduce varianza de la diferencia pareada y se mejora comparabilidad; el resultado queda condicionado a realizaciones concretas; no se estima variabilidad de inferencia; proponer varias réplicas comunes por condición.

**Repregunta:** ¿Cómo definiría y estimaría el rendimiento esperado sobre condiciones y ruido sin destruir el emparejamiento?

**Dificultad:** 4/5.

**Parte del TFM relacionada:** pp. 20-21.

### Nivel 5: tribunal estricto

**Pregunta 9**

**Pregunta:** ¿Cómo puede sostener que V1 y V2 aíslan la estrategia si una se describe mediante `timm` y la otra mediante `torchvision`?

**Por qué la formularía el tribunal:** la validez de la segunda hipótesis depende de la igualdad exacta de inicialización y preprocesado.

**Elementos mínimos de una buena respuesta:** identificar los modelos y pesos exactos; demostrar igualdad tensor a tensor o reconocer la confusión; explicar transformaciones; retirar la atribución causal si no puede probarse.

**Repregunta:** Si los pesos fueran distintos, ¿qué parte de la Tabla 12 seguiría siendo válida y qué interpretación dejaría de serlo?

**Dificultad:** 5/5.

**Parte del TFM relacionada:** pp. 15-16 y comparación V1-V2 de Tabla 12.

**Pregunta 10**

**Pregunta:** El bloque de 200 semillas ya se ha consultado. Si ahora repite entrenamientos, corrige hiperparámetros o añade ablaciones, ¿cómo evita convertirlo en conjunto de selección?

**Por qué la formularía el tribunal:** prueba la comprensión de independencia prospectiva y reutilización del test.

**Elementos mínimos de una buena respuesta:** congelar un nuevo protocolo antes de ejecutar; usar selección separada; reservar nuevas semillas finales nunca consultadas; mantener el bloque actual como resultado de la primera fase, no como test confirmatorio de modelos modificados; registrar una versión fechada.

**Repregunta:** ¿Qué decisiones pueden tomarse después de observar el test sin invalidarlo y cuáles obligan a reservar otro bloque?

**Dificultad:** 5/5.

**Parte del TFM relacionada:** Figura 2, pp. 19-20, apartado 4.3 y trabajo futuro de pp. 37-38.

## 24. Veredicto final

**¿Está listo para defensa? No todavía.**

El PDF ya contiene una prueba final disjunta y un tratamiento estadístico más riguroso que una selección por máximo. Ese avance sostiene la afirmación limitada de que V0 fue el mejor de los cinco puntos de control concretos sobre las 200 condiciones finales. No se detecta evidencia de que las semillas de prueba hayan seleccionado esos puntos ni de leakage entre los tres bloques.

Antes de la defensa deben resolverse, como mínimo, cuatro cuestiones. La primera es alinear el alcance de las conclusiones con una sola semilla de entrenamiento o añadir réplicas. La segunda es corregir la desigualdad y el carácter post hoc del entrenamiento si se pretende comparar estrategias. La tercera es aclarar la identidad de pesos y preprocesado de V1/V2. La cuarta es corregir las ecuaciones DDPM y los defectos documentales de reproducibilidad. La página de plantilla y los dos `??` también deben desaparecer de la versión final.

**Respuesta a la pregunta final del tribunal:** **NO.** En su estado actual, el TFM no demuestra todavía, con validación suficiente del nivel de entrenamiento y un procedimiento plenamente reproducible, que las estrategias de codificación formuladas respalden todas las conclusiones dentro del alcance declarado. Sí demuestra, con evidencia moderada y una prueba final independiente según lo documentado, que el punto de control V0 supera a los otros cuatro artefactos evaluados en la configuración concreta de Push-T. Esta evidencia es insuficiente para atribuir la ventaja al entrenamiento desde cero como estrategia general, para aislar el preentrenamiento o para sostener mecanismos de degradación de la representación.
