# INFORME DEL TRIBUNAL — REEVALUACIÓN INTEGRAL

**Trabajo:** *Comparación de codificadores visuales en Diffusion Policy: protocolo con prueba disjunta, rendimiento y coste en Push-T*
**Tipo:** Trabajo Fin de Máster — Máster Universitario en Análisis de Datos en Ingeniería, Tecnun, Universidad de Navarra.
**Versión evaluada:** la versión más reciente, con **69 páginas numeradas de memoria**, ampliada con la comparación frente al artefacto publicado y el ensayo de transferencia a un segundo simulador. 

Mantengo fuera de esta evaluación la rúbrica de defensa perteneciente a la otra universidad. La valoración se realiza con los criterios técnicos y metodológicos del agente evaluador, aplicados específicamente a un TFM experimental de machine learning, control visuomotor y análisis de datos.

---

# 1. Dictamen preliminar

## **APROBABLE SIN CAMBIOS RELEVANTES**

### **Puntuación global: 93 / 100**

Esta versión supera cualitativamente la anterior. Ya no encuentro ninguna corrección documental que condicione la defensa ni ningún problema metodológico que obligue a repetir experimentos.

Las limitaciones importantes permanecen —una sola semilla de entrenamiento, presupuestos de entrenamiento desiguales, interrupción temprana de V3 y ampliación post-preregistro de una a dos realizaciones del ruido—, pero ahora están **identificadas, cuantificadas cuando es posible y, sobre todo, incorporadas al alcance de las conclusiones**. Cap. 1.4, pp. 2–3; Cap. 4.10, pp. 53–54. 

Además, se han corregido puntos de versiones anteriores:

* el sorteo del dataset ya se formula correctamente distinguiendo ausencia de criterio sistemático y posible desequilibrio muestral por azar, Cap. 3.2, p. 14; 
* se diferencia el guardado técnico cada 10 épocas de V3/V4 de la evaluación en simulador cada 50 épocas, Tabla 8 y Cap. 3.5, pp. 21–22; 
* el Resumen distingue correctamente qué diferencias entre preentrenadas son robustas a ambos contrastes; Resumen, p. xi; 
* la comparación V0–Vpub ya no se presenta como equivalencia: se dice expresamente que V0 no rinde por debajo, pero tampoco queda demostrado que lo supere por una cantidad prácticamente relevante, Tabla 20 y Cap. 4.4, p. 42; 
* la Figura 5 explica que el valor mostrado 0,950 corresponde realmente a 0,94983 y que la puntuación 1,000 se debe al redondeo, evitando la aparente contradicción con el umbral de éxito, Fig. 5, p. 47; 
* las tarifas del presupuesto están ahora declaradas como tarifas convencionales de imputación y no como valores salariales obtenidos de una fuente externa, Cap. 6.4, p. 64. 

Por estas razones, mi dictamen cambia de **“aprobable con correcciones menores” a “aprobable sin cambios relevantes”**.

---

# 2. Resumen del trabajo evaluado

El trabajo compara cinco bloques visuales dentro de una misma Diffusion Policy sobre Push-T:

| Variante | Codificador        | Adaptación  |
| -------- | ------------------ | ----------- |
| V0       | ResNet-18          | Desde cero  |
| V1       | ResNet-18 ImageNet | Congelado   |
| V2       | ResNet-18 ImageNet | Ajuste fino |
| V3       | DINOv2 ViT-S/14    | Congelado   |
| V4       | CLIP ViT-B/16      | Congelado   |

La pregunta está formulada apropiadamente como una comparación de **cinco artefactos entrenados concretos**, no como una estimación del comportamiento esperado de las cinco estrategias si fueran reentrenadas. Esa distinción aparece desde la pregunta de investigación, el objetivo general y el alcance. Cap. 1.2–1.4, pp. 1–3. 

Se emplean 90 de 206 demostraciones para entrenamiento y cuatro para validación. La selección de checkpoints utiliza 50 condiciones reservadas; la prueba principal se realiza sobre otras 200 condiciones disjuntas. Cada condición final se ejecuta con dos realizaciones del ruido de difusión, promediadas antes del análisis, manteniendo la condición inicial como unidad estadística. La segunda realización es una ampliación posterior al preregistro y la memoria lo declara expresamente. Cap. 3.8, pp. 24–26. 

El resultado principal es:

$$
V0=0,887,\quad V1=0,642,\quad V3=0,580,\quad V2=0,571,\quad V4=0,490.
$$

Las diferencias V0–resto oscilan entre 0,245 y 0,397 y sobreviven a Holm con los dos procedimientos estadísticos utilizados. Tabla 19, pp. 40–41. 

El trabajo añade dos comprobaciones externas al contraste principal: V0 frente al checkpoint publicado Vpub y V0/Vpub sobre una reimplementación en otro motor físico. Esta segunda extensión está correctamente declarada como estadísticamente indeterminada debido a potencia insuficiente. Cap. 4.4–4.5, pp. 42–45. 

---

# 3. Puntuación

| Criterio                                 |    Máx. |     Nota | Dictamen                                                                                                            |
| ---------------------------------------- | ------: | -------: | ------------------------------------------------------------------------------------------------------------------- |
| A. Problema, justificación y alcance     |      10 |  **9,5** | Pregunta concreta y delimitación inferencial excepcionalmente clara.                                                |
| B. Objetivos y pregunta de investigación |      10 |  **9,5** | Objetivos medibles y trazables hasta resultados y conclusiones.                                                     |
| C. Estado del arte                       |      10 |  **9,0** | Suficiente, actual y directamente conectado con la elección experimental.                                           |
| D. Metodología                           |      15 | **14,0** | Diseño muy sólido; penalizan una semilla, presupuestos desiguales y decisiones durante la campaña.                  |
| E. Desarrollo técnico                    |      20 | **19,0** | Elevado nivel de detalle en arquitectura, preprocesamiento, configuración, reproducibilidad y simulación.           |
| F. Resultados, validación y discusión    |      15 | **14,5** | Prueba disjunta, incertidumbre, multiplicidad, sensibilidad al ruido y validación complementaria muy bien tratadas. |
| G. Evaluación del proyecto / coste       |      10 |  **8,5** | Completa y aritméticamente consistente; valores económicos son estimativos.                                         |
| H. Conclusiones y recomendaciones        |       5 |  **4,5** | Muy buena correspondencia con la evidencia y excelentes líneas futuras.                                             |
| I. Presentación académica                |       5 |  **4,5** | Documento estructurado, legible y técnicamente consistente.                                                         |
| **TOTAL**                                | **100** | **93,0** | **Nivel alto / muy alto de TFM.**                                                                                   |

---

# 4. Matriz de coherencia

| Objetivo                               | Evidencia                            | Resultado                                       | Estado                            |
| -------------------------------------- | ------------------------------------ | ----------------------------------------------- | --------------------------------- |
| Integrar cinco bloques visuales        | Tablas 5–8, pp. 19–22                | Cinco artefactos funcionales con interfaz común | **Cumplido**                      |
| Comparar rendimiento con incertidumbre | Fig. 2; Tablas 18–19, pp. 25 y 40–41 | Prueba disjunta, BCa, dos contrastes y Holm     | **Cumplido**                      |
| Caracterizar coste                     | Tablas 15, 23 y 24, pp. 34 y 50–51   | Tiempo, parámetros, latencia y memoria          | **Cumplido**                      |
| Delimitar validez                      | Cap. 4.10 y Tabla 25, pp. 53–60      | Distingue artefacto, estrategia y transferencia | **Cumplido con especial solidez** |

Especialmente destacable es el objetivo 4. El trabajo no emplea sus limitaciones como un descargo posterior: la delimitación de qué puede y qué no puede afirmarse forma parte explícita del diseño desde el Cap. 1.3–1.4. 

---

# 5. Fortalezas demostrables

### 5.1 Separación entre selección y prueba

La Figura 2 separa claramente entrenamiento, validación, selección y bloque final. Los 200 episodios finales no intervienen en la elección de los checkpoints. Esa es una de las decisiones más importantes del TFM porque evita presentar el máximo del conjunto de selección como resultado independiente. Cap. 3.8, Fig. 2, p. 25. 

### 5.2 El sesgo de selección no solo se menciona, se estima

La Tabla 17 evalúa época común, igual número de oportunidades \(K=4\) y optimismo mediante particiones 25/25. Incluso bajo esas lecturas V0 mantiene su ventaja. Cap. 4.1, Tabla 17, pp. 37–38. 

### 5.3 Resultado principal robusto

Las diferencias V0–V1/V2/V3/V4 son:

$$
0,245,\;0,316,\;0,307,\;0,397
$$

y todos sus IC95 % quedan holgadamente alejados de cero. Permutación y Wilcoxon coinciden en las cuatro comparaciones esenciales. Cap. 4.3, Tabla 19, pp. 40–41. 

Esto permite afirmar que **la conclusión principal no depende de escoger convenientemente un contraste estadístico**.

### 5.4 Muy buen tratamiento de multiplicidad

Las diez comparaciones forman una familia y se aplica Holm separadamente a los valores p de cada procedimiento. Además, la memoria distingue expresamente que los intervalos BCa no están corregidos por multiplicidad. Cap. 3.8, pp. 25–26; Tabla 19, p. 40. 

### 5.5 Correcta interpretación de resultados secundarios

La diferencia V1–V2 resulta significativa solo bajo permutación después de Holm, no bajo Wilcoxon. La memoria no transforma esta discrepancia en una conclusión fuerte. Lo mismo ocurre con V1–V3 y V2–V4. Cap. 4.3, pp. 40–41. 

### 5.6 Excelente delimitación causal

V0 y las preentrenadas no difieren únicamente en pesos: también cambian resolución, recorte, normalización y, en V3/V4, arquitectura. V1 y V2 ni siquiera cargan exactamente el mismo archivo de pesos iniciales. El documento no atribuye causalmente el resultado al preentrenamiento aislado. Cap. 3.4, Tablas 5–6, pp. 18–20. 

### 5.7 Tratamiento transparente de decisiones no preregistradas

El sign-flip sobre la media se declara post hoc; la segunda realización del ruido también. No se intenta reconstruir retrospectivamente un diseño aparentemente confirmatorio. Cap. 3.8, pp. 25–26; Cap. 4.10, pp. 53–54.  

### 5.8 Comparación con el modelo publicado bien planteada

V0 = 0,887 y Vpub = 0,846, diferencia = 0,041; el IC90 % \([0,011,0,074]\) no cabe dentro de ±0,05. La memoria concluye correctamente que no se establece equivalencia ni una ventaja prácticamente relevante. Cap. 4.4, Tabla 20, p. 42. 

### 5.9 El experimento Godot está estadísticamente bien acotado

Antes de interpretar el resultado se declara:

$$
SE\simeq0,038,
$$

$$
MDE\simeq0,105.
$$

Con 50 condiciones el estudio no puede detectar razonablemente la diferencia de 0,041 observada anteriormente ni demostrar equivalencia dentro de ±0,05. Cap. 3.10, pp. 29–31. 

Esto evita el error frecuente de interpretar \(p>0,05\) como «los modelos son iguales».

### 5.10 Buen uso del segundo simulador

La memoria mide que las nuevas observaciones solo modifican aproximadamente 2,4–2,7 % de los píxeles y limita la conclusión a un desplazamiento de dominio pequeño. Figuras 4–5 y Cap. 4.5, pp. 45–47. 

### 5.11 Reproducibilidad superior a la habitual en un TFM

Se documentan versiones de software, hashes de artefactos y dataset, revisión Git, pesos iniciales, semilla y configuración de ejecución. Tablas 12–14, pp. 32–33. 

---

# 6. Hallazgos críticos

## **No se identifican hallazgos CRÍTICOS.**

No encuentro evidencia de:

* fuga del bloque final hacia el entrenamiento;
* pseudorreplicación de 400 realizaciones como 400 unidades independientes;
* error aritmético capaz de alterar la conclusión;
* afirmaciones causales no permitidas por el diseño;
* equivalencia declarada a partir de ausencia de significación;
* ocultación de decisiones post hoc.

El resultado central es técnicamente defendible.

---

# 7. Observaciones mayores

## No existen observaciones MAYORES pendientes de corrección documental.

Sí permanecen **tres limitaciones estructurales de importancia**, que un tribunal probablemente discutirá.

### M1. Una única semilla de entrenamiento

Cada variante tiene una sola ejecución de entrenamiento. Las 200 condiciones proporcionan información sobre el comportamiento de cada checkpoint fijo, no sobre la variabilidad entre nuevos entrenamientos. Cap. 1.4, p. 2; Cap. 4.10, p. 53. 

Esto impide convertir:

> V0 superó a V1–V4

en:

> entrenar desde cero tiene mayor rendimiento esperado.

La memoria ya respeta correctamente esa frontera.

### M2. Presupuesto experimental desigual

V0/V1 completan 500 épocas, V2 266, V3 155 y V4 200. V3 se interrumpe incluso antes del mínimo establecido por la regla de parada. Cap. 3.5, Tabla 8, pp. 21–22; Cap. 4.10, p. 53.  

El análisis a época 150 y \(K=4\) reduce la preocupación, pero no transforma el estudio en una comparación de estrategias con igual presupuesto.

### M3. Ampliación post-preregistro

La segunda realización del ruido se decide después de observar la primera. La memoria la justifica, conserva el resultado preregistrado original y señala qué interpretaciones secundarias cambian. Cap. 3.8, pp. 25–26; Cap. 4.10, pp. 53–54. 

Esto disminuye el carácter puramente confirmatorio del análisis actualizado, pero no invalida la conclusión V0–resto porque esa conclusión permanece bajo ambas realizaciones y ambos contrastes.

---

# 8. Observaciones menores

Solo mantendría **dos observaciones menores**, ninguna necesaria para aprobar.

### m1. Definir explícitamente la «varianza intracondición»

**Ubicación:** Cap. 4.3–4.4, pp. 41–43.

Se reportan valores de 0,027 para V0 y 0,061–0,081 para las variantes preentrenadas, pero en el texto recuperado no aparece una ecuación que especifique exactamente el estimador utilizado para obtener esa «varianza intracondición» a partir de dos realizaciones. 

La interpretación cualitativa es razonable y no modifica la prueba principal, pero para reproducibilidad estadística añadiría una línea como:

$$
\hat{\sigma}^2_{\mathrm{intra}}
=\frac{1}{n}\sum_i \frac{(s_{i1}-s_{i2})^2}{2},
$$

**si esa es efectivamente la definición utilizada**.

La fórmula real del código:

**No verificable con la información proporcionada.**

### m2. Una expresión todavía puede ser ligeramente ambigua en la metodología de Vpub

En el Cap. 3.9 se indica que la comparación determina si el modelo obtenido «iguala al publicado». Posteriormente, resultados y conclusiones lo expresan mucho mejor y dicen explícitamente que **no se estableció equivalencia**. Cap. 3.9, pp. 27–28 frente a Cap. 4.4, p. 42 y Cap. 5.1, p. 56.  

No constituye ya un error conceptual porque el protocolo estadístico y las conclusiones son correctos. Si se quiere pulir al máximo el documento, cambiaría únicamente ese verbo de la metodología por:

> «determina cómo se sitúa el modelo obtenido respecto del publicado».

---

# 9. Auditoría técnica

### Dataset

Tabla 3:

$$
90+4+112=206
$$

y:

$$
11\,356+432+13\,862=25\,650.
$$

Coherente. Cap. 3.2, Tabla 3, p. 14–15. 

### Actualizaciones por época

$$
10\,726/64=167,59\Rightarrow168
$$

actualizaciones por época.

En V3/V4:

$$
336\text{ minibatches}/2=168
$$

actualizaciones. Coherente con Tabla 8. Cap. 3.5, pp. 20–22. 

### Diferencias principales

$$
0,887-0,642=0,245
$$

$$
0,887-0,571=0,316
$$

$$
0,887-0,580=0,307
$$

$$
0,887-0,490=0,397.
$$

Coinciden con Tabla 19. 

### V0 frente a Vpub

$$
(0,872+0,902)/2=0,887
$$

$$
(0,850+0,842)/2=0,846
$$

$$
0,887-0,846=0,041.
$$

Tabla 20 consistente. 

### Potencia en Godot

Con:

$$
s_d\approx0,267,\qquad n=50
$$

se obtiene aproximadamente:

$$
SE=\frac{0,267}{\sqrt{50}}=0,0378.
$$

Y con potencia 80 %:

$$
(1,96+0,84)(0,0378)\approx0,106,
$$

consistente con el MDE ≈ 0,105 declarado. Cap. 3.10, p. 30. 

### Coste temporal

Los valores declarados son V0 1,6 min/época, V1 5,3, V2 14,7, V3 2,3 y V4 4,0. El documento reconoce correctamente que el tiempo total no sirve para ordenar variantes porque los presupuestos fueron distintos. Cap. 3.12, pp. 34–35. 

### Presupuesto

$$
1800\frac{7}{48}=262,50€
$$

$$
180\frac{7}{60}=21€
$$

$$
60\frac{7}{36}=11,67€.
$$

Mano de obra:

$$
450(30)+40(60)=15\,900€.
$$

Subtotal:

$$
54+295,17+15\,900=16\,249,17€.
$$

Indirectos:

$$
0,15(16\,249,17)=2\,437,38€.
$$

Total:

$$
18\,686,55€.
$$

Tablas 27–30 consistentes. Cap. 6.2–6.5, pp. 63–65.  

### Reproducibilidad externa

La memoria proporciona hashes, versiones, IDs y revisión Git, pero no estoy ejecutando el repositorio ni recalculando BCa, Wilcoxon o las 10.000 permutaciones desde los vectores originales.

Por tanto, la exactitud externa de hashes, p-valores, bootstrap y resultados de cada episodio es:

**No verificable con la información proporcionada.**

Lo verificable es que el documento presenta una trazabilidad metodológica apropiada.

---

# 10. Cumplimiento de los objetivos

| Objetivo                                                     | Dictamen                        | Evidencia            |
| ------------------------------------------------------------ | ------------------------------- | -------------------- |
| 1. Implementar cinco bloques visuales                        | **CUMPLIDO**                    | Tablas 5–8           |
| 2. Comparar rendimiento en bloque disjunto con incertidumbre | **CUMPLIDO**                    | Fig. 2, Tablas 18–19 |
| 3. Medir cuatro dimensiones de coste                         | **CUMPLIDO**                    | Tablas 15, 23–24     |
| 4. Acotar validez e inferencia                               | **CUMPLIDO CON NIVEL MUY ALTO** | Cap. 4.10 y Tabla 25 |

A nivel de coherencia problema–objetivos–metodología–resultados–conclusiones, no encuentro ningún objetivo que haya quedado sin respuesta.

---

# 11. Evaluación de las conclusiones

| Afirmación                                                | Valoración                                  |
| --------------------------------------------------------- | ------------------------------------------- |
| V0 supera a V1–V4                                         | **SUSTENTADA**                              |
| Ningún bloque preentrenado evaluado mejora V0             | **SUSTENTADA para los cinco artefactos**    |
| Entrenar desde cero es generalmente mejor                 | **NO SUSTENTADA**, y no se afirma           |
| El preentrenamiento es la causa de la diferencia          | **NO SUSTENTADA**, correctamente descartada |
| V4 < V1                                                   | **SUSTENTADA con ambos contrastes**         |
| V4 < V3                                                   | **SUSTENTADA con ambos contrastes**         |
| V4 < V2                                                   | **DEPENDIENTE DEL CONTRASTE**               |
| V2 = V3                                                   | **NO DEMOSTRADO**                           |
| No se detecta diferencia V2–V3                            | **SUSTENTADO**                              |
| Congelar es mejor que ajustar como estrategia             | **NO SUSTENTADO**                           |
| V2 no mejoró V1 en esta ejecución                         | **SUSTENTADO**                              |
| V0 no rinde por debajo de Vpub bajo el contraste primario | **SUSTENTADO CON CAUTELA**                  |
| V0 y Vpub son equivalentes                                | **NO DEMOSTRADO**                           |
| V0 supera Vpub en >0,05                                   | **NO DEMOSTRADO**                           |
| Godot demuestra igualdad                                  | **NO DEMOSTRADO**                           |
| Godot es indeterminado por falta de potencia              | **SUSTENTADO**                              |
| Robustez ante pequeño desplazamiento visual medido        | **SUSTENTADA con alcance restringido**      |
| Robustez general de dominio                               | **NO SUSTENTADA**                           |
| V0 tiene menor coste por época                            | **SUSTENTADA en el equipo utilizado**       |
| Menor MSE implica mejor control                           | **CONTRADICHO por los resultados**          |

La Tabla 25 hace particularmente bien esta separación entre lo demostrado para artefactos, lo no demostrado para estrategias y lo no demostrado fuera del simulador. Cap. 5.1, Tabla 25, p. 60. 

---

# 12. Correcciones prioritarias

### Imprescindibles antes de defensa

**Ninguna.**

No exigiría al autor nuevos entrenamientos ni modificaciones metodológicas para permitir la defensa.

### Necesarias

**Ninguna de fondo.**

La memoria ya ha corregido los problemas de interpretación que podían generar objeciones metodológicas.

### Recomendadas

Solo haría dos retoques opcionales:

1. definir matemáticamente el estimador de «varianza intracondición»;
2. sustituir el verbo «iguala» que todavía aparece en la descripción metodológica de V0–Vpub, aunque los resultados posteriores ya eliminan correctamente cualquier afirmación de equivalencia.

No cambiarían nota ni dictamen.

---

# 13. Veredicto final

## **APROBABLE SIN CAMBIOS RELEVANTES — 93/100**

Considero que la memoria se encuentra **lista para defensa**.

El trabajo presenta una particularidad positiva: cuanto más complejo se volvió el análisis, más cuidadosa se volvió también la delimitación de las conclusiones. La introducción de un segundo ruido, el contraste contra Vpub y la transferencia a Godot podrían haber producido fácilmente sobreinterpretaciones. En la versión actual ocurre lo contrario: cada extensión lleva asociada su propia discusión de estimando, potencia, incertidumbre y alcance. 

La principal conclusión que un tribunal puede admitir es:

> **En las cinco ejecuciones concretas realizadas y sobre las 200 condiciones disjuntas ensayadas, el artefacto V0 entrenado desde cero obtuvo mayor rendimiento en bucle cerrado que los otros cuatro artefactos, con diferencias amplias y robustas a los dos contrastes empleados.**

Y la conclusión que el tribunal **no debería permitir** es:

> **Entrenar un codificador desde cero es generalmente superior al preentrenamiento en Diffusion Policy.**

El propio TFM distingue correctamente ambas afirmaciones.

El salto respecto de las primeras versiones es importante: las limitaciones siguen existiendo, pero ahora **no contradicen las conclusiones**. En metodología experimental eso es mucho más importante que intentar ocultarlas o compensarlas con más estadística.

---

# Preguntas que prepararía para la defensa

## Nivel 1 — Conceptual

### 1. ¿Cuál es exactamente la unidad de inferencia de tu estudio?

**Motivo:** comprobar que comprende la principal restricción metodológica.

**Respuesta mínima:** el artefacto entrenado/checkpoint. Las 200 condiciones permiten inferir sobre su comportamiento frente a nuevas condiciones iniciales, pero no estimar la distribución de resultados de volver a entrenar la estrategia.

**Repregunta:** ¿por qué 200 condiciones no equivalen a 200 réplicas experimentales?

**Dificultad:** 2/5.
**Ubicación:** Cap. 1.4, p. 2; Cap. 4.10, p. 53. 

---

## Nivel 2 — Diseño

### 2. ¿Por qué puede defenderse V0 frente a V3 si V3 solo entrenó 155 épocas?

**Respuesta mínima:** no puede afirmarse que la estrategia V3 haya recibido igualdad de oportunidad. Lo que se compara es el artefacto efectivamente obtenido. Como análisis de sensibilidad, a época común 150 y con \(K=4\) V0 ya mantiene una ventaja sustancial, pero eso no sustituye una réplica con presupuesto común.

**Repregunta:** ¿qué experimento eliminaría definitivamente esta objeción?

**Dificultad:** 4/5.
**Ubicación:** Tabla 8, pp. 21–22; Tabla 17, p. 38.  

---

### 3. ¿Por qué no utilizaste las 202 demostraciones disponibles para entrenamiento?

La respuesta debe explicar que los 90 episodios definen el régimen reducido seguido de la implementación de referencia y que el efecto de utilizar los restantes **no fue medido**.

**Repregunta:** ¿podría cambiar el resultado relativo de los preentrenados con más datos?

**Respuesta correcta:** sí; el trabajo no lo resuelve.

**Dificultad:** 3/5.
**Ubicación:** Cap. 3.2, pp. 14–15. 

---

## Nivel 3 — Estadística

### 4. ¿Por qué añadiste una prueba de permutación si ya tenías Wilcoxon?

**Respuesta mínima:** porque el estimando declarado es la diferencia media. Wilcoxon opera sobre rangos, mientras que el sign-flip utiliza directamente la media como estadístico. Se añadió post hoc y por ello Wilcoxon se conserva.

**Repregunta:** ¿qué haces cuando ambas pruebas discrepan?

**Respuesta correcta:** adopto una interpretación conservadora y no presento la diferencia como robusta.

**Dificultad:** 5/5.
**Ubicación:** Cap. 3.8, pp. 25–26; Tabla 19. 

---

### 5. ¿Por qué las dos realizaciones del ruido no elevan \(n\) de 200 a 400?

Porque están anidadas dentro de la misma condición inicial. Se promedian primero y el análisis continúa con 200 condiciones independientes.

**Repregunta:** ¿qué problema estadístico aparecería si trataras las 400 como independientes?

**Respuesta:** pseudorreplicación y subestimación de la incertidumbre.

**Dificultad:** 4/5.
**Ubicación:** Cap. 3.8, p. 25. 

---

## Nivel 4 — Inferencia

### 6. V0–Vpub tiene \(p=0,036\), pero el margen práctico es 0,05 y la diferencia es 0,041. ¿Qué puedes concluir?

**Respuesta correcta:** existe evidencia bajo el contraste primario de que V0 no rinde por debajo de Vpub, pero el IC90 % [0,011; 0,074] contiene efectos menores y mayores que 0,05; por tanto no puedo demostrar ni equivalencia ni una mejora superior al margen práctico.

**Repregunta:** ¿y qué añade que Wilcoxon dé \(p=0,092\)?

Que la detección de diferencia no es robusta a ambos procedimientos y debe interpretarse con cautela.

**Dificultad:** 5/5.
**Ubicación:** Cap. 4.4, Tabla 20, p. 42. 

---

### 7. En Godot obtienes \(p=0,53\). ¿Por qué no concluyes que los modelos tienen el mismo rendimiento?

Porque el experimento solo detecta diferencias aproximadamente superiores a 0,105 y no puede producir un IC90 % contenido en ±0,05 con \(n=50\). Es falta de resolución estadística, no evidencia de igualdad.

**Repregunta:** ¿cuál es la diferencia entre «no detectar una diferencia» y «demostrar equivalencia»?

**Dificultad:** 5/5.
**Ubicación:** Cap. 3.10, p. 30; Cap. 4.5, pp. 43–45. 

---

## Nivel 5 — Tribunal técnico

### 8. V2 tiene el menor MSE de acciones, pero V0 obtiene el mejor control. ¿Por qué?

**Respuesta esperada:** el MSE evalúa precisión offline sobre acciones demostradas, mientras que el rendimiento de control depende de una trayectoria cerrada. Un pequeño error modifica el siguiente estado observado, el error puede acumularse y además pueden existir múltiples acciones válidas para una observación. Por eso el MSE de acción no es un sustituto suficiente del rendimiento en bucle cerrado.

**Repregunta:** entonces, ¿por qué mantener el MSE?

Como diagnóstico de ajuste, no como variable primaria de éxito.

**Dificultad:** 4/5.
**Ubicación:** Cap. 4.6, Fig. 6, pp. 48–49.

---

### Las cuatro que considero más peligrosas en una defensa

**Una sola semilla de entrenamiento**, **V3 detenido en 155 épocas**, **la segunda realización añadida después del preregistro** y **la interpretación exacta de V0 frente a Vpub**.

Si esas cuatro se responden con precisión, la parte metodológica más vulnerable del TFM queda bien defendida.
