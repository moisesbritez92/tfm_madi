# Plan: realineamiento de la memoria del TFM tras la reevaluación

## Decisiones cerradas

- **No se entrena más.** Fase 2 (semillas 43/44) queda fuera de alcance → trabajo futuro.
- **Título nuevo elegido:**
  «Comparación de codificadores visuales en Diffusion Policy: protocolo con prueba
  disjunta, rendimiento y coste en Push-T»
  - Título corto (encabezados): «Codificadores visuales en Diffusion Policy» (ya está así)
  - Título inglés para el abstract: "Comparison of visual encoders in Diffusion Policy:
    a disjoint-test protocol, performance and cost on Push-T"
- **Consecuencia**: el estimando declarado es *cinco artefactos concretos*, no
  *estrategias de entrenamiento*. Todo objetivo, hipótesis y conclusión debe respetarlo.

## Contexto de partida

Informe: `evaluacion_tfm_reevaluacion.md`. Nota 73/100 bruta, techo 69/100.
El techo se activa por desajuste entre alcance prometido (estrategias) y evidencia
(n=1 ejecución por variante), no por fallo de ejecución. La palanca es realinear.

Datos ya disponibles sin GPU:
- `memoria/datos/prueba_final_episodios.csv` — 200 s_j × 5 variantes (bloque disjunto)
- `memoria/datos/prueba_final_{resumen,contrastes}.csv`
- `memoria/datos/seleccion_{optimismo,epoca_fija,k4}.csv`
- `memoria/datos/{latencia_inferencia,memoria_gpu}.csv`
- `memoria/datos/{demostraciones_episodios,condiciones_evaluacion}.csv`

## FASE 0 — Alcance (bloquea el resto)

- F0.1 `memoria/main.tex:15` `\tfmtitulo{...}` → título nuevo. Comprobar portada y
  `primera-hoja.tex`. Nota: `memoria/beamer/beamer.tex:3,26` también lleva el título
  (entregable aparte, actualizar al final).
- F0.2 `secciones/00-introduccion.tex:52-67` — objetivo general y específicos.
  General propuesto: «Cuantificar y comparar el rendimiento en bucle cerrado y el coste
  computacional de cinco bloques visuales integrados como codificador de observaciones de
  una misma Diffusion Policy en Push-T, cada uno entrenado una vez bajo un protocolo
  documentado, y delimitar el alcance inferencial de esa comparación.»
  Específicos (3 → 4):
  1. Implementar e integrar los cinco bloques con red generativa, datos e interfaz de
     condicionamiento constantes.
  2. Estimar, con bloque de prueba disjunto y preregistrado, las diferencias entre los
     cinco puntos de control con incertidumbre y control de multiplicidad.
  3. Caracterizar el coste en cuatro ejes (parámetros, min/época, latencia desglosada,
     pico de VRAM).
  4. Acotar la validez: cuantificar el sesgo de selección y establecer qué no puede
     concluirse con una ejecución de entrenamiento por variante.
- F0.3 `secciones/02-metodologia.tex:19-29` — «hipótesis» → «expectativas previas»,
  enunciadas sobre configuraciones completas. La tercera («rendimiento comparable») se
  elimina o recibe margen δ (ver B4). Decisión recomendada: eliminarla y sustituirla por
  «el orden interno de las preentrenadas no queda resuelto».
- F0.4 `secciones/00-introduccion.tex:69-84` (alcance) — «una sola semilla de
  entrenamiento» pasa de limitación a **premisa de diseño declarada**, justificada con las
  237,1 h de cómputo ya consumidas.
- F0.5 `secciones/resumen.tex:11` y `abstract.tex` — reescribir con el alcance nuevo.

## FASE A — Correcciones sin cómputo

| # | Hallazgo | Acción | Fichero |
|---|---|---|---|
| A1 | m1 | Redactar agradecimientos reales | `secciones/agradecimientos.tex` |
| A2 | m2 | **Los dos `??` son `\cref{anx:primero}` con `\include{secciones/anexos}` comentado en `main.tex:81`.** Descomentar `\cleardoublepage`/`\appendix`/`\include` y llenar el anexo con contenido real | `main.tex:78-81`, `secciones/anexos.tex` |
| A3 | M7 | `eq:perdida`: `A^0_t + ε^k` → `√ᾱ_k·A^0_t + √(1-ᾱ_k)·ε^k`. Definir α, γ, σ de `eq:muestreo` en función de β_k, α_k, ᾱ_k. Verificar coherencia con la ecuación del cap. 2 | `secciones/02-metodologia.tex:200-226`, `01-estado-del-arte.tex` |
| A4 | M3 | Verificar en WSL qué factory usa cada variante en `diffusion_policy/model/vision/pretrained_encoders.py`. Si V1 y V2 no cargan el mismo `state_dict`, **retirar** «El contraste entre V1 y V2 aísla el efecto de la estrategia de entrenamiento» | `02-metodologia.tex:263-275` |
| A5 | Concl. 7 | «la actualización de los pesos **degrada** la representación de partida» → hipótesis no comprobada | `04-conclusiones.tex:60-62` |
| A6 | m3 | Tabla única de transformaciones por variante: 96→recorte 76 (V0); resize 224 + norm ImageNet (V1-V2); 224 + norm propia (V3-V4). Interpolación incluida | `02-metodologia.tex`, `sec:variantes` |
| A7 | m4 | Integrar «Bibliografía» o citar formalmente el repositorio dentro de `Referencias` | `main.tex:64-75`, `bib/` |
| A8 | M9/m8 | Anexo A: config efectiva de Hydra por variante, versiones exactas, SHA-256 de los cinco checkpoints, commit `8bc22e7` del preregistro con marca temporal, identificadores exactos de pesos (`timm` model id, `torchvision` weights enum) | `secciones/anexos.tex` |
| A9 | — | Coherencia V3: metodología dice «índice 154», conclusiones «155 épocas» | ambos ficheros |

## FASE B — Reanálisis sin GPU

Script nuevo `memoria/scripts/analisis_prueba_final_v2.py` sobre
`datos/prueba_final_episodios.csv`.

- B1 (M6, prioritario) Test de **permutación sobre la diferencia media** pareada, junto a
  Wilcoxon. Resuelve el desajuste estimando↔contraste (pregunta 5 del tribunal). Holm
  sobre la familia nueva.
- B2 (M6) Declarar el método del IC de `tab:prueba-final`; recalcular por BCa para
  homogeneizar con `tab:contrastes`. No rotular «media ± EE» como intervalo.
- B3 (m5) IC binomiales (Wilson) para la tasa de éxito; declararla secundaria/descriptiva.
- B4 (M9) Si se conserva la expectativa de comparabilidad: TOST con δ declarado y
  etiquetado **post hoc**. Recomendado: eliminarla (ver F0.3).
- B5 (m7) `02-metodologia.tex` §3.2: sustituir «no introduce sesgo apreciable» por tamaños
  de efecto con IC. El argumento fuerte es el sorteo uniforme, no los p no significativos.
- B6 (m6) Tabla de controles de integridad del `zarr` (nulos, duplicados, rangos de
  acción, longitudes, hash del fichero). Script ligero con `zarr.open(...,'r')`.
- B7 (extra) Análisis de error por región: cruzar `prueba_final_episodios.csv` con
  posición/orientación inicial de las 200 condiciones. Figura nueva de discusión.
- B8 (M8) Validación de 4 episodios: no ampliable sin reentrenar. Degradar su papel a
  diagnóstico exploratorio y apoyar el sobreajuste en las curvas de rollout.

## FASE C — Descartada

- C1 (M5, réplicas de semilla de inferencia) requiere GPU (~2 h 33 min por réplica
  completa). **Descartado** por la decisión de no lanzar más cómputo. Pasa a trabajo
  futuro junto con la declaración explícita de que el estimando es condicional a una
  realización de ruido.
- C2 Evaluar checkpoints adicionales sobre 200000-200199: **prohibido**, convertiría el
  bloque de prueba en conjunto de selección.

## FASE D — Trabajo futuro (documentar, no ejecutar)

`sec:trabajo-futuro` debe recoger con diseño concreto: semillas 43/44 (M1), presupuesto
uniforme en actualizaciones con parada automática (M2), ablación factorial
resolución/recorte/agregación (M4), réplicas de ruido de inferencia (M5), validación
mayor por episodio (M8).

## FASE E — Cierre

- E1 Sección nueva en conclusiones: tabla de tres columnas — *demostrado para los cinco
  artefactos* / *no demostrado para las estrategias* / *no demostrado fuera del
  simulador*. Responde a las preguntas 7 y 10 del tribunal.
- E2 Podar conclusiones marcadas *no sustentada* (7, 9) o *parcial* (2, 4, 6, 15, 16) en
  el §19 del informe.
- E3 Resumen y abstract finales.
- E4 Recompilar: cero `??`, cero texto de plantilla, índices regenerados.
- E5 Actualizar `memoria/MEMORY.md` (decisión 19) y `memoria/beamer/beamer.tex`.

## Orden de ejecución

```
F0  →  A2 + A3 + A5  →  B1..B4  →  A4 (verificación en WSL)
    →  A1, A6..A9 + B5..B8  →  E1..E5
```

## Verificación

1. `cd memoria && latexmk -pdf -interaction=nonstopmode -file-line-error main.tex`
   → salida limpia, sin `??` en el PDF (`grep` sobre el texto extraído).
2. Comprobar que el anexo aparece en el índice y que `\cref{anx:primero}` resuelve.
3. Recorrer el §19 del informe conclusión a conclusión y verificar que cada una tiene
   respaldo en tabla o figura del capítulo 4.
4. Comprobar que ni título, ni objetivos, ni resumen, ni abstract, ni conclusiones
   contienen atribución causal a «estrategia de entrenamiento» o «preentrenamiento»
   aislado (grep de «estrategia de entrenamiento», «preentrenamiento mejora»).
5. Reejecutar `scripts/analisis_prueba_final_v2.py` y comprobar que las cifras de
   `tab:prueba-final` y `tab:contrastes` coinciden con los CSV regenerados.
