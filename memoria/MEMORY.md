# Contexto de trabajo: memoria del TFM

Estado y decisiones de la redaccion en LaTeX. El README.md documenta el uso; este
fichero documenta el *porque* y lo que hay pendiente.

Ultima actualizacion: 25 de agosto de 2026.

## Que es esto

Memoria del TFM del Master en Analisis de Datos en Ingenieria (Tecnun). Tema: comparacion
de cinco codificadores visuales como `obs_encoder` de una Diffusion Policy sobre Push-T
(V0–V4). El contexto experimental y operativo esta en `../CLAUDE.md`; los resultados y
checkpoints, en WSL. Esta carpeta contiene solo el documento.

- Autor: Moises Britez · Director: Diego Borro
- Titulo actual: «Influencia del codificador visual y su estrategia de entrenamiento en
  Diffusion Policy para manipulacion robotica: estudio en Push-T»
- Entrega prevista: septiembre de 2026

## Decisiones tomadas

1. **Formato modelado sobre `../../Vera Aguinaga, Jorge_TFM_MADI.pdf`**: portada (logo,
   tipo de trabajo, master en grande, titulo, autor, lugar y fecha, pie con la direccion
   de Tecnun), encabezados alternos, epigrafes numerados con punto, figuras y tablas con
   numeracion corrida (no por capitulo).
2. **Estructura solicitada**: Estado del arte · Metodologia · Resultados · Conclusiones ·
   Referencias · Bibliografia · Anexos. No se incluye el capitulo «Origen del proyecto»
   del modelo. El capitulo de Presupuesto se anadio despues, al aplicar las normas
   oficiales (decision 6).
3. **Citas en IEEE** con bibtex clasico e `IEEEtran.bst` v1.14 descargado de CTAN a
   `bst/`. No se usa biblatex.
4. **Dos listas de obras**, con `multibib`:
   - *Referencias* = obras citadas, numeradas `[n]` por orden de aparicion
     (`bib/referencias.bib`, `IEEEtran.bst`).
   - *Bibliografia* = obras consultadas no citadas, **sin numero** y ordenadas
     alfabeticamente (`bib/bibliografia.bib`, `IEEEtranS.bst`, `\sinetiquetasbib`).
   El modelo separaba «Referencias» (enlaces) de «Bibliografia» (obras academicas); con
   IEEE ese criterio no encaja, de ahi el cambio a citadas / no citadas.
5. **Clase `report`, `twoside`, `openright`**, 11 pt, interlineado sencillo, texto
   justificado, sin sangria y con espaciado entre parrafos.
6. **24 de agosto de 2026: el estilo se ajusto a `normas_redaccion.md`** (normas oficiales
   del centro). Cambios respecto a la primera version: margenes 35/30/30/25 mm, Helvetica
   sin escalar en lugar de Times, titulos de 16/14/13 pt con apartados subrayados,
   numeracion de apartados sin punto, pie «Pagina X de Y», primera hoja obligatoria y
   capitulo de Presupuesto. La portada dejo de usar el entorno `titlepage` porque reiniciaba
   el contador y descuadraba la paridad: los capitulos no caian en pagina impar.
   Se mantienen dos divergencias pedidas por el usuario: la seccion *Bibliografia*, que las
   normas no contemplan, y la numeracion corrida de figuras y tablas.
7. **24 de agosto de 2026: titulo nuevo y capitulo de objetivos.** El titulo paso a
   «Influencia del codificador visual y su estrategia de entrenamiento en Diffusion
   Policy para manipulacion robotica: estudio en Push-T». El anterior hablaba de
   codificadores «preentrenados», lo que dejaba fuera a V0 (ResNet-18 desde cero) y
   sugeria una ventaja del preentrenamiento que los datos no respaldan: V0 alcanza 0,8645
   frente a 0,668 de V1 y 0,6477 de V2. Se anadio `secciones/00-introduccion.tex` como
   capitulo 1, con contexto, problema, objetivos, alcance y estructura; el Estado del arte
   pasa a ser el capitulo 2. El fichero se llama `00-` para no renumerar los cinco
   existentes. El alcance se limito a un objetivo general y tres especificos
   (implementar, comparar rendimiento, analizar coste); robustez, perturbaciones visuales
   y contraste entre semillas quedan declarados fuera de alcance.
8. **24 de agosto de 2026: el coste de computo real, no la proyeccion.** La metodologia
   afirmaba que las cinco variantes sumaban «unas 75 horas de GPU», cifra proyectada desde
   V0. Los logs de `../logs_entrenamiento_2026-08-24/` dan **casi 200 h solo para V0, V1 y
   V2** (cifra afinada a 198,8 h en la decision 9): 17,4 + 96,5 + 84,9 h. El apartado 3.9
   (`sec:coste`) separa el tiempo del bucle de optimizacion (121,6 h) del tiempo total
   transcurrido, que incluye validacion y rollouts. V0 y V2 solo conservan el cronometro
   de parte de sus epocas (89/500 y 196/266), asi que sus acumulados se extrapolan de la
   media por epoca y se marcan con
   una daga. De V2 se descuenta la interrupcion de dos dias entre sus dos sesiones; la
   primera sesion (epocas 0-69) se estima con el ritmo de la segunda.
   Ademas, las epocas dejaron de presentarse como fijas: **500 es el presupuesto maximo y
   hay parada anticipada**, de modo que las variantes completan entre 200 y 500 epocas
   (V0 y V1, 500; V2, 266). La parada se decide por inspeccion, no de forma automatica, y
   por eso V0 y V1 agotaron el presupuesto: el criterio se adopto despues de verlas. La
   metodologia lo dice de forma explicita, porque afecta a la comparabilidad del coste.
9. **25 de agosto de 2026: coherencia entre metodologia, resultados y conclusiones.**
   El capitulo de resultados (redactado por otro agente) cuenta **266 epocas completas**
   en V2, no 267: el indice 266 quedo interrumpido tras cinco lotes y sin validacion. Se
   alineo la Tabla 7 con ese criterio y se recalcularon los estimados de V2 sobre 266
   epocas: bucle 65,3 h y total 84,9 h; los totales pasaron a 1.266 epocas, 121,6 h y
   198,8 h.
   Segundo ajuste, mas de fondo: **el tiempo total no sirve para comparar variantes**. V2
   acumula menos horas que V1 (84,9 frente a 96,5) pese a costar casi el triple por epoca,
   solo porque se detuvo antes. La comparacion valida es por epoca (1,5 / 5,3 / 14,7 min,
   factores 3,5 y 9,8). Metodologia, resultados y conclusiones lo dicen ahora de forma
   expresa; el resumen y el abstract usan directamente los minutos por epoca.
10. **25 de agosto de 2026: presupuesto.** Cinco apartados: fungible y suministros,
    amortizacion de equipos, licencias, mano de obra y resumen. La amortizacion es lineal,
   sin valor residual, imputando 7/48 meses del portatil (262,50 \euro). La mano de obra
   domina el presupuesto: 85 % del total frente al 1,6 % de los equipos. El capitulo cierra
    con la lectura economica del \cref{sec:coste}: ampliar a tres semillas triplicaria el
    computo sin apenas alterar el coste, porque el limite es el plazo de calendario y no el
    dinero. **Todos los importes son supuestos declarados en el texto, no datos medidos**;
    estan listados en Pendiente para que el director los confirme.
11. **25 de agosto de 2026: cobertura del preentrenamiento visual para control.** Tras la
    evaluacion externa M7, el apartado de codificadores se amplio con una subseccion sobre
    MVP, Voltron y VC-1/CortexBench. Los tres articulos se incorporaron al cuaderno
    «Diffusion Policy - Investigacion Arxiv». La sintesis ya no presenta el
    preentrenamiento para control como una linea ausente: la evidencia previa muestra que
    su transferencia depende del dominio, del objetivo representacional y de la estrategia
    de adaptacion. El hueco queda restringido a la comparacion controlada de la matriz V0-V4
    dentro de una Diffusion Policy fija y bajo un protocolo comun de Push-T.

12. **25 de agosto de 2026: respuesta al informe de evaluacion.** Tras la evaluacion
   externa (`../../EVALUACION_TFM.md`) se cerraron cuatro observaciones mayores sin
   computo nuevo, todas a partir de los logs ya guardados. **(a) Dispersion (M1):** los
   ficheros `raw/v*_logs_json.txt.gz` guardan los 50 `s_j` individuales
   (`test/sim_max_reward_1000XX`), de modo que la metrica admite desviacion tipica, error
   estandar, intervalo de confianza y prueba de Wilcoxon pareada sin volver a simular. Los
   calcula `scripts/analisis_dispersion.py` y su salida vive en `datos/`. Resultado clave:
   V0-V1 y V0-V2 excluyen el cero, **V1-V2 no (p = 0,82)**. **(b) Terminologia (M2):** lo
   que se llamaba *puntuacion media de prueba* pasa a *puntuacion media de evaluacion*,
   porque esas 50 condiciones deciden el punto de control y actuan como conjunto de
   seleccion; *prueba* queda reservado para una medida independiente que no se ha hecho.
   Se reporta ademas la media de las tres ultimas evaluaciones (0,862 / 0,547 / 0,585)
   como estimador menos sesgado. **(c) Factores confundidos (M3):** son **cuatro**, no
   tres; se anade la agregacion espacial (*spatial softmax* en V0 frente a descriptor
   global en el resto), y la formulacion de bloque abre ahora el apartado 5.1 en lugar de
   aparecer sesenta lineas mas abajo. **(d) Parada anticipada (M6):** el apartado 3.5 la
   describe en pasado, como decision adoptada durante la ejecucion. El
   `scripts/coste_parada_uniforme.py` cuantifica el contrafactual: bajo el criterio
   uniforme **V0 se habria detenido en la epoca 300 y V1 en la 250**, con 91 h de bucle y
   139 h totales frente a 121,6 y 198,8. El unico punto de control que cambiaria es el de
   V0 (epoca 200, 0,8643 frente a 0,8645): ninguna conclusion depende de eso.
   Las cifras de puntuacion pasaron a **tres decimales** en todo el documento.

## Origen de la bibliografia

`bib/referencias.bib` se construyo a partir del cuaderno de NotebookLM **«Diffusion Policy
Extendido»** (id `a3b2bf3c-6b83-4065-8107-ce975e391884`, 11 fuentes), extrayendo los
metadatos de la primera pagina de cada PDF. Las dos copias del articulo de Song y Ermon
(2019) son el mismo trabajo: 11 fuentes → 10 entradas. Se anadio ademas
`chi2023diffusion`, la version de congreso (RSS 2023) del articulo principal.

Dos campos **no** proceden de los PDF y llevan comentario `% verificar`: las paginas de
DDPM en las actas de NeurIPS 2020 y el volumen/paginas de PMLR de *Implicit Behavioral
Cloning*.

El estado del arte se redacto el 24 de agosto de 2026 a partir de los cuadernos
**«Diffusion Policy Extendido»** y **«Diffusion Policy - Investigacion Arxiv»**. Los
metadatos de las nuevas obras se contrastaron con arXiv y, para ImageNet, con el DOI del
articulo. La revision detecto dos antecedentes que obligan a formular el hueco con
precision: la ablacion visual del articulo original y el preprint DINOv3-Diffusion Policy.

El 25 de agosto se anadieron al segundo cuaderno los PDF de MVP (arXiv:2203.06173),
VC-1/CortexBench (arXiv:2303.18240) y Voltron (arXiv:2302.12766). Sus metadatos se
contrastaron con arXiv, las actas de NeurIPS 2023 y las actas de RSS 2023.

## Trampas ya resueltas (no repetirlas)

- `\usepackage[latin1]{inputenc}` en `main.tex` provoca *Option clash* con el `utf8` del
  estilo y romperia las tildes: los ficheros estan en UTF-8.
- Dentro de `tecnun-tfm.sty` la arroba ya es letra: un `\makeatother` suelto desactiva
  todas las macros `\@...` posteriores del propio fichero.
- `multibib` redefine `\bibname`; hay que reponerlo justo antes de `\bibliography` o la
  lista principal se titula «Bibliografia».
- BibTeX no admite comentarios `%` dentro de una entrada: van entre entradas.
- `amssymb` despues de `newtxmath` da `\Bbbk already defined`; `amsmath` se carga antes de
  `newtx` y `amssymb` no se carga.
- Las hojas de relleno de `\cleardoublepage` llevaban encabezado y numero; el estilo
  redefine la macro para dejarlas vacias.
- Una celda de tabla que empieza por `[` justo tras `\midrule` o `\\` se traga como
  argumento opcional y **cuelga la compilacion sin error**: escribir `{[texto]}`.
- Cambiar la opcion `es-lcroman` de babel invalida el `.toc` anterior (`\es@scroman`
  indefinido): hay que hacer `make clean` antes de recompilar.
- El logo es `img/logo.png` (no `logo-tecnun.pdf`); la ruta se fija con `\logotecnun` en
  `main.tex`.
- babel-spanish convierte **todo punto en modo matematico en coma**, incluido el separador
  de millares de `siunitx`: `\num{3000}` salia «3,000», que en espanol se lee como tres.
  Se corrige con la opcion `es-nodecimaldot` en la carga de babel (`tecnun-tfm.sty:13`).
  Cambiar opciones de babel invalida el `.toc`: `make clean` antes de recompilar.
- El agrupamiento de `siunitx` parte **tambien los decimales** si no se le dice lo
  contrario: `\num{0.8645}` salia «0,864.5». El `\sisetup` del estilo lleva ahora
  `group-digits=integer`. Ojo: solo se nota con cuatro o mas decimales, asi que puede
  colarse sin que nadie lo vea.
- El abstract en ingles reabre la marca decimal a punto con un `\sisetup` dentro de
  `\begin{otherlanguage}{english}`; el entorno abre grupo, asi que no se escapa.
- babel-spanish traduce `\max`, `\min` y `\lim` y compone la tilde como acento matematico;
  con `newtxsf` el acento cae sobre la letra y los glifos **se solapan** («mBx»). El estilo
  los redefine en modo texto. Debe hacerse dentro de `\addto\extrasspanish`, no solo en un
  `\AtBeginDocument`: babel reaplica sus definiciones en cada cambio de idioma.
- `\crefname{section}` es «Apartado», masculino: se escribe *el/del* `\cref{sec:...}`, no
  *la*. Con figuras, tablas y ecuaciones es al reves.

## Estilo de redaccion

Se sigue la skill personal `redactor-tesis` (`~/.claude/skills/redactor-tesis/`):
lenguaje impersonal, frases de 15–25 palabras, conectores explicitos entre parrafos, sin
metaforas ni relleno, siglas desarrolladas en su primera aparicion, tablas y figuras
citadas en el texto antes de aparecer. Regla dura: **no inventar referencias, DOI, anos ni
paginas**; lo que falte se marca `[PENDIENTE: referencia]`.

## Pendiente

- [x] Redactar el estado del arte y delimitar el hueco experimental frente a la ablacion
      visual original y DINOv3-Diffusion Policy.
- [x] Redactar la metodologia, incluidos el entorno de ejecucion (Tabla 5), los
      componentes de la pila de software (Tabla 6) y el coste temporal (Tabla 7).
- [x] Redactar conclusiones, resumen y abstract (25 de agosto de 2026), con las cifras de
      V0, V1 y V2.
- [x] Redactar resultados (otro agente) y presupuesto (25 de agosto de 2026). Cifras
      comunes a los cuatro capitulos: 0,8645 / 0,6676 / 0,6477; epocas 500 / 500 / **266**;
      1,5 / 5,3 / 14,7 min por epoca; 17,4 / 96,5 / 84,9 h de tiempo total.
- [ ] Confirmar con el director los supuestos economicos del presupuesto, que son
      estimaciones y no datos: 450 h del autor repartidas por fases, 40 h de direccion,
      30 y 60 \euro/h de coste horario, 1.800 \euro de precio del portatil, siete meses de
      imputacion (marzo-septiembre de 2026) y 15 % de costes indirectos. Si cambia alguno,
      recalcular la Tabla 13; el total actual es 18.685,51 \euro.
- [ ] Revisar conclusiones, resumen y abstract cuando entren V3 y V4: hay un
      `[PENDIENTE: resultados de V3 y V4]` al final del apartado 5.1, y tanto el resumen
      como el abstract dicen que solo se completaron las tres variantes convolucionales.
- [ ] Completar la Tabla 7 (`sec:coste`) con V3 y V4 cuando terminen de entrenarse; ahora
      figuran como «en ejecucion».
- [x] Versiones exactas de Hydra y Zarr en el entorno de WSL, consultadas en la maquina de
      entrenamiento el 25 de agosto de 2026: **hydra-core 1.2.0 y zarr 2.12.0**, ya en la
      Tabla 6 y en `../CLAUDE.md`. Faltaban porque ni `CLAUDE.md` ni
      `entorno_wsl.txt` las recogian: el volcado del entorno anoto plataforma, CUDA, Python
      y PyTorch, pero no un `pip freeze`. El `requirements.txt` de la raiz **no** sirve,
      describe el entorno de inferencia en Windows (torch 2.6, hydra 1.3.2, zarr 2.18.7),
      no el de entrenamiento (torch 1.12.1).
- [ ] Verificar los dos campos marcados `% verificar` en `bib/referencias.bib`.
- [ ] Anadir a `bib/referencias.bib` las obras de los codificadores visuales (ResNet,
      ImageNet, DINOv2, CLIP, ViT) y resolver el `[PENDIENTE: referencia]` de
      `secciones/00-introduccion.tex`, apartado 1.2.
- [x] Autor y ano de la monografia propia sobre Diffusion Policy: Moises Britez, 2026,
      registrados en `bib/bibliografia.bib`.
- [ ] Resultados de V3 (DINOv2) y V4 (CLIP): pendientes de entrenar, sin ellos el capitulo
      de resultados queda incompleto.
- [ ] Decidir si el encabezado se mantiene con filete y cursiva o se simplifica.
- [ ] **Del informe de evaluacion, lo que necesita GPU o acceso a WSL** (ver
      `../../respuestas_evaluador.md`): evaluar los tres puntos de control sobre semillas
      disjuntas 200000-200049, para convertir la puntuacion de seleccion en prueba
      independiente (P2.1, tres inferencias); medir pico de VRAM y latencia de inferencia,
      que el apartado 3.7 promete y el capitulo 4 no reporta (P2.3, M4); caracterizar el
      `zarr` con distribucion de longitudes y puntuaciones de las demostraciones y explicitar
      el reparto train/val/descartados (P2.8, M5); y los dos runs de ablacion que separan
      los cuatro factores confundidos (~85 h a 224 px, ~17 h a 96 px sin recorte).
- [x] M7 del informe (literatura de preentrenamiento visual para control) **ya estaba
      resuelto**: el apartado 2.5.4 cubre MVP, Voltron y VC-1/CortexBench. El informe
      describe un apartado 2.4 que no corresponde a la version actual.
