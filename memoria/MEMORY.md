# Contexto de trabajo: memoria del TFM

Estado y decisiones de la redaccion en LaTeX. El README.md documenta el uso; este
fichero documenta el *porque* y lo que hay pendiente.

Ultima actualizacion: 27 de agosto de 2026.

## Que es esto

Memoria del TFM del Master en Analisis de Datos en Ingenieria (Tecnun). Tema: comparacion
de cinco codificadores visuales como `obs_encoder` de una Diffusion Policy sobre Push-T
(V0–V4). El contexto experimental y operativo esta en `../CLAUDE.md`; los resultados y
checkpoints, en WSL. Esta carpeta contiene solo el documento.

- Autor: Moises Britez · Director: Diego Borro
- Titulo actual: «Comparacion de codificadores visuales en Diffusion Policy: protocolo
  con prueba disjunta, rendimiento y coste en Push-T» (decision 19)
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
4. ~~**Dos listas de obras**, con `multibib`~~ — **revertida el 27/08/2026, decision 19.**
   Ahora hay una sola lista IEEE y no hay anexos. Se conserva el registro del porque:
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
   V0. Los logs de `../logs_entrenamiento/` dan **casi 200 h solo para V0, V1 y
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

13. **26 de agosto de 2026: entran V3 y V4.** Se copiaron sus checkpoints desde WSL a
    `../diffuser/models/V{3,4}/` (SHA-256 verificado) y se exportaron sus logs a
    `../logs_entrenamiento/`, carpeta que perdio la fecha del nombre. Cifras: **V3
    (DINOv2 ViT-S/14 congelada) 0,622 en la epoca 100** y **V4 (CLIP ViT-B/16 congelada)
    0,535 en la epoca 100**. Dos matices que la redaccion no puede pasar por alto:
    **(a) el presupuesto de epocas no fue uniforme** — 500 en V0 y V1, 300 en V2 y V3,
    200 en V4 —, asi que V3 y V4 solo tienen 4 evaluaciones frente a las 10 de V0 y V1,
    y V3 quedo detenido en la epoca 154 de sus 300. **(b) el orden entre las cuatro
    variantes no ganadoras no es estadisticamente distinguible tras corregir la familia de
    contrastes**. V1-V4 alcanza p = 0,0084 sin corregir, pero queda en p de Holm = 0,0501;
    V0 supera a las cuatro con valores de Holm iguales o inferiores a 0,0030.
    La conclusion se refuerza: lo que separa a V0 no es la familia de codificador sino su
    entrenamiento conjunto con la politica.
    La exportacion dejo de ser manual. `../diffuser/scripts/exportar_logs_wsl.sh` vuelca
    logs, config efectiva y un `raw/*_meta.json` con la mtime del log (de ahi sale el
    campo `end`, que no existe dentro del JSON); `scripts/resumen_entrenamiento.py`
    reconstruye el CSV de tiempos por epoca y la entrada de `resumen.json`. El parser se
    valido contra V1: reproduce sus 500 epocas y sus 157.702 s exactos.

14. **26 de agosto de 2026: memoria cerrada con V0-V4.** Se regeneraron las figuras de
    puntuacion y perdidas, se ampliaron los contrastes a los diez pares y se aplico la
    correccion secuencial de Holm. Resultados seleccionados: 0,864 / 0,668 / 0,648 / 0,622 /
    0,535 para V0-V4. La tabla temporal incorpora las cinco ejecuciones y corrige V0 con
    sus 500 epocas cronometradas: 1,6 min por epoca y 13,0 h de bucle, sin estimacion. Los
    totales pasan a 1.621 epocas, 141,3 h de bucle y 237,1 h transcurridas. El presupuesto
    se recalculo con 60 kWh y asciende a 18.686,55 euros. Resumen, abstract, resultados y
    conclusiones ya no presentan V3 y V4 como ejecuciones pendientes.

15. **26 de agosto de 2026: latencia de inferencia y pico de VRAM medidos (M4).** El
    apartado 3.7 prometia cuatro indicadores de coste y el capitulo 4 solo reportaba dos.
    Ahora estan los cuatro, con dos tablas nuevas en el apartado 4.4: `tab:latencia` y
    `tab:memoria-gpu`. Se hicieron **dos tablas y no una** porque los indicadores se miden
    en entornos distintos y una tabla unica obligaria a mezclar milisegundos con gibibytes
    y dos pilas de software en la misma nota.
    - **Latencia** (`scripts/latencia_inferencia.py` -> `datos/latencia_inferencia.csv`),
      medida en Windows con `.venv_diffuser_infer`. Round-robin de 10 rondas x 5
      repeticiones = 50 cronometradas por celda, 20 de calentamiento, `cudnn.benchmark`
      desactivado, observaciones reales del simulador (no tensores sinteticos). Resultado
      central: **la llamada completa no distingue las variantes** (1.723,8-1.749,1 ms con
      lote 1, un 1,5 %, por debajo del propio IQR), mientras el **codificador aislado va de
      2,7 a 16,1 ms** (factor 5,9) y con lote 8, de 3,1 a 131,8 ms (factor 43). Los 100
      pasos de difusion diluyen el codificador. Por eso el desglose no es opcional.
      Consecuencia: 215,5-218,6 ms por accion, unas 4,6 acciones/s, frente a los 10 Hz del
      simulador. **Ninguna variante alcanza el control en tiempo real en este equipo.**
    - **Pico de VRAM** (`../diffuser/scripts/memoria_gpu.py` +
      `medir_memoria_gpu.sh` -> `datos/memoria_gpu.csv`), medido **en WSL con torch
      1.12.1**, un proceso por variante y modo, 12 pasos de optimizacion completos con
      optimizador, EMA y acumulacion reales. Hallazgo fuerte: **V2 reserva 10,293 GiB sobre
      una tarjeta de 8 (128,7 %)**; no aborta porque el controlador respalda el exceso con
      memoria del sistema, y ahi esta la explicacion de sus 14,7 min por epoca frente a los
      5,3 de V1, con la que comparte todo salvo la retropropagacion por el codificador. En
      inferencia ninguna variante pasa del 23 %: el problema de memoria es de
      entrenamiento, no de despliegue.
    - **Comprobacion de coherencia, como pedia el prompt.** El 3.3 (descartar el ajuste
      fino de los ViT por VRAM) **queda respaldado** y ahora cita la cifra. El 3.4 (lote
      reducido en V3/V4) **solo se sostiene para V4** (90,3 % de la tarjeta); V3 se queda en
      78,2 % y el texto lo reconoce ahora como cautela, no como necesidad.
    - **Dos errores de la memoria detectados al verificar los supuestos**, ambos corregidos:
      la Tabla 8 decia «lote 64 (32 en V4)» cuando **V3 tambien usa 32**, y tanto 3.4 como
      4.4 afirmaban que eso duplicaba las actualizaciones por epoca. Es falso: **V3 y V4
      acumulan el gradiente durante dos lotes**, asi que el lote efectivo es 64 en las cinco
      y las actualizaciones por epoca son 168 en todas (verificado contra los `global_step`
      de los logs: 336 lotes/epoca en V3-V4 frente a 168 en V0-V2). Lo que se duplica son
      las pasadas por el codificador.
    - Ojo tambien con **la resolucion**: solo V0 opera a 96 px (con recorte a 76). V1 y V2
      redimensionan a 224 igual que los transformadores.
    - **Trampa de medida**: el asignador de PyTorch no devuelve el pool ya crecido, de modo
      que medir lote 1 y lote 8 en el mismo proceso reporta el primero dos veces. De ahi un
      proceso por combinacion. Y en WSL los factories de `pretrained_encoders.py` fijan
      `pretrained=True` en su firma (la copia de Windows si acepta el parametro), asi que
      instanciarlos **bloquea el proceso descargando pesos**; el script envuelve
      `timm.create_model` para forzar `pretrained=False`, sin tocar el arbol de WSL.

16. **26 de agosto de 2026: MiKTeX desactualizado rompia `compilar.sh`.** El fallo era
    previo a los cambios de M4: `lastpage` 2025 exigia un `hyperref` mas nuevo, y el
    `hyperref` 2025 exige un nucleo de LaTeX posterior a 2024-11. Se actualizaron
    `hyperref`, `ltxbase`, `l3backend`, `l3packages`, `latex-firstaid`, `latex-tools`,
    `amsmath`, `graphics`, `unicode-data`, `etoolbox`, `bookmark`, `rerunfilecheck` y
    `oberdiek`, y se regenero el formato con `initexmf --dump=pdflatex`. Si vuelve a
    fallar con `\IfDocumentMetadataT` indefinido, el nucleo se ha quedado atras otra vez.
    Nota: `siunitx` **no admite numeros de version** en `\num{}`; `\num{2.6.0}` aborta la
    compilacion. Las versiones van en `\texttt{}`.

17. **26 de agosto de 2026: caracterizacion del conjunto de datos (M5).** El apartado 3.2
    describia el `zarr` pero no lo caracterizaba. Ahora lleva la Tabla 3 (`tab:particion`)
    y la Figura 1 (`fig:caracterizacion-dataset`), la primera figura del capitulo 3.
    - **El reparto es 90 / 4 / 112**, es decir el **54,4 % de los episodios no se usa**.
      `task/pusht_image.yaml` fija `dataset.seed: 42` como literal, no como `${seed}`: la
      particion **no depende de la semilla de entrenamiento** y sera la misma en la fase 2.
      Los cuatro episodios de validacion son los indices 18, 90, 134 y 157.
    - **El regimen de pocas demostraciones, en cifras**: 11.356 transiciones, 10.726
      ventanas de horizonte 16 y 18,9 min de teleoperacion de los 42,8 del fichero. Las
      10.726 ventanas a lote efectivo 64 dan las 168 actualizaciones por epoca que ya
      aparecian en los `global_step`; es una comprobacion cruzada, no una coincidencia.
    - **El descarte es representativo**, y esa es la respuesta al tribunal: sorteo
      aleatorio uniforme, sin criterio de calidad. Entrenamiento frente a descarte da
      p = 0,752 (Mann-Whitney) y p = 0,999 (KS) en longitud, y p = 0,302 y p = 0,418 en
      puntuacion.
    - **El demostrador humano promedia 0,892 y ninguna demostracion alcanza el umbral de
      exito de 0,95** (cobertura maxima 0,902). El 0,864 de V0 es el **96,9 %** de esa
      referencia. El dato entra en 3.2 y se cita una vez en 4.2; no se toco ni el resumen
      ni las conclusiones por eso.
    - **Las demostraciones son las semillas 0 a 205 del entorno**:
      `PushTEnv(legacy=True).seed(i).reset()` reproduce el primer instante de la
      demostracion `i` con error 0,000 px en las 206. De ahi salen dos consecuencias. La
      primera, que demostraciones y evaluacion muestrean el mismo generador con semillas
      disjuntas (los cinco contrastes KS quedan entre 0,232 y 0,696). La segunda, que
      **las seis condiciones `train/` del evaluador son las de las demostraciones 0 a 5, y
      el sorteo dejo las seis fuera del entrenamiento** (el primer episodio de
      entrenamiento es el 7). El apartado 3.7 afirmaba que comparar esa puntuacion con la
      de evaluacion detecta sobreajuste: es falso, y el texto lo advierte ahora.
    - **Cuidado con `legacy` al leer el `zarr`.** Para *reconstruir* un estado guardado
      vale `legacy=False` (error de 1,16 px por resolucion de contactos, y 0,46 px contra
      los `keypoint` almacenados); con `legacy=True` el error llega a 90 px porque el giro
      se aplica despues de colocar la pieza. Para *generar* una condicion inicial desde una
      semilla es al reves: manda `legacy=True`, que es lo que usan la recogida de datos y
      el evaluador. Los dos scripts lo comprueban con `assert`.
    - Trampa: crear un `PymunkKeypointManager` sobre el mismo entorno que se usa para medir
      cobertura altera su espacio de `pymunk` y **la cobertura sale 0** sin avisar.
    - Se corrigio de paso un error factual repetido: resumen, abstract y conclusiones
      decian **206 demostraciones** donde el ajuste usa **90**.
    - Reparto de scripts: `../diffuser/scripts/caracterizar_dataset.py` extrae en WSL a
      `datos/demostraciones_episodios.csv` y `datos/condiciones_evaluacion.csv`;
      `scripts/figuras_dataset.py` dibuja la figura en Windows y reimprime los contrastes.
      La figura se regenera sin WSL.

18. **27 de agosto de 2026: prueba final sobre semillas disjuntas (C1).** El informe
    `../evaluacion_tfm_I.md` aplicaba un techo de **59/100** por una sola razon material:
    las 50 condiciones `100000-100049` elegian el punto de control y ademas producian la
    media, el IC y el *p* que la memoria reportaba como resultado. Ya no.
    - **El preregistro es la pieza clave, no el numero.** `preregistro_prueba_final.md`
      fija bloque, *n*, variables de resultado, metodo de IC, tratamiento de ceros y
      familia de Holm, con los SHA-256 de los cinco checkpoints, y **se commiteo antes de
      ejecutar nada** (commit `8bc22e7`, 08:31). La marca de tiempo del commit es la
      evidencia que el evaluador exige. Si se rehace algo de esto, el preregistro se
      escribe primero, siempre.
    - **Resultado.** Sobre `200000-200199`, n = 200: **V0 0,872 · V1 0,649 · V2 0,586 ·
      V3 0,578 · V4 0,490**. La ordenacion no cambia y la ventaja de V0 **crece**: los
      margenes pasan de 0,197-0,329 a 0,223-0,382, con Holm entre 1,8e-12 y 2,6e-20.
    - **El sesgo optimista existia y era desigual, pero no en la direccion temida.** V0 fue
      la unica que no perdio (-0,007, gana un poco); las cuatro preentrenadas perdieron
      entre 0,019 y 0,062. Es decir, el sesgo iba **en contra** de la conclusion, no a
      favor. Este es el argumento fuerte para la defensa.
    - **Cuidado con la estimacion split-half.** `scripts/analisis_seleccion.py` la calcula
      (0,040 / 0,045 / 0,007 / 0,030 / 0,021) y **no predice el optimismo realizado
      variante a variante**: a V2 le atribuia 0,007 y perdio 0,062. Acota el orden de
      magnitud del fenomeno, nada mas. La memoria lo dice de forma expresa; no venderla
      como mas de lo que es.
    - **Lecturas libres de maximo, sin GPU** (mismo script, `datos/seleccion_*.csv`): a
      epoca comun 150, V0 0,828 frente a 0,668 / 0,648 / 0,579 / 0,435; con las
      oportunidades igualadas a K = 4, V0 0,837. **Ojo: la comparacion a epoca fija solo
      se puede hacer con los logs.** No hay checkpoint de la epoca 150 de V0 (guarda
      200/350/400) y la interseccion de epocas guardadas entre las cinco variantes es
      vacia.
    - **Numeros aleatorios comunes (M4), verificados.** `evaluar_bloque_test.py` siembra
      `torch.manual_seed(base*1000003 + tanda*1000 + paso)` antes de cada
      `predict_action`, envolviendo la instancia sin tocar el arbol de WSL. Dos pasadas de
      V4 sobre las mismas 8 semillas dan puntuaciones **identicas bit a bit**. Funciona
      porque en `eval()` el `CropRandomizer` hace recorte central y el codificador no
      consume RNG; el unico consumo es el `torch.randn` de la politica, con forma
      (8, 16, 2) en las cinco variantes.
    - **La cordura antes de gastar las horas.** Reevaluar V0 sobre el propio conjunto de
      seleccion dio 0,8566 frente a los 0,8645 registrados, desvio 0,0079. No sale exacto
      (el estado del RNG durante el entrenamiento no era este) y no tiene por que.
    - **Coste real:** 2 h 33 min para las cinco variantes, ~30 min cada una, 25 tandas de
      8 condiciones a ~80 s. La latencia medida en Windows con torch 2.6 subestima: en WSL
      con torch 1.12 el rollout va mas lento.
    - **Lo que sigue abierto y hay que declararlo:** solo se guardaron los tres mejores
      checkpoints *segun la metrica contaminada*, asi que el pool de candidatos viene
      prefiltrado. El bloque disjunto elimina el sesgo del maximo, no el del pool. Y M1
      sigue intacto: una sola semilla de entrenamiento.
    - En la memoria: apartado 3.8 nuevo (`sec:seleccion-prueba`) con estimando, unidades y
      estadistica completa (cierra M6 y M9), Figura 2 con el flujo, `tab:seleccion-sesgo`,
      `tab:prueba-final` y `tab:contrastes` **trasladada al bloque de prueba**. La Tabla 9
      se titula ahora «conjunto de seleccion, estimaciones optimistas». Resumen, abstract,
      contraste de hipotesis y conclusiones rehechos con las cifras nuevas.
    - Trampa nueva: **los heredoc de bash de esta sesion se comen la contrabarra antes de
      apostrofo**, de modo que `\'o` se convierte en `'o` y rompe el LaTeX en silencio. Los
      ficheros `.tex` se editan con la herramienta de edicion, no con `sed`/heredoc.

19. **27 de agosto de 2026: realineamiento tras la reevaluacion.** El informe
    `../evaluacion_tfm_reevaluacion.md` da 73/100 bruto con techo de **69/100**, y el
    techo **no lo activa un fallo de ejecucion** sino el desajuste entre el alcance
    prometido (estrategias de entrenamiento) y la evidencia disponible (una ejecucion por
    variante). La palanca es realinear el alcance, no computar mas. Se ejecuto el plan de
    `../plan-realineamientoMemoriaTfm.prompt.md`.
    - **Titulo nuevo:** «Comparacion de codificadores visuales en Diffusion Policy:
      protocolo con prueba disjunta, rendimiento y coste en Push-T». El anterior prometia
      inferencia sobre *estrategias*.
    - **El estimando declarado son cinco artefactos concretos**, no estrategias. Objetivo
      general reescrito, cuatro objetivos especificos (el cuarto es *acotar la validez*),
      y las «hipotesis» pasan a **expectativas previas E1-E3** enunciadas sobre
      configuraciones completas. La tercera ya no afirma comparabilidad: deja abierto el
      orden interno, porque no hay margen de equivalencia ni TOST.
    - **Una sola semilla deja de ser limitacion y pasa a premisa de diseno declarada**,
      justificada con las 237,1 h ya consumidas frente a las ~700 h que exigirian tres
      semillas. Aparece en 1.4, en 3.1 y al abrir el capitulo 5.
    - **Dos errores factuales encontrados al verificar A4, ambos corregidos.** Son el
      hallazgo mas importante de la sesion:
      1. **Ninguna variante usa `spatial softmax`.** La config `pusht_image` pasa por
         `MultiImageObsEncoder` + `model_getter.get_resnet`, que hace `fc = Identity`:
         es promediado global. El punto de control de V0 no tiene ningun modulo softmax.
         Lo que si distingue a V0 es `use_group_norm: True` (sin `num_batches_tracked` en
         el ckpt). El *spatial softmax* del articulo pertenece al codificador de robomimic,
         no a esta configuracion. Los cuatro factores confundidos son ahora: pesos
         iniciales, resolucion con recorte, capa de normalizacion (grupos/lotes) y, frente
         a V1, congelacion.
      2. **V1 y V2 no parten del mismo archivo de pesos.** V1 usa
         `timm.create_model("resnet18", pretrained=True)`, que resuelve a
         **`resnet18.a1_in1k`** (receta A1 de *ResNet strikes back*); V2 usa
         `torchvision` con `weights: IMAGENET1K_V1`. Verificado tensor a tensor sobre el
         ckpt congelado de V1: **120/120 identicos con timm, 0/120 con torchvision**
         (max|dif| = 3,7e5). Se retiro la afirmacion «el contraste V1-V2 aisla la
         estrategia» de metodologia, resultados y conclusiones.
    - **Prueba de permutacion sobre la diferencia media (M6).**
      `scripts/analisis_prueba_final_v2.py`, declarado **post hoc** en 3.8 y en 4.3.
      Wilcoxon contrasta rangos, no la media que la memoria reporta. Resultado que cambia
      la lectura: V0 sigue ganando con ambos, pero **V2-V4 y V3-V4 cruzan 0,05 con
      permutacion (Holm 0,040) y no con Wilcoxon (0,052 y 0,051)**. La memoria adopta la
      lectura conservadora: **V4 queda por debajo, el orden entre V1, V2 y V3 no se
      resuelve**. No se cambio el contraste principal por el mas favorable.
    - **IC declarados (B2) y Wilson (B3).** La Tabla 11 llevaba «media ± EE» rotulado como
      intervalo; ahora es BCa de 10.000 remuestreos, el mismo metodo que las diferencias.
      La tasa de exito lleva IC de Wilson y se declara secundaria y descriptiva, fuera de
      toda familia corregida.
    - **Ecuaciones DDPM corregidas (M7).** La ecuacion 3 decia `A^0 + eps`. Ahora:
      `A^k = sqrt(a_barra_k) A^0 + sqrt(1-a_barra_k) eps`, y el muestreo define alpha,
      gamma y sigma en funcion de beta_k, alpha_k y a_barra_k. Se documenta el plan
      **coseno al cuadrado** (`squaredcos_cap_v2`, s = 0,008, beta acotado en 0,999):
      ojo, **`beta_start` y `beta_end` de la config no se usan** con ese plan.
    - **Controles de integridad del zarr (m6).** `../diffuser/scripts/integridad_zarr.py`,
      corre en WSL, abre en lectura y recorre `img` por bloques. Todo limpio: 0 NaN, 0 Inf,
      0 estados duplicados, longitudes 49-246, `img` en [65, 255] float32.
      SHA-256 del arbol: `c235ab79...275e36a4`; 31,0 MB en disco.
    - **Sin anexos y una sola lista bibliografica**, por decision del usuario en esta
      sesion: todo el material de apoyo vive en el repo de GitHub, que ahora **se cita
      formalmente** (`britez2026repo`), igual que la monografia (`britez2026guia`). Se
      retiraron `secciones/anexos.tex`, `bib/bibliografia.bib`, el paquete `multibib` de
      `main.tex` y el `bibtex con` de `compilar.sh`. **Esto revierte la decision 4.**
      El material que iba a ser el anexo se traslado al cuerpo, que es donde el informe lo
      exige: `tab:identificadores` y la preespecificacion en 3.9, `tab:config-efectiva` en
      3.5, `tab:preprocesado` en 3.4 y `tab:integridad` en 3.2. De paso desaparecen las dos
      referencias `??`, que apuntaban al anexo comentado.
    - **Tabla `tab:alcance`** al cierre de las conclusiones: tres columnas con lo
      demostrado para los cinco artefactos, lo no demostrado para las estrategias y lo no
      demostrado fuera del simulador. Responde a las preguntas 7 y 10 del tribunal.
    - Trabajo futuro reescrito con **nueve lineas y diseno concreto**, encabezado por la
      replicacion con varias semillas. Beamer actualizado con las cifras de la prueba
      final. Agradecimientos redactados: **conviene que el usuario los personalice**.
    - Compila limpio: 71 paginas, 1 aviso (parche `footnote` de microtype, inocuo),
      cero `??`, 32 referencias sin avisos de BibTeX.

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
- [x] Redactar conclusiones, resumen y abstract con las cifras finales de V0-V4.
- [x] Redactar resultados y presupuesto. Cifras finales V0-V4: 0,864 / 0,668 / 0,648 /
      0,622 / 0,535; epocas 500 / 500 / 266 / 155 / 200; 1,6 / 5,3 / 14,7 / 2,3 / 4,0
      min por epoca; 237,1 h de tiempo total.
- [ ] Confirmar con el director los supuestos economicos del presupuesto, que son
      estimaciones y no datos: 450 h del autor repartidas por fases, 40 h de direccion,
      30 y 60 \euro/h de coste horario, 1.800 \euro de precio del portatil, siete meses de
      imputacion (marzo-septiembre de 2026) y 15 % de costes indirectos. Si cambia alguno,
      recalcular la Tabla 14; el total actual es 18.686,55 \euro.
- [x] Revisar conclusiones, resumen y abstract con V3 y V4, incluidos los diez contrastes
      por pares y la correccion de Holm.
- [x] Completar la Tabla 7 (`sec:coste`) con V3 y V4: **5 h 52 min de bucle y 10,9 h
      totales en V3** (155 epocas cronometradas, 2 min 16 s por epoca) y **13 h 20 min de
      bucle y 27,4 h totales en V4** (200 epocas, 4 min 0 s por epoca). Ojo: son los
      unicos dos runs con cronometro completo junto a V1, no llevan daga.
- [x] Rehacer el apartado sobre el presupuesto de epocas: la
      metodologia dice 500 como maximo, pero V2 y V3 se lanzaron con 300 y V4 con 200.
- [x] **Rehacer la fila de V0 en la Tabla 7.** Al validar
      `scripts/resumen_entrenamiento.py` aparecio que el log de V0 si conserva las
      **500 epocas cronometradas, no 89**: la extraccion manual de agosto se dejo 411 por
      un patron demasiado estricto. El total medido es **46.793 s = 13,0 h de bucle** y
      la fila ya no lleva daga. El parser conserva ahora los metadatos manuales de los runs
      antiguos cuando no existen los ficheros auxiliares de las exportaciones nuevas.
- [x] Versiones exactas de Hydra y Zarr en el entorno de WSL, consultadas en la maquina de
      entrenamiento el 25 de agosto de 2026: **hydra-core 1.2.0 y zarr 2.12.0**, ya en la
      Tabla 6 y en `../CLAUDE.md`. Faltaban porque ni `CLAUDE.md` ni
      `entorno_wsl.txt` las recogian: el volcado del entorno anoto plataforma, CUDA, Python
      y PyTorch, pero no un `pip freeze`. El `requirements.txt` de la raiz **no** sirve,
      describe el entorno de inferencia en Windows (torch 2.6, hydra 1.3.2, zarr 2.18.7),
      no el de entrenamiento (torch 1.12.1).
- [x] **M4 del informe: latencia de inferencia y pico de VRAM.** Medidos el 26 de agosto
      de 2026 sobre los puntos de control seleccionados, sin reentrenar. Ver la decision 15.
- [ ] Verificar los dos campos marcados `% verificar` en `bib/referencias.bib`.
- [x] Anadir a `bib/referencias.bib` las obras de los codificadores visuales (ResNet,
      ImageNet, DINOv2, CLIP y ViT) y citarlas en el texto.
- [x] Autor y ano de la monografia propia sobre Diffusion Policy: Moises Britez, 2026,
      registrados en `bib/bibliografia.bib`.
- [x] Resultados de V3 (DINOv2) y V4 (CLIP): entrenados y volcados el 26 de agosto de
      2026. Ver la decision 13.
- [ ] Decidir si el encabezado se mantiene con filete y cursiva o se simplifica.
- [x] **Agradecimientos y anexos de plantilla.** Resuelto el 27 de agosto de 2026: los
      anexos se retiraron por completo (el material vive en el repo, que ahora se cita) y
      los agradecimientos se redactaron. Ver la decision 19.
- [ ] **Revisar y personalizar los agradecimientos.** Se redactaron a partir de la nota
      previa («universidad, amigos, familia, profesores, ANDE»). Son sobrios y correctos,
      pero el texto es del agente, no del autor: conviene reescribirlos.
- [x] **Prueba final sobre semillas disjuntas.** Hecha el 27 de agosto de 2026 sobre las
      **cinco** variantes y **200** semillas (`200000-200199`), no las tres y 50 que se
      habian previsto. Ver la decision 18.
- [ ] **Ablaciones de los factores confundidos**: los runs que separan inicializacion,
      preprocesado y normalizacion interna, ~85 h a 224 px y ~17 h a 96 px sin recorte,
      mas uno con normalizacion por lotes. Siguen pendientes y necesitan GPU. Ojo: el
      cuarto factor **no es la agregacion espacial**, que es promediado global en las
      cinco variantes; ver el error 1 de la decision 19.
- [x] **Actualizar `beamer/beamer.tex`.** Hecho el 27 de agosto de 2026: subtitulo,
      resumen, tabla de contrastes, tabla de coste, limitaciones y cierre pasan a las
      cifras de la prueba final (0,872 / 0,649 / 0,586 / 0,578 / 0,490) y se retira el
      *spatial softmax*. **Quedan por revisar** las diapositivas de respaldo y la figura
      de perdidas, que aun describen solo V0-V2.
- [x] **Resolver las dos referencias `??`.** Hechas el 27 de agosto de 2026: apuntaban al
      anexo comentado y el anexo se retiro, de modo que se reapuntaron a
      `tab:identificadores` (3.9) y `tab:config-efectiva` (3.5). El PDF ya no contiene
      ningun `??`.
- [ ] **Replicar con las semillas 43 y 44** (M1, la limitacion que activa el techo de
      69/100). Es la unica via para inferir sobre estrategias y no sobre artefactos.
      Necesita GPU: ~700 h para las cinco variantes, o menos si se reduce la matriz.
      Declarado como primera linea de trabajo futuro.
- [ ] **Replicas de ruido de inferencia** (M5): repetir el bloque de 200 condiciones con
      varias semillas de difusion, ~2,5 h por replica de las cinco variantes. Descartado
      en esta ronda por la decision de no lanzar mas computo.
- [ ] **Rehacer V1 desde los pesos de `torchvision`** para que su contraste con V2 aisle
      de verdad la estrategia de adaptacion (~96 h). Ver el error 2 de la decision 19.
- [x] **M5 del informe: caracterizacion del `zarr`** con distribucion de longitudes y
      puntuaciones y el reparto train/val/descartados explicito (P2.8). Resuelto el 26 de
      agosto de 2026 sin computo de GPU. Ver la decision 17.
- [x] M7 del informe (literatura de preentrenamiento visual para control) **ya estaba
      resuelto**: el apartado 2.5.4 cubre MVP, Voltron y VC-1/CortexBench. El informe
      describe un apartado 2.4 que no corresponde a la version actual.
