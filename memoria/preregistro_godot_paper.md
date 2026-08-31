# Preregistro de la comparacion entre V0 y el punto de control publicado, dentro de Godot

**Fecha:** 31 de agosto de 2026.
**Autor:** Moises Britez.
**Estado:** cerrado antes de ejecutar ninguna de las ocho corridas. La marca de tiempo del
commit que introduce este fichero es la evidencia de que el protocolo precede a los
resultados. A diferencia del preregistro hermano, aqui **no hay integridad parcial**: ninguna
de las ocho celdas existe todavia.

## Por que existe este documento

El 31 de agosto de 2026 se cerro el contraste entre V0 y `V_Paper` sobre el bloque disjunto
`200000-200199`, en el simulador original: diferencia media pareada `+0,0408` a favor de V0,
IC95 BCa `[+0,0049, +0,0792]`, permutacion `p = 0,036`. Lo que quedo establecido es que **V0
no es peor que el punto de control publicado** sobre la distribucion de imagenes con la que
los dos se entrenaron.

Ese contraste no dice nada sobre que pasa cuando la distribucion cambia, y el repositorio ya
tiene el aparato para cambiarla. `diffuser/godot/` reimplementa Push-T en Godot 4.7.2 con dos
condiciones: **A**, la fisica la simula Godot y la imagen la dibuja el rasterizador del
entrenamiento; **B**, la fisica la simula Godot y los pixeles tambien son de Godot. Y ya hay
evidencia de que ese cambio de dominio separa modelos que en el simulador original no se
separan: sobre cinco semillas donde V0 y V3 puntuan los dos `1,000` en la prueba final, en
condicion B V0 promedia `0,995` y V3, `0,429`.

La pregunta que este documento fija: **sobrevive la paridad entre V0 y V_Paper al cambio de
motor de fisica y de renderizador, o uno de los dos codificadores esta mas sintonizado al
renderizador de pygame que el otro.** Es una hipotesis con contenido y no una pesca:
`V_Paper` usa el codificador de robomimic, ResNet-18 con `SpatialSoftmax` de 32 puntos clave
y recorte 84 por 84, que localiza posiciones; V0 usa `MultiImageObsEncoder` con promediado
global y recorte 76 por 76. No hay razon para esperar que dos mecanismos tan distintos
degraden igual cuando cambia el renderizador.

## Cambio de estatus de `diffuser/godot/`, acotado

Hasta hoy esa carpeta **no produce cifras reportables**, por construccion: un episodio por
celda y semillas elegidas condicionando en el exito previo. `perturbaciones.md` y
`barrido_color.md` siguen siendo bitacoras y no cambian de estatus.

Este documento crea la unica excepcion, y esta acotada a lo que declara: **dos brazos,
cincuenta condiciones iniciales no condicionadas al exito, dos condiciones de observacion y
dos realizaciones de ruido**. Todo lo demas que salga de Godot sigue siendo ilustrativo. En
particular, esto **no** convierte la demostracion de la defensa en una medicion, y **no**
mide las cinco variantes.

## Puntos de control congelados

No se elige ningun punto de control nuevo. Son exactamente los dos del preregistro hermano,
verificados por SHA-256 antes de cada corrida.

| Brazo | Codificador | Punto de control | SHA-256 |
|---|---|---|---|
| V0 | ResNet-18 desde cero | `epoch=0350-test_mean_score=0.865.ckpt` | `5310551ee71075d9efcf956c34670809741d84e06808809551e7675674e8ce63` |
| V_Paper | ResNet-18 con *spatial softmax* de robomimic | `epoch=0500-test_mean_score=0.884.ckpt` | `bac7221f7e34cd51162dc1972e1a39ffcddc87de1dc1780c44ffa61b88c4ff76` |

Los dos se leen de `diffuser/models/`, que es donde estan las copias de Windows.

## Que compara y que no compara este contraste

Los **cuatro factores que ya confundia** el contraste en el simulador original siguen
confundidos aqui, y hay que repetirlo en cualquier texto que cite estas cifras: difieren a la
vez el codificador, el presupuesto de entrenamiento (3050 epocas con punto de control en la
500, frente a 500 con punto de control en la 350), el bloque de seleccion y el linaje de
implementacion.

Este trabajo **no separa ninguno de ellos**. Anade un quinto eje, el dominio de evaluacion, y
lo que mide de forma limpia es la **degradacion de cada artefacto respecto de si mismo** al
cambiar de dominio. La unidad de inferencia sigue siendo el artefacto entrenado, no la
estrategia de entrenamiento: hay una sola ejecucion de entrenamiento por brazo y nada de lo
que sigue estima la variacion entre entrenamientos. El Hallazgo M1 permanece abierto.

## Entorno de ejecucion

Todo corre en **Windows**, con `.venv_diffuser_infer` (Python 3.11, torch 2.6.0+cu124, timm
1.0.7, robomimic 0.2.0 instalado con `--no-deps`), porque Godot corre en Windows y el
servidor de politica tiene que hablar con el por TCP en la misma maquina.

Esto **difiere** del preregistro hermano, que exigia WSL y `robodiff` con torch 1.12.1. La
razon de aquella exigencia era no meter un cambio de version de torch **dentro** de un
contraste pareado. Aqui esa razon se respeta igual: **los dos brazos corren en el mismo
entorno de Windows**, de modo que el cambio de torch es comun a los dos y no entra en la
diferencia pareada. Lo que si introduce es una diferencia respecto de las cifras de
`logs_entrenamiento/prueba_final/`, y por eso existe el segundo porton, que la acota antes de
gastar nada.

## Bloque de evaluacion

- Semillas del entorno `200000` a `200049`, es decir **n = 50** condiciones iniciales.
- El bloque es **prefijo del bloque final** `200000-200199`, de modo que cada condicion tiene
  una cifra de referencia ya publicada en `prueba_final/` para los dos brazos.
- Es disjunto de las semillas `0-205` que generaron las demostraciones, de las
  `100000-100049` con las que se eligio el punto de control de V0 y de las `4300000-4300049`
  con las que los autores eligieron el suyo.
- **Las semillas no estan condicionadas al exito de ninguna variante.** Es la diferencia
  metodologica central con `perturbaciones.md`, cuyas semillas si lo estaban y cuyas cifras
  absolutas por eso no son interpretables.
- **Advertencia de contaminacion, declarada:** once semillas del bloque
  (`200000`-`200005`, `200007`, `200019`, `200021`, `200023` y `200024`) ya se han visto en
  Godot con V0 durante la exploracion de `perturbaciones.md` y `barrido_color.md`. No se
  selecciono nada a partir de aquellos episodios y no se ejecuto ninguno con V_Paper, de modo
  que no sesgan el contraste entre brazos; se declara aqui para que no haya que descubrirlo
  despues.
- El episodio termina al superar cobertura `0,95` o a los **300 pasos de control**, con
  `n_obs_steps = 2` y `n_action_steps = 8`, que son los del entrenamiento y los de la prueba
  final. La puntuacion del episodio es el **maximo** de la cobertura a lo largo del episodio,
  igual que en `PushTEnv`.
- `perturbacion = ninguna` en las cuatrocientas celdas. Este contraste no perturba la escena.

## Las dos condiciones de observacion

| Condicion | Bandera | Que cambia respecto del entrenamiento |
|---|---|---|
| A | `--obs estado` | Solo el motor de fisica. La imagen la dibuja `servidor/rasterizador_pusht.py` con el codigo del entrenamiento. |
| B | `--obs godot` | El motor de fisica **y** los pixeles, que salen del `SubViewport` ortogonal cenital. |

**B es la condicion primaria.** A es el control que hace atribuible cualquier caida: sin ella,
una degradacion en B no se puede repartir entre el cambio de motor y el cambio de pixeles.

Se ejecutan sobre **las mismas cincuenta semillas** y con las mismas dos semillas base de
ruido, de modo que A y B tambien estan pareadas por condicion inicial.

## Estocasticidad de inferencia, y una limitacion propia de Godot

Se ejecutan **dos realizaciones de ruido de difusion por brazo y condicion**, con las semillas
base del preregistro hermano: realizacion A `20260827` y realizacion B `20260831`. Antes de
cada llamada a la politica el servidor fija
`torch.manual_seed(base * 1000003 + semilla_episodio * 1000 + paso)`, que es el esquema de
`servidor_politica.py` y no se modifica.

**Los episodios de Godot no son reproducibles bit a bit, y esto se declara por adelantado.**
Esta medido: la semilla `200003` en condicion A resolvio la tarea en 183 pasos con cobertura
`0,9726` una vez y en 242 pasos con `0,9545` otra. El ruido de difusion si esta sembrado; lo
que no es determinista son las convoluciones en GPU, y una simulacion con contactos amplifica
cualquier diferencia diminuta.

La consecuencia es que la componente de varianza que separa las dos realizaciones **mezcla**
el cambio de semilla de ruido con el no determinismo de la GPU, y no se puede atribuir a uno
de los dos. Se reporta asi. No invalida el contraste pareado: esa varianza esta presente en
los dos brazos y las dos realizaciones se promedian dentro de cada condicion antes de
contrastar.

## Estimando, unidad y endpoints

- **Estimando primario:** diferencia entre la puntuacion media de V0 y la de V_Paper **en
  condicion B**, sobre el generador de condiciones iniciales de Push-T y sobre la realizacion
  del ruido. El estimando integra los dos niveles.
- **Unidad de observacion:** la condicion inicial, n = 50. Las dos realizaciones se promedian
  **dentro** de cada condicion antes de contrastar, de modo que las 50 unidades siguen siendo
  independientes y no se inflan a 100.
- **Endpoint primario:** media de las 50 diferencias pareadas `d_i = s_V0(i) - s_Paper(i)` en
  condicion B, donde cada `s` es la media de las dos realizaciones en esa condicion.
- **Secundarios, todos sin correccion de multiplicidad y etiquetados como tales:**
  1. El mismo contraste en **condicion A**.
  2. La **caida A a B** por brazo, es decir la media de `s_A(i) - s_B(i)`, con su IC95 BCa.
     Es la cifra que responde a la pregunta del documento.
  3. La **deriva respecto de `prueba_final/`** por brazo y condicion, comparando contra las
     50 primeras puntuaciones de `prueba_v0.json` y `prueba_paper.json`.
  4. **Tasa de exito**, fraccion de condiciones con puntuacion mayor o igual a `0,999`,
     calculada **por realizacion** y contrastada con McNemar exacto sobre los pares
     discordantes. No se agrega entre realizaciones, porque el promedio de dos indicadores no
     es un indicador.
  5. **Componente de varianza entre realizaciones**, por brazo y condicion, con la salvedad
     de atribucion declarada en la seccion anterior.

## Potencia, calculada antes de mirar

Del contraste en el simulador original: n = 200 y EE `0,0189`, de donde la desviacion tipica
de las diferencias pareadas es aproximadamente `0,267`. Con **n = 50** y las dos
realizaciones promediadas, el error estandar esperado es **aproximadamente `0,038`**.

A dos colas, con `alfa = 0,05` y 80 % de potencia, el **efecto minimo detectable es
aproximadamente `0,105`**.

Se dice sin rodeos lo que eso implica: **este estudio no esta dimensionado para replicar el
`+0,041` que se midio en pymunk**, y no encontrar diferencia no sera evidencia de que no la
hay. Esta dimensionado para detectar una separacion grande, del orden de la que ya se observo
entre V0 y V3 en condicion B. El tamano lo fija el coste: la condicion B necesita ventana y no
se puede paralelizar, y cuatrocientos episodios a unos dos minutos son alrededor de trece
horas de maquina.

## Analisis, fijado antes de ver los datos

1. Por brazo, condicion y realizacion: media, desviacion tipica, error estandar y tasa de
   exito sobre las 50 condiciones.
2. **Contraste primario: prueba de permutacion por inversion de signo** sobre la media de las
   50 diferencias pareadas de la condicion B. Bilateral, `B = 10000` permutaciones,
   `p = (1 + k) / (B + 1)`, generador `numpy.random.default_rng(42)`. Es la funcion
   `permutacion_media` de `memoria/scripts/analisis_prueba_final_v2.py`, que se **importa y no
   se copia**. Se elige permutacion y no Wilcoxon como primaria porque el estimando declarado
   es una diferencia de medias y Wilcoxon opera sobre rangos.
3. Intervalo de confianza del 95 % de la diferencia media por **bootstrap BCa con 10.000
   remuestreos**, semilla `42`, reutilizando `ic_bca` de `analisis_prueba_final.py`. Es
   descriptivo.
4. **Equivalencia. Margen `delta = 0,05`, el mismo del preregistro hermano y fijado aqui de
   nuevo antes de mirar.** Se declara equivalencia practica si el intervalo BCa **al 90 %**
   queda contenido por completo en `(-0,05, +0,05)`. Ese criterio es el TOST a `alfa = 0,05`.
   El margen se justifica igual que alli: tres evaluaciones de estados de entrenamiento casi
   identicos de V3 se separan `0,097` y dos de V4 se separan `0,047`, del orden de un error
   estandar del rollout, alrededor de `0,055`. Una diferencia menor que la dispersion de
   volver a medir no tiene contenido practico.
   **Advertencia declarada:** con EE `0,038`, el IC90 mide alrededor de `0,125` de ancho y
   **no cabe en un intervalo de `0,10`**. Es decir, **este diseno no puede declarar
   equivalencia**, pase lo que pase. La casilla «equivalencia practica» esta, en la practica,
   fuera de alcance, y se deja en la tabla para no reescribir la regla de decision despues de
   verla. Que no se pueda declarar equivalencia no es un resultado.
5. **Regla de decision, con las cuatro casillas escritas antes de mirar:**

   | | `p < 0,05` | `p >= 0,05` |
   |---|---|---|
   | **IC90 dentro de +/- delta** | diferencia detectable pero **practicamente irrelevante** | **equivalencia practica** |
   | **IC90 fuera de +/- delta** | **diferencia relevante** | **indeterminado**, potencia insuficiente |

   El informe nombrara la casilla en la que cae el resultado, sea cual sea, y con n = 50 la
   casilla mas probable a priori es «indeterminado».
6. Wilcoxon se calcula y se reporta como comprobacion de robustez, no como prueba primaria.
7. **Multiplicidad: no hay que corregir nada en el primario.** Hay un solo contraste
   primario, entre dos brazos, sobre un solo endpoint primario y una sola condicion. Los cinco
   secundarios se reportan sin correccion y etiquetados como secundarios, y **no se ascienden
   a primarios** si salen mas favorables.
8. **La familia de diez comparaciones de `preregistro_prueba_final.md` no se toca**, y
   tampoco el contraste de `preregistro_comparacion_paper.md`. Este analisis escribe ficheros
   nuevos y se comprueba que no reescribe `prueba_final_*.csv` ni `comparacion_paper_*.csv`.
9. Ningun analisis adicional se anadira despues de ver los resultados sin declararlo
   expresamente como exploratorio.

## Comprobaciones previas

Tres portones, con su criterio numerico fijado de antemano, antes de gastar las trece horas.

1. **Protocolo, sin GPU.** SHA-256 de los dos puntos de control contra los de la tabla de
   arriba; el bloque `200000-200049` disjunto de las demostraciones y de los dos conjuntos de
   seleccion; Godot localizable; `.venv_diffuser_infer` con `robomimic` importable. Coste,
   segundos. Si falla, no se ejecuta nada.
2. **Deriva de entorno.** Con el servidor en `--obs estado` y `servidor/cliente_prueba.py`,
   que cierra el bucle contra el **`PushTEnv` original de pymunk**, se evalua cada brazo sobre
   las ocho condiciones `200000-200007`. Eso aisla el cambio de torch 1.12/WSL a torch
   2.6/Windows y **nada mas**: ni el motor de Godot ni sus pixeles intervienen.
   Referencia: `logs_entrenamiento/prueba_final/deriva_v0.json`, media `0,9242` en esas ocho
   semillas, para V0; y las ocho primeras puntuaciones de `prueba_paper.json` para V_Paper.
   **Criterio de paso: desvio de la media menor o igual a `0,07`**, la misma tolerancia del
   porton de cordura del preregistro hermano. **No se exige coincidencia bit a bit, y no
   podria exigirse:** ademas de que cambian torch y CUDA, el servidor siembra el ruido como
   `base * 1000003 + semilla_episodio * 1000 + paso` mientras que el evaluador de WSL lo
   siembra por indice de tanda, de modo que **la corriente de ruido no es la misma por
   construccion**. Lo que este porton acota es, por tanto, el efecto conjunto de cambiar de
   entorno y de corriente de ruido con la fisica fija en pymunk; no aisla la version de
   torch. Sigue sirviendo para lo que se necesita, que es separar «todo lo que no es Godot»
   de «Godot».
   El mismo porton comprueba que el `estado0` que devuelve `reset` coincide **exactamente**
   con el que muestrea el entorno en local, que es la razon de que el sorteo del estado
   inicial no se moviera a GDScript.
   **Contingencia declarada:** si el desvio supera `0,07` en algun brazo, el contraste **si se
   ejecuta igualmente** —los dos brazos comparten entorno y la diferencia pareada sigue siendo
   valida—, pero la comparacion contra las cifras de `prueba_final/` se retira del informe y
   el secundario 3 no se reporta.
3. **El brazo nuevo cierra el bucle en Godot.** Un episodio, semilla `200000`, condicion B,
   con ventana. Criterio de paso: el episodio termina, escribe su JSON, y **no** aparece por
   consola el aviso de `_avisar_historial_plano`. Ese aviso significa que las dos
   observaciones son identicas pese a que los estados difieren, es decir que el cerrojo
   `_congelado` de `vista3d.gd` esta roto; el sintoma seria una puntuacion algo peor y
   ninguna otra senal. Si aparece, **no se ejecuta nada** hasta arreglarlo.

## Compromisos

1. **El tamano de muestra esta fijado aqui**, en 50 condiciones por dos realizaciones por dos
   condiciones de observacion, y no se ampliara ni reducira en funcion del resultado. En
   particular, **no se anadiran semillas si el resultado sale indeterminado**: eso seria
   muestreo opcional.
2. **El margen de equivalencia esta fijado aqui**, en `0,05`, y no se movera.
3. **La condicion primaria es B**, fijada aqui, y no se cambiara a A si A resulta mas
   favorable.
4. Una celda ya escrita no se repite. Si la ejecucion se interrumpe, se reanuda por celda con
   `--reanudar`; repetir una celda ya escrita exige `--forzar` y una razon anadida a este
   documento.
5. Un episodio que agote el tiempo de espera se reintenta **una sola vez**, y el reintento
   queda anotado en el fichero de salida. Un segundo fallo se registra como celda perdida y se
   declara en el informe; no se sustituye por otra semilla.
6. **No se seleccionan puntos de control.** Son los dos congelados y no hay nada que elegir.
7. **Las bitacoras de `diffuser/godot/` no se mezclan con esto.** `perturbaciones.md` y
   `barrido_color.md` siguen sin ser mediciones, y sus cifras no entran en ninguna tabla junto
   a las de aqui.
8. Sea cual sea el resultado, se reporta. Si V_Paper aguanta mejor el cambio de dominio que
   V0, ese es el resultado y la memoria lo dira.

## Ejecucion

```powershell
# portones 1 y 2
.venv_diffuser_infer\Scripts\python.exe diffuser\godot\servidor\comparar_godot_paper.py --portones

# las ocho corridas: 2 brazos x 2 condiciones x 2 realizaciones x 50 semillas
.venv_diffuser_infer\Scripts\python.exe diffuser\godot\servidor\comparar_godot_paper.py

# analisis
.venv_diffuser_infer\Scripts\python.exe memoria\scripts\analisis_godot_paper.py
```

Resultados en bruto: `logs_entrenamiento/godot_paper/<realizacion>_<brazo>_<condicion>.json`,
ocho ficheros, cada uno con las 50 parejas semilla-puntuacion, el SHA-256 del punto de
control, la semilla base de ruido, las versiones de la pila de software, la fecha en formato
ISO y el commit de git.

Analisis: `memoria/datos/godot_paper_episodios.csv`, `godot_paper_resumen.csv` y
`godot_paper_contrastes.csv`.
