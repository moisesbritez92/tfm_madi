# Preregistro de la comparacion entre V0 y el punto de control publicado del articulo

**Fecha:** 31 de agosto de 2026.
**Autor:** Moises Britez.
**Estado:** cerrado antes de ejecutar la evaluacion del brazo nuevo. La marca de tiempo del
commit que introduce este fichero es la evidencia de que el protocolo precede a los
resultados, con la salvedad parcial que se declara en la seccion cuarta.

## Por que existe este documento

El director pide comparar la mejor de las cinco variantes entrenadas, V0, con el punto de
control que los autores de *Diffusion Policy* publicaron para Push-T. La comparacion no es
un anadido decorativo: el informe de evaluacion externa
(`evaluacion_tfm_reevaluacion.md`, lineas 470 y 582) admite a V0 como linea base pertinente
**porque reproduce la opcion de referencia del articulo**, y esa afirmacion nunca se ha
contrastado contra el artefacto real. Este documento fija el procedimiento para
contrastarla.

De paso responde a dos hallazgos mayores que siguen abiertos:

- **Hallazgo M9.** La memoria usa la palabra «comparable» sin margen de equivalencia y sin
  TOST. El evaluador exige «definir un margen practico antes de observar los datos y aplicar
  un analisis de equivalencia». Sin margen, no rechazar una diferencia no demuestra nada.
  Aqui se fija `delta = 0,05` antes de mirar.
- **Hallazgo M5.** El evaluador exige «fijar varias semillas de inferencia por condicion,
  manteniendo numeros aleatorios comunes dentro de cada replica» y declarar si el estimando
  integra la condicion, el ruido o ambos. Aqui se ejecutan **dos realizaciones de ruido por
  brazo** y el estimando integra los dos niveles.

## Puntos de control congelados

No se elige ningun punto de control nuevo. V0 es el que la memoria ya reporta. V_Paper es el
unico fichero publicado por los autores que hay en el repositorio, tal como se descargo.

| Brazo | Codificador | Punto de control | SHA-256 |
|---|---|---|---|
| V0 | ResNet-18 desde cero | `epoch=0350-test_mean_score=0.865.ckpt` | `5310551ee71075d9efcf956c34670809741d84e06808809551e7675674e8ce63` |
| V_Paper | ResNet-18 con *spatial softmax* de robomimic | `epoch=0500-test_mean_score=0.884.ckpt` | `bac7221f7e34cd51162dc1972e1a39ffcddc87de1dc1780c44ffa61b88c4ff76` |

V0 reside en `~/tfm/diffusion_policy/data/outputs/encoder_exp/v0_seed42/checkpoints/`, con
copia en `diffuser/models/V0/`. V_Paper reside solo en `diffuser/models/V_Paper/` y se lee
desde WSL a traves de `/mnt/c`. V_Paper no tiene directorio de ejecucion ni
`.hydra/config.yaml`: su configuracion viaja dentro del propio fichero, en `payload["cfg"]`,
y de ahi se lee.

**El `0,884` del nombre del fichero no es una cifra de este trabajo y no es comparable con
las del TFM.** Los autores la midieron sobre las semillas `4300000-4300049`, un bloque
distinto tanto del conjunto de seleccion `100000-100049` como del bloque final
`200000-200199`. Se usa aqui solo como criterio del segundo porton de cordura.

## Que compara y que no compara este contraste

V_Paper **no es una sexta variante del experimento** y no se incorpora a la tabla de las
cinco. Comparte con V0 el conjunto de datos y su particion (`seed 42`, `val_ratio 0.02`, 90
episodios de entrenamiento), la red de difusion (`down_dims [512, 1024, 2048]`), el
planificador DDPM de 100 pasos y los horizontes `horizon 16`, `n_obs_steps 2`,
`n_action_steps 8`.

Difiere en **cuatro factores a la vez**, y ninguno queda aislado:

1. **El codificador visual.** V_Paper usa el codificador de robomimic: ResNet-18 con
   `SpatialSoftmax` de 32 puntos clave, capa lineal de 64 a 64 y recorte de 84 por 84 con
   `eval_fixed_crop`. V0 usa `MultiImageObsEncoder` con promediado global, recorte de 76 por
   76 y `use_group_norm`.
2. **El presupuesto de entrenamiento.** El calendario de los autores es de 3050 epocas y su
   punto de control publicado es el de la epoca 500. El de V0 es de 500 epocas y su punto de
   control es el de la epoca 350.
3. **El bloque de seleccion.** Los autores eligieron con `4300000-4300049`; V0 se eligio con
   `100000-100049`. Ambos son disjuntos del bloque final, de modo que la prueba es limpia
   para los dos, pero el criterio de seleccion no fue el mismo.
4. **El linaje de implementacion.** Codigo original de los autores frente a la bifurcacion
   propia.

**La unidad de inferencia sigue siendo el artefacto entrenado, no la estrategia.** Hay una
sola ejecucion de entrenamiento por brazo, de modo que nada de lo que sigue estima la
variacion entre entrenamientos. Esa limitacion (Hallazgo M1) permanece abierta.

Lo que este contraste si establece: si el artefacto que se produjo desde cero en este
trabajo iguala al artefacto publicado, sobre el mismo bloque de condiciones y el mismo
protocolo de evaluacion.

## Integridad parcial de este preregistro

Hay que decirlo sin adornarlo. Las puntuaciones de V0 en la realizacion de ruido A ya
existen y son conocidas: son las de `logs_entrenamiento/prueba_final/prueba_v0.json`, con
media `0,871873`. Este documento fija por adelantado el brazo nuevo entero, la realizacion B
de **ambos** brazos y la totalidad del analisis. No puede fijar lo que ya se observo.

La consecuencia practica es que el preregistro protege contra la eleccion oportunista del
analisis y contra la eleccion oportunista del brazo nuevo, que es donde esta el riesgo real,
pero no equivale a un preregistro de un experimento entero sin observar. Se declara asi en la
memoria.

## Entorno de ejecucion

La evaluacion de V_Paper se hace **en WSL con el entorno `robodiff`**, el mismo que produjo
los cinco JSON de la prueba final: torch 1.12.1, CUDA 11.6, robomimic 0.2.0. **No** se usa
`.venv_diffuser_infer` de Windows, aunque alli tambien se instalo robomimic y el punto de
control carga. El motivo es que un cambio de version de torch entre los dos brazos
introduciria una diferencia de procedimiento dentro del propio contraste pareado, y el
contraste dejaria de medir lo que dice medir.

## Bloque de evaluacion

- Semillas del entorno `200000` a `200199`, es decir **n = 200** condiciones iniciales. Son
  exactamente las de la prueba final, de modo que la comparacion es pareada por condicion.
- El bloque es disjunto de las semillas `0-205` que generaron las demostraciones, de las
  `100000-100049` con las que se eligio el punto de control de V0 y de las
  `4300000-4300049` con las que los autores eligieron el suyo.
- Se evalua con `legacy_test = true`, `max_steps = 300`, `n_obs_steps = 2`,
  `n_action_steps = 8` y `fps = 10`. El script comprueba por asercion que los valores del
  evaluador de V_Paper coinciden con los registrados en `prueba_v0.json`; si no coincidieran,
  la comparacion no seria del mismo experimento y no se ejecuta.
- `n_envs = 8`, en 25 tandas exactas de ocho condiciones.
- Se emplean los pesos de la media movil exponencial en los dos brazos.
- El script se niega a sobrescribir un resultado ya escrito.

## Estocasticidad de inferencia

Se ejecutan **dos realizaciones de ruido de difusion por brazo**, no una. Es la correccion
que pide el Hallazgo M5.

- Realizacion A: semilla base `20260827`, la misma de la prueba final.
- Realizacion B: semilla base `20260831`, **declarada aqui y no elegida despues**.
- Antes de cada llamada a la politica se fija
  `torch.manual_seed(base * 1000003 + tanda * 1000 + paso)`, donde `tanda` es el indice del
  grupo de ocho condiciones y `paso` el indice del ciclo de control dentro de la tanda. Es el
  mismo mecanismo de `diffuser/scripts/evaluar_bloque_test.py`, sin modificar.
- Dentro de cada realizacion, los dos brazos comparten los numeros aleatorios (*common
  random numbers*), lo que exige que el tensor de ruido tenga la misma forma `(8, 16, 2)` en
  los dos y que el codificador no consuma generador antes de la difusion. Con `crop_shape`
  fijo en evaluacion el recorte es central en los dos brazos, de modo que no deberia
  consumirlo; **no se da por supuesto y se comprueba en el tercer porton**.
- V0 en la realizacion A **no se vuelve a ejecutar**: son las puntuaciones ya publicadas de
  `prueba_v0.json`. El primer porton verifica que la ruta de codigo sigue reproduciendolas.

El determinismo del esquema esta comprobado: `logs_entrenamiento/prueba_final/crn_a_v4.json`
y `crn_b_v4.json` son dos ejecuciones de V4 sobre las ocho semillas `300000-300007` con la
misma semilla base, y sus puntuaciones son identicas bit a bit; lo unico que difiere entre
los dos ficheros es la etiqueta y el tiempo de pared. Esa evidencia estaba en el repositorio
sin que ningun documento la citara.

## Estimando, unidad y endpoints

- **Estimando primario:** diferencia entre la puntuacion media de cobertura de V0 y la de
  V_Paper sobre el generador de condiciones iniciales de Push-T **y sobre la realizacion del
  ruido de difusion**. El estimando integra los dos niveles; esa es la respuesta explicita al
  Hallazgo M5.
- **Unidad de observacion:** la condicion inicial, n = 200. Las dos realizaciones de ruido se
  promedian **dentro** de cada condicion antes de contrastar, de modo que las 200 unidades
  siguen siendo independientes y no se inflan a 400.
- **Endpoint primario:** media de las 200 diferencias pareadas `d_i = s_V0(i) - s_Paper(i)`,
  donde cada `s` es la media de las dos realizaciones en esa condicion.
- **Endpoint secundario 1:** tasa de exito, fraccion de condiciones con puntuacion mayor o
  igual a `0,999`, calculada **por realizacion** y contrastada con la prueba exacta de
  McNemar sobre los pares discordantes. No se agrega entre realizaciones, porque el promedio
  de dos indicadores no es un indicador.
- **Endpoint secundario 2:** componente de varianza del ruido, es decir la varianza
  intra-condicion entre las dos realizaciones, por brazo. Es la cifra que el Hallazgo M5 pide
  cuando habla de «componente de varianza correspondiente».

## Analisis, fijado antes de ver los datos

1. Por brazo y realizacion: media, desviacion tipica, error estandar y tasa de exito sobre
   las 200 condiciones.
2. **Contraste primario: prueba de permutacion por inversion de signo** sobre la media de las
   200 diferencias pareadas. Bilateral, `B = 10000` permutaciones,
   `p = (1 + k) / (B + 1)` con `k` el numero de permutaciones tan extremas o mas que la
   observada, y generador `numpy.random.default_rng(42)`. Es la funcion `permutacion_media`
   que ya existe en `memoria/scripts/analisis_prueba_final_v2.py`, que se importa y no se
   copia. Se elige la permutacion y no Wilcoxon como prueba primaria porque el estimando
   declarado es una diferencia de medias y Wilcoxon opera sobre rangos; esa objecion es la
   que motivo el propio `analisis_prueba_final_v2.py`.
3. Intervalo de confianza del 95 % de la diferencia media por **bootstrap BCa con 10.000
   remuestreos** de los pares, con semilla `42`, reutilizando la funcion `ic_bca` de
   `memoria/scripts/analisis_prueba_final.py`. Es descriptivo.
4. **Equivalencia. Margen `delta = 0,05`, fijado aqui.** Se declara equivalencia practica si
   el intervalo BCa **al 90 %** de la diferencia media queda contenido por completo en el
   intervalo `(-0,05, +0,05)`. Ese criterio es el TOST a `alfa = 0,05`.
   El margen se justifica por la dispersion de repetir la propia medicion: tres evaluaciones
   de estados de entrenamiento casi identicos de V3 se separan `0,097` y dos de V4 se separan
   `0,047`, del orden de un error estandar del rollout, alrededor de `0,055`. Una diferencia
   menor que la dispersion de volver a medir no tiene contenido practico. `delta = 0,05` es
   ademas la cuarta parte de la brecha entre V0 y V1 en la prueba final, `0,223`, de modo que
   no absorbe ninguna de las diferencias que la memoria si considera relevantes.
5. **Regla de decision, con las cuatro casillas escritas antes de mirar:**

   | | `p < 0,05` | `p >= 0,05` |
   |---|---|---|
   | **IC90 dentro de ±delta** | diferencia detectable pero **practicamente irrelevante** | **equivalencia practica** |
   | **IC90 fuera de ±delta** | **diferencia relevante** | **indeterminado**, potencia insuficiente |

   El informe nombrara la casilla en la que cae el resultado, sea cual sea.
6. La prueba de los rangos con signo de Wilcoxon se calcula y se reporta como comprobacion de
   robustez, no como prueba primaria. Los ceros se descartan por el metodo de Wilcoxon.
7. **Multiplicidad: no hay que corregir nada.** Hay un solo contraste primario, entre dos
   brazos, sobre un solo endpoint primario. Los dos endpoints secundarios se reportan sin
   correccion y etiquetados como secundarios.
8. Los contrastes de V_Paper contra V1, V2, V3 y V4 **no se ejecutan**. Si alguna vez se
   ejecutaran, serian exploratorios y no entrarian en la familia de diez comparaciones de
   `preregistro_prueba_final.md`, que no se toca: las cifras de la prueba final se quedan
   exactamente como estan.
9. Ningun analisis adicional se anadira despues de ver los resultados sin declararlo
   expresamente como exploratorio.

## Comprobaciones previas

Tres portones, con su criterio numerico fijado de antemano, antes de gastar las corridas
buenas.

1. **Deriva de la ruta de codigo.** Se reevalua V0 con la semilla base `20260827` sobre las
   ocho primeras condiciones del bloque, `200000-200007`. Criterio de paso: las ocho
   puntuaciones deben coincidir **bit a bit** con las ocho primeras de `prueba_v0.json`. Si
   no coinciden, `prueba_v0.json` no se reutiliza y hay que ejecutar tambien V0 en la
   realizacion A. Coste, alrededor de un minuto.
2. **Cordura de V_Paper.** Se evalua V_Paper sobre **su propio** bloque publicado,
   `4300000-4300049`, n = 50. Criterio de paso: la media debe caer a menos de `0,07` de los
   `0,884` que los autores dejaron en el nombre del fichero. La tolerancia son dos errores
   estandar de una media de 50 episodios con desviacion tipica del orden de `0,25`. No se
   espera coincidencia exacta, porque el estado del generador aleatorio de los autores no era
   este. Un desvio mayor significaria que la ruta de inferencia no reproduce la de los
   autores, y en ese caso la comparacion **no se ejecuta**. Coste, alrededor de nueve
   minutos.
3. **Alineacion del ruido comun.** Con la misma semilla fijada a mano, se captura el primer
   tensor de ruido que consume `conditional_sample` en cada brazo y se comprueba que son
   identicos. Coste, segundos.
   **Contingencia declarada por adelantado:** si no coinciden, por ejemplo porque el
   codificador de robomimic consumiera generador antes de la difusion, el pareo por condicion
   inicial sigue siendo valido y **el analisis no cambia en nada**. Lo unico que se pierde es
   la reduccion de varianza de los numeros aleatorios comunes, y se declara asi en el
   informe. No es motivo para abortar.

## Compromisos

1. **No se vuelve a seleccionar despues de ver el bloque.** Si otro punto de control de V0
   rindiera mejor sobre `200000-200199`, ese dato no se usa para cambiar la eleccion. Del
   lado del articulo solo hay un fichero publicado y no hay nada que elegir.
2. **El tamano de muestra esta fijado aqui**, en 200 condiciones por dos realizaciones de
   ruido, y no se ampliara ni reducira en funcion del resultado.
3. **El margen de equivalencia esta fijado aqui**, en `0,05`, y no se movera despues de ver
   el intervalo.
4. Una corrida ya escrita no se repite. Si la ejecucion se interrumpe, se reanuda por corrida.
5. **La prueba final de las cinco variantes no se toca.** Este analisis no reescribe
   `prueba_final_resumen*.csv` ni `prueba_final_contrastes*.csv`, y se comprueba que no lo
   hace.
6. Sea cual sea el resultado, se reporta. Si V_Paper supera a V0 de forma relevante, ese es
   el resultado y la memoria lo dira; si el segundo porton falla, se reportara que la cifra
   publicada no se reproduce en esta ruta de inferencia, en lugar de omitirlo.

## Ejecucion

```bash
# los tres portones
wsl -d Ubuntu -- bash /mnt/c/Users/moise/Documents/0001_MADI/TFM/diffuser/scripts/evaluar_paper_bloque_test.sh --portones
# las tres corridas buenas: V_Paper en A y en B, V0 en B
wsl -d Ubuntu -- bash /mnt/c/Users/moise/Documents/0001_MADI/TFM/diffuser/scripts/evaluar_paper_bloque_test.sh
# analisis, en Windows
python memoria/scripts/analisis_comparacion_paper.py
```

Resultados en bruto: `logs_entrenamiento/prueba_final/prueba_paper.json`,
`ruido_b_paper.json` y `ruido_b_v0.json`, con las 200 parejas semilla-puntuacion, el SHA-256
del punto de control, la configuracion efectiva del evaluador, las versiones de la pila de
software y, en los ficheros nuevos, la fecha en formato ISO y el commit de git. Los dos
ultimos campos faltaban en los cinco ficheros de la prueba final, carencia recogida en el
Hallazgo m8.
