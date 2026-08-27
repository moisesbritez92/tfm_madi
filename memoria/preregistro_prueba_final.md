# Preregistro de la prueba final sobre un bloque disjunto de semillas

**Fecha:** 27 de agosto de 2026.
**Autor:** Moises Britez.
**Estado:** cerrado antes de ejecutar la evaluacion. La marca de tiempo del commit que
introduce este fichero es la evidencia de que el protocolo precede a los resultados.

## Por que existe este documento

El informe de evaluacion externa (`evaluacion_tfm_I.md`, Hallazgo C1 y Prioridad 1.1)
constata que las 50 condiciones iniciales `100000-100049` cumplen dos funciones a la vez:
eligen el punto de control de cada variante y producen la media, el intervalo de confianza y
el valor *p* que la memoria presenta como resultado. No existe medida independiente, de modo
que los intervalos y los contrastes no admiten la lectura confirmatoria que se les atribuye.

La correccion que el evaluador exige, y que este documento fija por adelantado, consiste en
evaluar los puntos de control **ya congelados** una sola vez sobre un bloque de condiciones
que nunca se ha consultado. Todo lo que sigue queda decidido antes de observar ningun
resultado de ese bloque.

## Puntos de control congelados

Se evaluan los cinco puntos de control que la memoria ya reporta. No se elige ninguno nuevo.
Los ficheros residen en `~/tfm/diffusion_policy/data/outputs/encoder_exp/<variante>_seed42/checkpoints/`
y estan copiados en `diffuser/models/V{0..4}/`.

| Variante | Codificador | Punto de control | SHA-256 |
|---|---|---|---|
| V0 | ResNet-18 desde cero | `epoch=0350-test_mean_score=0.865.ckpt` | `5310551ee71075d9efcf956c34670809741d84e06808809551e7675674e8ce63` |
| V1 | ResNet-18 ImageNet congelada | `epoch=0150-test_mean_score=0.668.ckpt` | `bb5012c206c2631ff8960592060eb0409244c9b9319172bd28ed720b0e7c175b` |
| V2 | ResNet-18 ImageNet con ajuste fino | `epoch=0150-test_mean_score=0.648.ckpt` | `36deb633c82033d81ed6b7fb1b16dcbfd0bf42c1dc7fb20b5dfe784fd357eff5` |
| V3 | DINOv2 ViT-S/14 congelada | `epoch=0100-test_mean_score=0.622.ckpt` | `1403e6ba251ac818f3ce1d7a8faf50048517b32c87f3efdf43f751c74f1a412c` |
| V4 | CLIP ViT-B/16 congelada | `epoch=0100-test_mean_score=0.535.ckpt` | `2fcd5857bcbb1417e9e6f159b091b8bffda948e0bb64ab28791502c679bca8b0` |

**Limitacion que se declara aqui y se repetira en la memoria.** El conjunto de candidatos
esta prefiltrado: durante el entrenamiento solo se guardaron los tres mejores puntos de
control de cada variante *segun la metrica contaminada*. Este protocolo acota el sesgo de
seleccion, no lo anula. La afirmacion que sostiene es que la puntuacion medida sobre el
bloque disjunto es una estimacion insesgada del rendimiento **de estos cinco artefactos
concretos**, no del mejor artefacto que cada estrategia podria haber producido.

## Bloque de evaluacion

- Semillas del entorno `200000` a `200199`, es decir **n = 200** condiciones iniciales.
- El bloque es disjunto de las semillas `0-205` que generaron las demostraciones y de las
  `100000-100049` que se usaron para seleccionar. El script lo comprueba con asercion.
- Se usa un desplazamiento de `200000` y no la continuacion `100050` para que el bloque sea
  visualmente inconfundible en las tablas de la memoria.
- Se evalua con `legacy_test = true`, `max_steps = 300`, `n_obs_steps = 2`,
  `n_action_steps = 8` y `fps = 10`, heredados sin modificacion de la configuracion efectiva
  de cada ejecucion de entrenamiento.
- `n_envs = 8`, en 25 tandas exactas de ocho condiciones. La agrupacion no altera las
  metricas: la agregacion del evaluador recorre las condiciones una a una.
- Se emplean los pesos de la media movil exponencial, como en el entrenamiento.
- **Una sola pasada.** No se ejecuta ningun piloto sobre estas semillas. El script se niega a
  sobrescribir un resultado ya escrito.

## Estocasticidad de inferencia

La politica arranca de ruido y anade ruido en cada uno de los cien pasos de difusion. Para
que dos variantes evaluadas en la misma condicion inicial formen un par legitimo, el ruido se
sincroniza entre variantes (*common random numbers*):

- Semilla base `20260827`.
- Antes de cada llamada a la politica se fija
  `torch.manual_seed(20260827 * 1000003 + tanda * 1000 + paso)`, donde `tanda` es el indice
  del grupo de ocho condiciones y `paso` el indice del ciclo de control dentro de la tanda.
- El esquema produce ruido identico entre variantes porque las cinco comparten la forma del
  tensor de ruido `(8, 16, 2)`, el numero de tandas y el numero de llamadas por tanda. En
  modo evaluacion el recorte aleatorio de V0 pasa a ser recorte central, de modo que el
  codificador no consume numeros aleatorios.
- Se ejecuta **una trayectoria de difusion por condicion**, no varias. El estimando queda
  definido en consecuencia: la puntuacion del punto de control en esa condicion bajo esa
  realizacion del ruido, comun a las cinco variantes.

## Estimando y unidad experimental

- **Estimando primario:** puntuacion media de cobertura de un punto de control fijo sobre el
  generador de condiciones iniciales de Push-T, estimada con 200 condiciones muestreadas de
  ese generador mediante semillas consecutivas.
- **Unidad de observacion:** la condicion inicial. La unidad de replicacion **no** es la
  ejecucion de entrenamiento: hay una sola semilla de entrenamiento por variante, de modo que
  nada de lo que sigue estima la variacion entre entrenamientos. Esa limitacion (Hallazgo M1)
  permanece abierta y se declara en la memoria.
- **Endpoint primario:** media de las 200 puntuaciones por variante.
- **Endpoint secundario:** tasa de exito, fraccion de condiciones con puntuacion mayor o
  igual a `0,999`.

## Analisis, fijado antes de ver los datos

1. Por variante: media, desviacion tipica, error estandar y tasa de exito sobre las 200
   condiciones.
2. Diferencias pareadas por condicion inicial entre las diez parejas de variantes.
3. Intervalo de confianza del 95 % de cada diferencia media por **bootstrap BCa con 10.000
   remuestreos** de los pares, con semilla `42`. Se declara asi porque la memoria anterior no
   especificaba el metodo de sus intervalos, deficiencia recogida en el Hallazgo M6.
4. Prueba de los rangos con signo de Wilcoxon sobre los pares. Los ceros se tratan por el
   metodo de Wilcoxon, es decir se descartan. Con 200 pares la distribucion exacta no es
   aplicable, de modo que se usa la aproximacion normal con correccion de continuidad; el
   metodo empleado se registra en cada fila de la tabla de contrastes.
5. Multiplicidad: familia de las **diez** comparaciones por pares del endpoint primario,
   corregida por el procedimiento secuencial de Holm. El umbral de significacion es `0,05`
   sobre los valores ajustados. Los intervalos de confianza no incorporan esa correccion y se
   presentan como descriptivos.
6. Ningun analisis adicional se anadira despues de ver los resultados sin declararlo
   expresamente como exploratorio.

## Compromisos

1. **No se vuelve a seleccionar despues de ver el bloque de prueba.** Si otro punto de
   control de una variante rindiera mejor sobre `200000-200199`, ese dato no se usa para
   cambiar la eleccion.
2. **Si la ordenacion cambia respecto al conjunto de seleccion, el resultado principal es el
   nuevo.** La memoria reportara la prueba final como resultado y el conjunto de seleccion
   como material descriptivo, sea cual sea el sentido del cambio.
3. **El tamano de muestra esta fijado aqui**, en 200, y no se ampliara ni reducira en funcion
   del resultado.
4. Si la ejecucion se interrumpe, se reanuda por variante. Una variante ya evaluada no se
   repite.

## Comprobacion de cordura previa

Antes de gastar la pasada buena se reevalua V0 sobre el conjunto de seleccion
`100000-100049` con la misma ruta de codigo. Criterio de paso fijado de antemano: la media
debe caer a menos de `0,0793` (dos errores estandar) de los `0,8645` registrados durante el
entrenamiento. No se espera coincidencia exacta, porque el estado del generador aleatorio
durante el entrenamiento no era este. Un desvio mayor significaria que la ruta de inferencia
no reproduce la del entrenamiento, y en ese caso la prueba final no se ejecuta.

## Ejecucion

```bash
# comprobacion de cordura
wsl -d Ubuntu -- bash /mnt/c/Users/moise/Documents/0001_MADI/TFM/diffuser/scripts/evaluar_bloque_test.sh --cordura
# pasada final, una sola vez
wsl -d Ubuntu -- bash /mnt/c/Users/moise/Documents/0001_MADI/TFM/diffuser/scripts/evaluar_bloque_test.sh
# analisis, en Windows
python memoria/scripts/analisis_prueba_final.py
```

Resultados en bruto: `logs_entrenamiento/prueba_final/prueba_v{0..4}.json`, con las 200
parejas semilla-puntuacion, el SHA-256 del punto de control, la configuracion efectiva del
evaluador y las versiones de la pila de software.
