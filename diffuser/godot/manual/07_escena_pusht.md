# 07. La escena Push-T y cómo lanzarla

## Objetivo del módulo

Dejar claro qué hay en la escena, qué modos de ejecución existen y cómo se pone
en marcha la demostración el día de la defensa sin tener que recordar rutas.

## Qué se está enseñando

El punto de control congelado de V0 —ResNet-18 desde cero, época 350— resolviendo
Push-T en un motor que no es el que lo entrenó. Es una demostración visual, no
una medición.

> **Ninguna cifra que aparezca en esta ventana es un resultado del TFM.** El
> resultado es la pasada preregistrada sobre el bloque disjunto 200000–200199,
> que está en `logs_entrenamiento/prueba_final/`. El panel lo dice también, en
> pantalla, para que no haya ambigüedad si alguien fotografía la diapositiva.

## Cómo se lanza

```powershell
cd diffuser\godot

.\lanzar.ps1                       # condición A, en vivo, semilla 10000
.\lanzar.ps1 -Obs godot            # condición B: la política ve los píxeles de Godot
.\lanzar.ps1 -Semilla 200003       # otra condición inicial
.\lanzar.ps1 -Modo grabar          # episodio completo a grabaciones\
.\lanzar.ps1 -Modo reproducir      # reproduce lo grabado, sin GPU y sin servidor
```

El lanzador arranca el servidor de política, espera a que el puerto responda
—cargar V0 son unos treinta segundos— lanza Godot y se lleva por delante el
servidor al salir.

A mano, si hace falta:

```powershell
.venv_diffuser_infer\Scripts\python.exe diffuser\godot\servidor\servidor_politica.py --obs estado
godot --path diffuser\godot -- modo=vivo obs=estado seed=10000
```

## Los modos

| Modo | Para qué | ¿Servidor? | ¿Ventana? |
|---|---|---|---|
| `vivo` | La demostración. Bucle cerrado, la ventana se queda abierta al terminar. | sí | sí |
| `grabar` | Episodio completo a `grabaciones/<modo>_<obs>_seed<n>.json`, y cierra. | sí | solo en condición B |
| `reproducir` | Anima una trayectoria ya grabada. | **no** | sí |
| `observacion` | Vuelca a PNG lo que ve la cámara de observación. | sí, para el reinicio | sí |
| `comparar` | Guion de acciones fijo, para contrastar físicas. | no | no |
| `cobertura` | Cobertura en poses conocidas, para contrastar geometría. | no | no |

**`reproducir` es la red de seguridad de la defensa.** No necesita GPU, ni
CUDA, ni servidor, ni siquiera que el equipo tenga la tarjeta libre: relee el
JSON y recoloca los cuerpos. Conviene grabar los episodios elegidos el día antes
y ensayar el modo reproducir con el servidor apagado.

## El ritmo, y por qué es lento

V0 tarda una mediana de **1743,9 ms por llamada** en este equipo
(`memoria/datos/latencia_inferencia.csv`, magnitud `llamada`, lote 1). Cada
llamada devuelve ocho pasos de control, que a 10 Hz son 0,8 s de simulación.

El techo del bucle cerrado es, por tanto, **0,46 veces el tiempo real**, y no hay
forma de sortearlo salvo cambiar el muestreador o el número de pasos de difusión.
La demostración lo rotula en el panel en lugar de disimularlo: es la misma
limitación de latencia que sostiene el capítulo de resultados, solo que ahora se
ve.

Para el vídeo del beamer, el camino es grabar y reproducir a la velocidad que se
quiera con `-Velocidad`.

## Un episodio no se repite igual dos veces

La misma semilla, en la misma condición, con el mismo punto de control, da
trayectorias distintas entre ejecuciones. Medido: la semilla 200003 en la
condición A resolvió la tarea en 183 pasos con cobertura 0,9726 una vez y en 242
pasos con 0,9545 otra.

No es un fallo del port. El ruido de difusión sí está sembrado —el servidor fija
`torch.manual_seed` antes de cada llamada, con el mismo esquema que la pasada
preregistrada— pero las convoluciones en GPU no son deterministas bit a bit entre
ejecuciones, y una simulación con contactos amplifica cualquier diferencia
diminuta. Es la misma magnitud de ruido que el capítulo de resultados ya
atribuye a la evaluación por *rollout*.

Consecuencia práctica para la defensa: **graba y reproduce**. Si un episodio sale
bonito, queda guardado; volver a lanzarlo en vivo no lo devuelve.

## La escena

Raíz `Node`, con dos subárboles que no se mezclan:

- **La física es 2D.** `FisicaPushT` construye paredes, pieza y agente con
  `RigidBody2D`, `StaticBody2D` y `AnimatableBody2D`, en un mundo de 512×512 con
  el eje y hacia abajo, que es el convenio nativo de Godot 2D y también el de
  pygame: no hay ninguna conversión de coordenadas en ningún sitio.
- **El 3D es solo la vista.** `vista3d.gd` copia las poses 2D a nodos 3D en cada
  fotograma.

Se simula en 2D a propósito. Con física 3D de verdad —gravedad, fricción de
mesa— la pieza en T volcaría y V0 no tendría nada que hacer: la tarea dejaría de
ser la tarea que aprendió. Simulando en 2D, «otro motor de física» sigue siendo
cierto (Godot Physics 2D no es Chipmunk) y el 3D aporta lo que se le pide, que
es que se vea bien.

`Engine.physics_ticks_per_second = 100` no es decorativo: cada tick de física es
uno de los diez subpasos de `dt = 0,01` que el entorno original ejecuta por paso
de control. Diez ticks hacen un paso de control; ocho pasos de control agotan una
decisión de la política; 300 pasos de control terminan el episodio.

## El panel

Muestra la observación activa, el paso de control, la cobertura instantánea y la
máxima del episodio, la recompensa, el número de decisiones, la latencia de la
última y el factor de tiempo real que sale de ella. Y el aviso fijo de que la
cifra es ilustrativa.

La cobertura es el solape entre la pieza y la pose objetivo dividido por el área
del objetivo, 6300 px². Se termina con éxito por encima de 0,95, y la recompensa
es esa cobertura dividida por 0,95 y recortada a 1, igual que en el original.

## Qué mirar durante la demostración

- **Condición A frente a condición B.** En A la política ve exactamente los
  píxeles con los que se entrenó y solo cambia quién mueve los cuerpos. En B ve
  los de Godot. Que las dos funcionen dice algo; que una fallara diría más.
- **El agente da vueltas alrededor de la pieza.** Reposiciona antes de empujar
  en vez de empujar en línea recta: es la multimodalidad de la política de
  difusión, no una indecisión.
- **Los últimos puntos de cobertura son los caros.** Llegar a 0,90 es rápido;
  pasar de 0,95 exige ajustar el ángulo, y ahí es donde se agotan los 300 pasos.

## Una excepción, y hasta dónde llega

Desde el 31 de agosto de 2026 hay **una** medición que sí sale de este proyecto de
Godot, y está acotada por `memoria/preregistro_godot_paper.md`: V0 contra el punto
de control publicado por los autores, sobre las 50 condiciones `200000-200049`, en
las dos condiciones de observación y con dos realizaciones de ruido. Pregunta si la
paridad entre los dos —establecida en el simulador original— sobrevive al cambio de
motor de física y de renderizador.

Lo que la hace distinta de `perturbaciones.md` son tres cosas, y ninguna es
opcional: el protocolo está escrito y commiteado **antes** de ejecutar; **las
semillas no se eligen condicionando en el éxito**; y cada celda son 50 episodios,
no uno. Se ejecuta con `servidor/comparar_godot_paper.py` y se analiza con
`memoria/scripts/analisis_godot_paper.py`.

Todo lo demás que salga de esta carpeta sigue siendo ilustrativo. **Lo que muestre
el panel durante la defensa no es un resultado**, y `perturbaciones.md` y
`barrido_color.md` siguen siendo bitácoras.

## Qué sigue

Lo que queda abierto está en `indice.md`. Ampliar la excepción de arriba a una
medición general exigiría ejecutar el protocolo sobre las 200 condiciones del
bloque disjunto y con las cinco variantes, no con dos modelos y 50 condiciones.
Eso no forma parte del alcance de este trabajo: el contraste preregistrado está
dimensionado para detectar una separación de unas 0,105 y nada más fino.

## Referencias

- Módulo 06: el puente, el port de la física y sus trampas.
- Preregistro de la prueba final: `memoria/preregistro_prueba_final.md`.
- Preregistro del contraste en Godot: `memoria/preregistro_godot_paper.md`.
- Latencia medida: `memoria/datos/latencia_inferencia.csv`.
