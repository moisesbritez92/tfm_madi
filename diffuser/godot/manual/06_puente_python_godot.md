# 06. El puente entre Godot y la política

## Objetivo del módulo

Explicar cómo la política V0 del TFM acaba conduciendo una simulación de Push-T
escrita en Godot, y dejar por escrito las trampas del port, que son cinco y
ninguna es evidente.

Este módulo ya no es material de aprendizaje: describe el código que vive en
`escenas/`, `scripts/` y `servidor/`, justo encima de este manual.

## El reparto de responsabilidades

Godot es el entorno. Python es la política y nada más.

```
Godot 4 (diffuser/godot/)                 Python (.venv_diffuser_infer)
┌────────────────────────────┐   TCP     ┌──────────────────────────────┐
│ física 2D (RigidBody2D)    │  JSON     │ servidor_politica.py         │
│ vista 3D cenital           │ ────────► │  · V0 (load_policy_bundle)   │
│ SubViewport obs 512 → 96   │ ◄──────── │  · rasterizador (condición A)│
│ cobertura, HUD, grabador   │           │  · reset(seed) con pymunk    │
└────────────────────────────┘           └──────────────────────────────┘
```

El protocolo es JSON por líneas sobre TCP en `127.0.0.1`. Cuatro comandos:

| Comando | Entrada | Salida |
|---|---|---|
| `hola` | — | variante, punto de control, dispositivo, modo de observación |
| `reset` | `seed` | `estado0`, cinco componentes |
| `act` | `agent_pos` (2×2) y, según el modo, `estado` (2×5) o `imagen` (2 PNG en base64) | `accion` de 8×2 en píxeles 0–512, y la latencia en ms |
| `cobertura` | `estado` | cobertura de referencia con shapely |

No hay hilos en el lado de Godot. La petición se envía y la respuesta se sondea
en `_process`, así que la ventana sigue viva mientras V0 piensa. La simulación
sí se detiene, porque no tiene acciones que ejecutar.

## Qué punto de control, y qué escena

El servidor no está atado a V0. `--variante {v0,v1,v2,v3,v4}` elige cuál de los
cinco puntos de control congelados se pone en el bucle. Solo
`v0_inference_utils` exporta `load_policy_bundle`; los otros cuatro módulos son
envoltorios que redefinen tres constantes, así que el cargador es siempre el de
V0 y de la variante salen el punto de control y su carpeta de artefactos. Sirve
igual para V3 y V4: el cargador ya fija `rgb_model.pretrained = False` cuando el
codificador vive en `pretrained_encoders`, de modo que no se descarga nada.

Godot **no** recibe la variante por línea de órdenes: la lee de la respuesta a
`hola`. Un parámetro menos que puede desincronizarse con el servidor que de
verdad tiene los pesos cargados.

Del lado de la escena, `perturbacion={ninguna,t_roja,sombras}` cambia lo que
Godot dibuja, y solo tiene efecto en la condición B. Los resultados de ese
experimento están en `perturbaciones.md`, en la raíz del proyecto, con la
advertencia que les corresponde: son una bitácora, no una medición.

## Las dos condiciones

La demostración distingue qué parte del entorno original se ha sustituido:

- **Condición A, `--obs estado`.** Godot manda el estado y el fotograma de 96×96
  lo dibuja `rasterizador_pusht.py` con el mismo código que generó las
  demostraciones. Lo único que ha cambiado es el motor de física.
- **Condición B, `--obs godot`.** La observación sale del `SubViewport`
  ortográfico cenital de Godot. Han cambiado la física y los píxeles.

Es una bandera del servidor. El código de Godot es el mismo en las dos.

## Por qué el estado inicial lo sortea Python

`PushTEnv.reset` usa `np.random.RandomState`, es decir el Mersenne Twister
antiguo de numpy, y además `randn()*2*pi - pi` para el ángulo, que casi con
seguridad era un fallo pero está en todos los datos publicados. Reimplementar
eso en GDScript sería trabajo puro con riesgo puro.

En vez de eso, `reset` instancia el `PushTEnv` de verdad, lo siembra y devuelve
las cinco componentes **tal como quedan** después de la asignación en orden
`legacy` y del `space.step(0.01)` que `_set_state` ejecuta al final. Godot
recibe números, no una receta.

Está comprobado: `cliente_prueba.py` contrasta el `estado0` que llega por el
socket con el que produce `PushTEnv.seed(s); reset()` para cinco semillas, y la
diferencia es exactamente cero.

## Trampa 1: la amortiguación total

`space.damping = 0` no es un rozamiento suave. En Chipmunk la amortiguación es
la fracción de velocidad que se conserva **por segundo**, de modo que
`v *= 0 ^ dt` anula la velocidad al principio de cada subpaso. La pieza solo se
mueve por el impulso de colisión generado dentro de ese mismo subpaso: nada
desliza por inercia, y la simulación es cuasi-estática.

Un `RigidBody2D` normal de Godot conserva la inercia, sigue viajando después de
que el agente lo suelte, y la tarea deja de ser la tarea. La solución está en
`scripts/bloque_t.gd`:

```gdscript
func _integrate_forces(estado: PhysicsDirectBodyState2D) -> void:
	estado.linear_velocity = Vector2.ZERO
	estado.angular_velocity = 0.0
```

Anularla ahí da el mismo resultado neto que anularla al principio del paso: la
velocidad con la que se integra la posición es, en los dos casos, solo la que
los impulsos de ese paso han producido.

Es el detalle que más fácil se pasa por alto de todo el port.

## Trampa 2: el centro de gravedad no está en el origen

`body.center_of_gravity = (0, 45)`, y `body.position` es el origen del cuerpo,
no su centro de masa. La consecuencia práctica: **fijar el ángulo desplaza la
posición**, porque la rotación es alrededor del centro de gravedad.

Por eso, para reproducir una pose conocida, el ángulo va primero y la posición
después. El orden contrario es el que usa `_set_state` en modo `legacy`, pero
eso pertenece a cómo se sortea un estado en el reinicio, no a cómo se reproduce
un estado que ya se conoce. Confundir las dos cosas cuesta una tarde.

Hay más números heredados que conviene no «corregir»:

- **El momento de inercia es 3000, no 7875.** `add_tee` calcula el segundo
  momento con los vértices del primero. Es un fallo del original, y es el número
  con el que se generaron las demostraciones.
- **`body.friction = 1` no hace nada.** En Chipmunk la fricción vive en la
  forma, y ahí nunca se asigna: los contactos van sin rozamiento y sin rebote.

## Trampa 3: el aliasing forma parte del conjunto de datos

El original dibuja a 512×512 **sin antialiasing** y baja a 96×96 con
`cv2.resize` bilineal. Ese aliasing característico está en cada imagen con la
que V0 se entrenó. Si en Godot se renderiza a 96 con MSAA, la distribución de
imágenes es otra.

Lo que hace `vista3d.gd`: `SubViewport` de 512×512, MSAA desactivado, sin TAA,
sin debanding, y reducción a 96 con `Image.INTERPOLATE_BILINEAR`.

Con eso, y con la corrección de medio píxel de la cámara ortográfica —Godot
muestrea en el centro del píxel, pygame rellena el píxel entero—, el 97 % de los
píxeles coincide con el renderizador original dentro de una tolerancia de 8
niveles, con una diferencia media de 1,0 sobre 255.

## Trampa 4: los puntos de colisión

`space.debug_draw` pinta los puntos de contacto del último paso. El rasterizador
coloca los cuerpos y **no** ejecuta física, así que arrastraría los contactos del
reinicio aleatorio con el que se construyó: unos puntos rojos que no tienen nada
que ver con el estado que se está dibujando, y que además cambian entre
ejecuciones.

`RasterizadorPushT._vaciar_contactos` separa los cuerpos y da dos pasos para que
no quede ningún árbitro vivo. Como después no se ejecuta física nunca, tampoco
vuelve a crearse ninguno.

Con la corrección, el fotograma del rasterizador es **idéntico píxel a píxel** al
que produce el entorno mientras no haya contactos, y difiere en uno o dos
píxeles de 9216 cuando el entorno sí los tiene y los pinta. Sin ella, arrastra
manchas rojas que cambian de una ejecución a otra. Se prefiere el sesgo
constante y pequeño al ruido aleatorio.

## Trampa 5: capturar dos poses distintas en el mismo fotograma

La política mira dos pasos de observación, no uno. En la condición B eso obliga
a dibujar dos veces: la pose de hace un paso de control y la actual. Se puede
porque la física está detenida mientras se pide la acción, así que basta con
mover los nodos a cada pose histórica y capturar.

El problema es el orden dentro del fotograma. Los `_process` de todos los nodos
corren **antes** del dibujado, y `vista3d._process` copia las poses de la física
a los nodos 3D. Así que la secuencia real era: coloco la pose histórica → llega
el fotograma → `_process` la sobrescribe con la actual → se dibuja → recojo la
imagen. Las dos observaciones salían idénticas.

Y salían idénticas **sin ningún síntoma**: la política seguía funcionando, solo
que con un historial congelado y algo peor. En la semilla 10000, 0,9159 en lugar
de 0,9453.

La solución es un cerrojo, `_congelado`, que hace que `_process` no toque nada
mientras dura la captura. Y como esta clase de fallo es silenciosa, el servidor
lleva un aviso: si las dos imágenes llegan idénticas byte a byte mientras los dos
estados difieren, lo dice por consola una vez.

Moraleja general para Godot: cuando algo tiene que ser cierto **en el momento del
dibujado**, no basta con dejarlo escrito antes; hay que impedir que nadie lo
reescriba entre medias.

## Las dos camaras del mismo mundo

Cada elemento de la escena existe dos veces, con la misma transformada y en dos
capas de visibilidad distintas:

- capa 1, la versión iluminada, con material `StandardMaterial3D` normal, que ve
  la cámara en perspectiva de la demostración;
- capa 2, la versión plana, `SHADING_MODE_UNSHADED` en los colores exactos del
  entrenamiento, que ve la cámara ortográfica de observación.

Sin esa separación, la condición B fracasaría por una sombra o por un brillo
especular, y no se aprendería nada de ella. Las cámaras se reparten el mundo con
`cull_mask`, y cada una lleva su propio `Environment`: fondo blanco y sin luz
ambiente para la de observación, fondo gris y ambiente suave para la otra.

## Verificación

Cuatro comprobaciones, en orden, cada una más cara que la anterior:

```powershell
# 1 y 2: el servidor y los estados iniciales, sin Godot de por medio
.venv_diffuser_infer\Scripts\python.exe diffuser\godot\servidor\cliente_prueba.py

# 3: la geometría, Geometry2D contra shapely
godot --headless --path diffuser\godot -- modo=cobertura
.venv_diffuser_infer\Scripts\python.exe diffuser\godot\servidor\verificar_port.py cobertura

# 4: las dos físicas, con el mismo guion de acciones
godot --headless --path diffuser\godot -- modo=comparar
.venv_diffuser_infer\Scripts\python.exe diffuser\godot\servidor\verificar_port.py fisica

# 6: la observación de Godot al lado de la del renderizador original
godot --path diffuser\godot -- modo=observacion seed=10000
.venv_diffuser_infer\Scripts\python.exe diffuser\godot\servidor\comparar_observacion.py
```

Lo que salió al escribir esto, y sirve de referencia para detectar una
regresión:

| Comprobación | Resultado |
|---|---|
| Estados iniciales, semillas 100000–100004 | diferencia 0, exacta |
| Cobertura, nueve poses | peor diferencia 3,1e-07 frente a shapely |
| Física, 250 pasos de control con el mismo guion | salto de 2,3 px en el primer contacto; separación final 0,30 px y 0,005 rad |
| Observación de Godot contra el original | 2,9 % de píxeles con diferencia mayor que 8; media 1,0 |

Y para elegir los episodios de la defensa, `servidor/resumen_grabaciones.py`
pone las grabaciones al lado de la puntuación preregistrada de cada semilla.

Sobre la física: las dos trayectorias **no** tienen por qué coincidir, porque
Chipmunk y Godot Physics 2D resuelven los contactos de forma distinta, y esa
diferencia es precisamente lo que la demostración enseña. Lo que se vigila es
que la separación sea gradual. Un salto grande en el primer contacto significa
que falta anular la velocidad en cada subpaso, o que el centro de masa no es
`(0, 45)`.

## Qué sigue

El módulo 07 describe la escena, los modos de ejecución y cómo se lanza todo.

## Referencias

- Entorno original: `diffuser/repo/diffusion_policy/diffusion_policy/env/pusht/`
- Renderizador: `pymunk_override.py` en esa misma carpeta
- Física 2D de Godot: https://docs.godotengine.org/en/stable/tutorials/physics/index.html
- `SubViewport`: https://docs.godotengine.org/en/stable/tutorials/rendering/viewports.html
