# 03. GDScript, Input y Senales

## Objetivo del modulo

Empezar a programar en Godot sin salir del flujo normal del editor. Al terminar deberias poder adjuntar un script a un nodo, ejecutar logica en tiempo real, capturar input y conectar eventos mediante senales.

## Por que GDScript aqui

Para esta ruta de aprendizaje conviene empezar con GDScript por tres razones:

- reduce friccion,
- esta muy integrado con el editor,
- y te deja concentrarte en la estructura del entorno antes de abrir debates sobre arquitectura de lenguaje.

Mas adelante podrias usar C#, pero ahora mismo no es la prioridad.

## Donde vive el codigo en Godot

Un script suele adjuntarse a un nodo. Eso significa que el codigo opera desde el contexto de ese nodo y puede acceder a sus propiedades y a parte de la jerarquia circundante.

La consecuencia practica es importante: no sueles programar una aplicacion entera en un unico archivo, sino pequenos comportamientos ligados a nodos o escenas concretas.

## Ciclo de vida minimo que debes reconocer

Hay tres callbacks que aparecen muy pronto:

| Metodo | Cuando usarlo |
|---|---|
| `_ready()` | Inicializacion cuando el nodo ya esta listo en la escena. |
| `_process(delta)` | Logica por frame, util para actualizacion no fisica. |
| `_physics_process(delta)` | Logica ligada al bucle de fisica. |

Regla simple:

- si el comportamiento es visual o general, piensa en `_process`,
- si afecta movimiento o fisica, piensa en `_physics_process`.

## Primer script util

Adjunta este script a un `Node3D` o a un `MeshInstance3D` para comprobar que puedes ejecutar codigo y modificar una propiedad visible.

```gdscript
extends Node3D

@export var rotation_speed: float = 1.5

func _process(delta: float) -> void:
	rotate_y(rotation_speed * delta)
```

Que deberias observar:

- el nodo rota de manera continua,
- el valor `rotation_speed` aparece en el Inspector,
- puedes cambiarlo sin tocar el script.

Esto enseña una idea central de Godot: editor y script trabajan juntos.

## Sintaxis minima de GDScript

No hace falta cubrir todo el lenguaje ahora. Te basta con dominar:

- `extends` para indicar el tipo base.
- variables tipadas.
- funciones.
- condicionales e iteracion basica.
- exportacion de variables al Inspector con `@export`.

Ejemplo corto:

```gdscript
extends Node

var score: int = 0

func add_point() -> void:
	score += 1
	print(score)
```

## Input: que es y como se organiza

No conviene leer teclas sueltas por nombre duro desde el primer dia. Godot trabaja mejor con acciones definidas en el `Input Map` del proyecto.

Ejemplo de acciones utiles desde el principio:

- `move_left`
- `move_right`
- `move_forward`
- `move_back`
- `jump`

La ventaja es que luego puedes reasignar teclas o combinar teclado y mando sin reescribir toda la logica.

## Ejemplo basico de input

El siguiente script desplaza un nodo usando acciones definidas en el Input Map. No sustituye al movimiento fisico definitivo de un personaje, pero es un puente claro entre input y transformacion.

```gdscript
extends Node3D

@export var speed: float = 3.0

func _process(delta: float) -> void:
	var direction := Vector3.ZERO

	if Input.is_action_pressed("move_left"):
		direction.x -= 1.0
	if Input.is_action_pressed("move_right"):
		direction.x += 1.0
	if Input.is_action_pressed("move_forward"):
		direction.z -= 1.0
	if Input.is_action_pressed("move_back"):
		direction.z += 1.0

	if direction != Vector3.ZERO:
		translate(direction.normalized() * speed * delta)
```

Este ejemplo sirve para:

- confirmar que el Input Map funciona,
- observar movimiento en escena,
- y entender por que luego necesitaremos `CharacterBody3D` cuando entremos en fisica.

## Como configurar el Input Map

Pasos recomendados:

1. Abrir `Project Settings`.
2. Ir a `Input Map`.
3. Crear las acciones `move_left`, `move_right`, `move_forward` y `move_back`.
4. Asignar teclas como `A`, `D`, `W`, `S` o las flechas.

No saltes este paso. Si el input falla, muchas veces el error no esta en el script sino en la accion no definida.

## Senales: el mecanismo de eventos de Godot

Las senales permiten que un nodo avise a otro de que algo ha ocurrido sin acoplar ambos de forma rigida.

Piensalas asi:

- un emisor lanza un evento,
- un receptor escucha,
- y responde con una funcion conectada.

Este modelo es clave para mantener escenas modulares.

## Ejemplo simple con `Timer`

Anade un `Timer` como hijo de un nodo cualquiera y conectale la senal `timeout`.

Script posible en el nodo padre:

```gdscript
extends Node

func _ready() -> void:
	$Timer.start()

func _on_timer_timeout() -> void:
	print("Se disparo el temporizador")
```

Este ejemplo te enseña dos cosas:

- una senal puede lanzar logica sin que tu la invoques manualmente por frame,
- y Godot genera flujos de trabajo muy basados en eventos.

## Cuando usar senales y cuando no

Usa senales cuando:

- un nodo debe reaccionar a un evento externo,
- quieres desacoplar emisor y receptor,
- o necesitas comunicar escenas sin dependencia fuerte.

No uses senales para todo. Si un comportamiento es puramente interno y lineal, una llamada directa puede ser suficiente.

## Ejercicio guiado del modulo

1. Crea un nodo 3D visible.
2. Adjuntale el script de rotacion para confirmar que entiendes `_process`.
3. Define acciones en el Input Map.
4. Sustituye la rotacion por el script de movimiento y mueve el nodo en la escena.
5. Anade un `Timer` y conecta la senal `timeout`.
6. Verifica que la consola imprime un mensaje al dispararse la senal.

## Errores tipicos

- Escribir scripts sobre nodos del tipo incorrecto.
- Usar `_process` para todo sin pensar si deberia ir en `_physics_process`.
- Olvidar crear acciones en el Input Map.
- Confiar en nombres de nodos que luego cambian y rompen rutas como `$Timer`.
- Intentar resolver con codigo un problema que era de jerarquia o conexion de senales.

## Hito del modulo

Al terminar deberias poder:

- adjuntar un script y ejecutarlo,
- exponer variables al Inspector,
- capturar input mediante acciones,
- y conectar una senal sencilla.

## Que sigue

El siguiente bloque debe llevar esto a una escena 3D mas seria: camaras, luces, referencia espacial y una primera navegacion util dentro del entorno.

## Referencias oficiales

- Scripting languages: https://docs.godotengine.org/en/stable/getting_started/step_by_step/scripting_languages.html
- First script: https://docs.godotengine.org/en/stable/getting_started/step_by_step/scripting_first_script.html
- Player input: https://docs.godotengine.org/en/stable/getting_started/step_by_step/scripting_player_input.html
- Signals: https://docs.godotengine.org/en/stable/getting_started/step_by_step/signals.html