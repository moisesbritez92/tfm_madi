# 05. Fisica, Colisiones y Movimiento

## Objetivo del modulo

Entender que tipo de cuerpo usar en cada caso y construir el primer comportamiento 3D interactivo sin mezclar mal fisica, input y transformaciones manuales.

## La pregunta clave en Godot 3D

Antes de escribir una sola linea de movimiento, debes responder esto: que tipo de cuerpo representa cada objeto de la escena.

Si eliges mal el tipo de nodo, el proyecto se vuelve inconsistente muy rapido.

## Los cuerpos que mas te interesan en esta fase

| Nodo | Cuando usarlo |
|---|---|
| `StaticBody3D` | Suelo, paredes y elementos del entorno que no deben ser movidos por la fisica. |
| `CharacterBody3D` | Personajes o controladores movidos por script con respuesta de colision. |
| `RigidBody3D` | Objetos fisicos simulados que deben caer, rebotar o ser empujados. |
| `Area3D` | Zonas de deteccion o triggers, no cuerpos para movimiento normal. |

Regla corta:

- entorno fijo: `StaticBody3D`,
- personaje controlado: `CharacterBody3D`,
- objeto dinamico: `RigidBody3D`,
- deteccion: `Area3D`.

## CollisionShape3D no es opcional

Un cuerpo fisico sin forma de colision esta incompleto.

Necesitas un `CollisionShape3D` hijo para definir el volumen que usa la fisica. Y aqui hay una advertencia importante:

- no escales la forma de colision usando la propiedad de escala del nodo,
- ajusta su tamano con el recurso de forma o con sus manejadores.

Escalar colisiones de forma descuidada es una forma muy comun de introducir errores raros.

## Collision layers y masks

En proyectos pequenos es facil ignorarlo al principio. En cuanto el entorno crece, deja de ser opcional.

- `collision_layer` indica en que capa existe el objeto.
- `collision_mask` indica que capas consulta para detectar colisiones.

Pensar en capas desde el inicio te ahorra reestructuras posteriores.

Propuesta simple para este laboratorio:

- capa 1: entorno,
- capa 2: jugador,
- capa 3: objetos dinamicos,
- capa 4: triggers.

## CharacterBody3D: personaje controlado por script

`CharacterBody3D` no esta gobernado por la fisica como un objeto libre. Se mueve con codigo, pero conserva deteccion de colisiones y una API muy util para suelos, paredes y pendientes.

La herramienta central aqui es `move_and_slide()`.

Dos reglas que conviene fijar ya:

1. Usa `move_and_slide()` dentro de `_physics_process()`.
2. No multipliques por `delta` la velocidad horizontal que pasas a `velocity` antes de llamar a `move_and_slide()`.

La gravedad si se integra con `delta` porque es una aceleracion acumulada.

## Script base de movimiento 3D

Este ejemplo es una primera base razonable para un personaje controlable.

```gdscript
extends CharacterBody3D

@export var move_speed: float = 5.0
@export var jump_velocity: float = 4.5
var gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")

func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity.y -= gravity * delta

	var input_vector := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	var direction := Vector3(input_vector.x, 0.0, input_vector.y)

	if direction != Vector3.ZERO:
		direction = direction.normalized()
		velocity.x = direction.x * move_speed
		velocity.z = direction.z * move_speed
	else:
		velocity.x = move_toward(velocity.x, 0.0, move_speed)
		velocity.z = move_toward(velocity.z, 0.0, move_speed)

	if Input.is_action_just_pressed("jump") and is_on_floor():
		velocity.y = jump_velocity

	move_and_slide()
```

Que enseña este script:

- gravedad separada del movimiento horizontal,
- uso correcto de `_physics_process`,
- uso del Input Map,
- salto condicionado por contacto con el suelo,
- y movimiento desacoplado de cambios directos de posicion.

## RigidBody3D: objetos simulados

Usa `RigidBody3D` cuando quieres que el motor de fisica resuelva el movimiento por ti.

Ejemplos tipicos:

- cajas que caen,
- esferas que ruedan,
- objetos empujables,
- elementos que reaccionan a impulsos.

La regla mas importante aqui es esta:

- no intentes controlar un `RigidBody3D` moviendolo a mano cada frame como si fuera un `Node3D` normal.

Si necesitas intervenir, usa fuerzas, impulsos o integracion especifica.

## Fuerza frente a impulso

Conviene distinguirlos pronto:

- una fuerza se aplica a lo largo del tiempo,
- un impulso representa un golpe o evento instantaneo.

Si aplicas un impulso cada frame, estaras modelando algo incorrecto y dependiente del framerate.

## Ejemplo breve con impulso

Supongamos que quieres lanzar una caja una sola vez:

```gdscript
extends RigidBody3D

@export var impulse_strength: float = 8.0

func _ready() -> void:
	apply_central_impulse(Vector3(0.0, 0.0, -impulse_strength))
```

Esto sirve como demostracion de que un `RigidBody3D` puede reaccionar a la fisica sin que escribas un controlador completo.

## Contact monitoring y senales fisicas

Si quieres que un `RigidBody3D` informe de colisiones mediante senales, necesitas configurar dos cosas:

- `contact_monitor = true`
- `max_contacts_reported > 0`

Sin eso, muchas comprobaciones de contacto devolveran resultados vacios.

## Area3D: detectar sin bloquear

`Area3D` es util cuando necesitas detectar presencia o activar eventos sin tratar el objeto como un cuerpo dinamico normal.

Ejemplos:

- zona de recogida,
- trigger de puerta,
- area de peligro,
- volumen de activacion de un evento.

Es el nodo correcto cuando tu pregunta no es "como se mueve esto" sino "como detecto que algo ha entrado aqui".

## Colisiones y lectura del entorno

Antes de pensar en interacciones avanzadas, prueba este pequeño laboratorio:

1. Mantener `Ground` como `StaticBody3D`.
2. Crear un `Player` con `CharacterBody3D` y `CollisionShape3D`.
3. Crear una caja con `RigidBody3D`, `CollisionShape3D` y `MeshInstance3D`.
4. Colocar la caja frente al jugador.
5. Mover el jugador y comprobar que empuja o colisiona con la caja y con el suelo.

Este escenario ya te enseña casi todo lo esencial del modulo.

## Sobre pendientes, suelo y techo

`CharacterBody3D` incluye funciones muy utiles:

- `is_on_floor()`
- `is_on_wall()`
- `is_on_ceiling()`

Ademas, propiedades como `up_direction`, `floor_snap_length` o `floor_max_angle` controlan como interpreta el terreno.

No necesitas afinarlas en profundidad todavia, pero si necesitas comportamiento estable en suelos inclinados, ese es el lugar correcto donde mirar.

## Error tipico con `move_and_slide()`

El error mas comun al empezar es este:

- calcular una velocidad,
- multiplicarla por `delta`,
- asignarla a `velocity`,
- y luego llamar a `move_and_slide()`.

Eso mezcla velocidad con desplazamiento. `velocity` debe seguir siendo velocidad, no un desplazamiento ya integrado.

## Primera configuracion recomendada de capas

Si quieres empezar con una base ordenada:

1. Nombra capas de fisica en la configuracion del proyecto.
2. Pon el suelo y paredes en la capa `entorno`.
3. Pon el personaje en la capa `jugador`.
4. Pon cajas empujables en `dinamicos`.
5. Pon zonas de activacion en `triggers`.

Aunque el prototipo sea pequeno, esta disciplina ayuda mucho cuando anadas sensores, zonas o NPCs.

## Ejercicio guiado del modulo

1. Crea un `Ground` con `StaticBody3D`, colision y malla.
2. Crea un `Player` con `CharacterBody3D`, malla simple y `CollisionShape3D`.
3. Asigna al jugador el script de movimiento basico.
4. Crea una caja con `RigidBody3D` y colision.
5. Ejecuta la escena y confirma:
   - el jugador no atraviesa el suelo,
   - puede desplazarse,
   - puede saltar,
   - y la caja responde como objeto fisico.

## Errores tipicos

- Mover un `RigidBody3D` actualizando su transformacion a mano cada frame.
- Usar `Node3D` cuando realmente necesitabas `CharacterBody3D`.
- Olvidar `CollisionShape3D`.
- Escalar colisiones con `scale` en lugar de ajustar la forma.
- No distinguir capa y mascara.

## Hito del modulo

Al terminar este bloque deberias tener un laboratorio 3D donde un personaje controlado por script colisiona con el suelo y con al menos un objeto fisico dinamico.

## Que sigue

Con esto ya tienes una base valida para empezar a importar assets, montar un entorno mas realista y pensar en navegacion o interacciones mas ricas.

## Referencias oficiales

- Physics introduction: https://docs.godotengine.org/en/stable/tutorials/physics/physics_introduction.html
- CharacterBody3D: https://docs.godotengine.org/en/stable/classes/class_characterbody3d.html
- RigidBody3D: https://docs.godotengine.org/en/stable/classes/class_rigidbody3d.html