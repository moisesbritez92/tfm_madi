class_name FisicaPushT
extends Node2D

## El entorno Push-T sobre Godot Physics 2D.
##
## Reproduce la geometria, las masas y el controlador del entorno original; lo
## que no reproduce, ni pretende, es el solucionador. Chipmunk y Godot Physics 2D
## resuelven los contactos de forma distinta, y esa diferencia es justamente lo
## que la demostracion ensena.
##
## El mundo es de 512 x 512 con el eje y hacia abajo, que es el convenio nativo
## de Godot 2D y tambien el de pygame, asi que no hay ninguna conversion de
## coordenadas en ningun sitio.

signal paso_control_completado

var bloque: BloqueT
var agente: AnimatableBody2D

## Objetivo posicional del controlador, en pixeles. Es la accion que devuelve la
## politica: una posicion absoluta, no un incremento.
var objetivo_agente: Vector2 = Vector2(256.0, 400.0)
## Velocidad del agente. Persiste entre pasos de control; el original tampoco la
## reinicia (`pusht_env.py:109-122`).
var velocidad_agente: Vector2 = Vector2.ZERO

var _subpasos: int = 0
var _corriendo: bool = false


func _ready() -> void:
	_construir_paredes()
	_construir_bloque()
	_construir_agente()
	set_physics_process(true)


# --- construccion -----------------------------------------------------------

func _construir_paredes() -> void:
	# Los cuatro segmentos de Chipmunk tienen radio 2, asi que la cara interior
	# queda a 7 y a 504. Aqui se usan rectangulos con esa misma cara interior;
	# la unica diferencia son las esquinas, que en Chipmunk van redondeadas.
	var lo := PushTConst.PARED_MIN
	var hi := PushTConst.PARED_MAX
	var r := PushTConst.PARED_RADIO
	var largo := hi - lo + 2.0 * r
	var paredes := {
		"izquierda": [Vector2(lo, (lo + hi) * 0.5), Vector2(2.0 * r, largo)],
		"derecha": [Vector2(hi, (lo + hi) * 0.5), Vector2(2.0 * r, largo)],
		"arriba": [Vector2((lo + hi) * 0.5, lo), Vector2(largo, 2.0 * r)],
		"abajo": [Vector2((lo + hi) * 0.5, hi), Vector2(largo, 2.0 * r)],
	}
	for nombre in paredes:
		var cuerpo := StaticBody2D.new()
		cuerpo.name = "Pared_" + str(nombre)
		cuerpo.position = paredes[nombre][0]
		cuerpo.physics_material_override = _material()
		var forma := CollisionShape2D.new()
		var rect := RectangleShape2D.new()
		rect.size = paredes[nombre][1]
		forma.shape = rect
		cuerpo.add_child(forma)
		add_child(cuerpo)


func _construir_bloque() -> void:
	bloque = BloqueT.new()
	bloque.name = "BloqueT"
	bloque.mass = PushTConst.MASA
	bloque.inertia = PushTConst.INERCIA
	bloque.center_of_mass_mode = RigidBody2D.CENTER_OF_MASS_MODE_CUSTOM
	bloque.center_of_mass = PushTConst.CENTRO_GRAVEDAD
	bloque.gravity_scale = 0.0
	bloque.linear_damp_mode = RigidBody2D.DAMP_MODE_REPLACE
	bloque.linear_damp = 0.0
	bloque.angular_damp_mode = RigidBody2D.DAMP_MODE_REPLACE
	bloque.angular_damp = 0.0
	# Dormirse congelaria la pieza en medio de un empujon: con la velocidad
	# anulada en cada subpaso, el motor la ve quieta casi siempre.
	bloque.can_sleep = false
	bloque.physics_material_override = _material()
	bloque.contact_monitor = true
	bloque.max_contacts_reported = 8
	for vertices in [PushTConst.VERTICES_BARRA, PushTConst.VERTICES_VASTAGO]:
		var forma := CollisionShape2D.new()
		var convexa := ConvexPolygonShape2D.new()
		convexa.points = PackedVector2Array(vertices)
		forma.shape = convexa
		bloque.add_child(forma)
	add_child(bloque)


func _construir_agente() -> void:
	agente = AnimatableBody2D.new()
	agente.name = "Agente"
	# El agente del original es cinematico: empuja y nada lo desvia.
	# sync_to_physics hace que Godot deduzca su velocidad del movimiento del
	# transform y la use al resolver los contactos, que es lo que hace Chipmunk.
	agente.sync_to_physics = true
	agente.physics_material_override = _material()
	var forma := CollisionShape2D.new()
	var circulo := CircleShape2D.new()
	circulo.radius = PushTConst.RADIO_AGENTE
	forma.shape = circulo
	agente.add_child(forma)
	add_child(agente)


func _material() -> PhysicsMaterial:
	var material := PhysicsMaterial.new()
	material.friction = PushTConst.FRICCION
	material.bounce = PushTConst.REBOTE
	return material


# --- estado -----------------------------------------------------------------

## Situa agente y pieza a partir del vector de cinco componentes del servidor.
func colocar(estado: Array) -> void:
	var pos_agente := Vector2(float(estado[0]), float(estado[1]))
	var pos_bloque := Vector2(float(estado[2]), float(estado[3]))
	var angulo := float(estado[4])

	agente.global_position = pos_agente
	velocidad_agente = Vector2.ZERO
	objetivo_agente = pos_agente

	# La rotacion de un RigidBody2D es alrededor de su centro de masa, igual que
	# en Chipmunk, asi que el angulo va primero y la posicion despues: al reves,
	# fijar el angulo arrastraria el origen fuera de pos_bloque.
	var transformada := Transform2D(angulo, Vector2.ZERO)
	transformada.origin = pos_bloque
	PhysicsServer2D.body_set_state(
		bloque.get_rid(), PhysicsServer2D.BODY_STATE_TRANSFORM, transformada
	)
	PhysicsServer2D.body_set_state(
		bloque.get_rid(), PhysicsServer2D.BODY_STATE_LINEAR_VELOCITY, Vector2.ZERO
	)
	PhysicsServer2D.body_set_state(
		bloque.get_rid(), PhysicsServer2D.BODY_STATE_ANGULAR_VELOCITY, 0.0
	)
	bloque.global_transform = transformada

	_subpasos = 0


## Las cinco componentes que el resto del sistema entiende por estado.
func estado() -> Array:
	return [
		agente.global_position.x,
		agente.global_position.y,
		bloque.global_position.x,
		bloque.global_position.y,
		bloque.global_rotation,
	]


func cobertura() -> float:
	return Cobertura.de_pose(bloque.global_position, bloque.global_rotation)


func recompensa() -> float:
	return PushTConst.recompensa(cobertura())


func terminado() -> bool:
	return cobertura() > PushTConst.UMBRAL_EXITO


# --- avance -----------------------------------------------------------------

## Deja correr la fisica. Cada tick de `_physics_process` es un subpaso de
## dt = 0,01, y diez de ellos hacen un paso de control.
func arrancar() -> void:
	_corriendo = true
	bloque.freeze = false


func parar() -> void:
	_corriendo = false
	# Congelar mientras se espera al planificador evita que un solape residual
	# siga empujando la pieza durante los 1,7 s que tarda una decision.
	bloque.freeze = true


func _physics_process(_delta: float) -> void:
	if not _corriendo:
		return

	# Control PD integrado explicitamente, igual que el original: la aceleracion
	# se calcula contra la posicion actual, se integra a velocidad y la posicion
	# la mueve el propio motor. `_physics_process` corre antes de que el paso de
	# fisica se resuelva, asi que el empujon de este subpaso ya lleva la
	# velocidad nueva.
	var pos := agente.global_position
	var aceleracion := PushTConst.K_P * (objetivo_agente - pos) \
		- PushTConst.K_V * velocidad_agente
	velocidad_agente += aceleracion * PushTConst.DT
	agente.global_position = pos + velocidad_agente * PushTConst.DT

	_subpasos += 1
	if _subpasos >= PushTConst.SUBPASOS:
		_subpasos = 0
		paso_control_completado.emit()
