extends Node

## Orquestador de la demostracion.
##
## Construye la fisica, decide el modo y cierra el bucle con el servidor de
## politica. Los modos de verificacion (`comparar`, `cobertura`) corren sin
## servidor y sin ventana, y son los que se ejecutan primero: fallan barato.
##
## Argumentos, despues de `--`:
##   modo=vivo|grabar|reproducir|comparar|cobertura|observacion
##   seed=10000            condicion inicial
##   obs=estado|godot      solo informativo; quien decide es el servidor
##   host=127.0.0.1 puerto=5555
##   pasos=300             tope de pasos de control
##   velocidad=1.0         multiplicador de tiempo en modo grabar
##   salida=ruta.json      destino de comparar/cobertura/grabar
##
## Ninguna cifra que salga de aqui es un resultado del TFM.

const MODOS_SIN_SERVIDOR := ["comparar", "cobertura"]

var fisica: FisicaPushT
var cliente: ClientePolitica
var vista: Node3D
var hud: CanvasLayer

var modo: String = "vivo"
var seed_episodio: int = 10000
var modo_obs: String = "estado"
var host: String = "127.0.0.1"
var puerto: int = 5555
var max_pasos: int = PushTConst.MAX_PASOS_CONTROL
var velocidad: float = 1.0
var ruta_salida: String = ""

# --- estado del episodio ----------------------------------------------------

var _historia: Array = []          ## ultimos N_OBS_STEPS estados
var _acciones: Array = []          ## acciones pendientes de la decision actual
var _pasos: int = 0
var _decisiones: int = 0
var _cobertura_max: float = 0.0
var _ms_ultima: float = 0.0
var _fin: bool = false
var _mensaje: String = ""
var _traza: Array = []             ## para grabar y para comparar

# --- verificacion -----------------------------------------------------------

## Trayectoria fija del modo `comparar`: un recorrido que empuja la pieza sin
## depender de ninguna politica, para que el mismo guion pueda ejecutarse en
## pymunk y contrastar las dos fisicas.
const GUION := [
	Vector2(256.0, 420.0), Vector2(256.0, 330.0), Vector2(256.0, 300.0),
	Vector2(180.0, 300.0), Vector2(140.0, 240.0), Vector2(220.0, 200.0),
	Vector2(300.0, 220.0), Vector2(340.0, 300.0), Vector2(256.0, 360.0),
	Vector2(256.0, 260.0),
]
const PASOS_POR_TRAMO: int = 25


func _ready() -> void:
	_leer_argumentos()

	fisica = FisicaPushT.new()
	fisica.name = "Fisica"
	add_child(fisica)
	fisica.paso_control_completado.connect(_al_paso_control)

	match modo:
		"cobertura":
			_modo_cobertura()
		"comparar":
			_arrancar_comparacion()
		"reproducir":
			_arrancar_reproduccion()
		"observacion":
			_modo_observacion()
		_:
			_arrancar_vivo()


func _leer_argumentos() -> void:
	for argumento in OS.get_cmdline_user_args():
		var partes := argumento.trim_prefix("--").split("=", true, 1)
		if partes.size() != 2:
			continue
		var clave := partes[0]
		var valor := partes[1]
		match clave:
			"modo": modo = valor
			"seed": seed_episodio = int(valor)
			"obs": modo_obs = valor
			"host": host = valor
			"puerto": puerto = int(valor)
			"pasos": max_pasos = int(valor)
			"velocidad": velocidad = float(valor)
			"salida": ruta_salida = valor


func _sin_ventana() -> bool:
	return DisplayServer.get_name() == "headless"


# --- modo cobertura: verificacion 3 -----------------------------------------

func _modo_cobertura() -> void:
	# Poses elegidas a mano: la propia pose objetivo, desplazamientos puros,
	# giros puros y un caso sin solape. El servidor recalcula lo mismo con
	# shapely y `verificar_port.py` compara las dos columnas.
	var poses := [
		[PushTConst.OBJETIVO_POS, PushTConst.OBJETIVO_ANG],
		[PushTConst.OBJETIVO_POS + Vector2(20.0, 0.0), PushTConst.OBJETIVO_ANG],
		[PushTConst.OBJETIVO_POS + Vector2(0.0, 35.0), PushTConst.OBJETIVO_ANG],
		[PushTConst.OBJETIVO_POS, PushTConst.OBJETIVO_ANG + 0.3],
		[PushTConst.OBJETIVO_POS, PushTConst.OBJETIVO_ANG + PI / 2.0],
		[PushTConst.OBJETIVO_POS, 0.0],
		[Vector2(120.0, 140.0), 1.1],
		[Vector2(400.0, 420.0), -2.4],
		[PushTConst.OBJETIVO_POS + Vector2(60.0, 60.0), PushTConst.OBJETIVO_ANG],
	]
	var filas := []
	for pose in poses:
		var posicion: Vector2 = pose[0]
		var angulo: float = pose[1]
		filas.append({
			"pos": [posicion.x, posicion.y],
			"ang": angulo,
			"cobertura_godot": Cobertura.de_pose(posicion, angulo),
		})
		print("pose (%.1f, %.1f) ang %.4f -> cobertura %.6f"
			% [posicion.x, posicion.y, angulo, filas[-1]["cobertura_godot"]])
	_escribir({"modo": "cobertura", "filas": filas})
	_terminar()


# --- modo comparar: verificacion 4 ------------------------------------------

func _arrancar_comparacion() -> void:
	# Condicion inicial fija y conocida, sin servidor: se trata de contrastar
	# fisicas, no de evaluar la politica.
	fisica.colocar([256.0, 400.0, 256.0, 300.0, 0.0])
	_historia = [fisica.estado()]
	_traza = [{"paso": 0, "estado": fisica.estado(), "accion": null}]
	max_pasos = min(max_pasos, GUION.size() * PASOS_POR_TRAMO)
	fisica.objetivo_agente = GUION[0]
	fisica.arrancar()


func _accion_del_guion(paso: int) -> Vector2:
	return GUION[int(paso / PASOS_POR_TRAMO) % GUION.size()]


# --- modo vivo y grabar -----------------------------------------------------

func _arrancar_vivo() -> void:
	if not _sin_ventana():
		_montar_vista()

	cliente = ClientePolitica.new()
	var error := cliente.conectar(host, puerto)
	if error != "":
		_fallar(error)
		return

	var saludo := cliente.pedir({"cmd": "hola"})
	if not saludo.get("ok", false):
		_fallar(str(saludo.get("error", "saludo rechazado")))
		return
	modo_obs = str(saludo.get("modo_obs", modo_obs))
	_mensaje = "%s | %s | %s | obs=%s" % [
		saludo.get("variante", "?"), saludo.get("punto_control", "?"),
		saludo.get("dispositivo", "?"), modo_obs,
	]
	print(_mensaje)

	var inicio := cliente.pedir({"cmd": "reset", "seed": seed_episodio})
	if not inicio.get("ok", false):
		_fallar(str(inicio.get("error", "reset rechazado")))
		return

	fisica.colocar(inicio["estado0"])
	# El envoltorio de evaluacion rellena la historia repitiendo la primera
	# observacion, asi que los dos pasos coinciden en la primera decision.
	_historia = []
	for i in PushTConst.N_OBS_STEPS:
		_historia.append(fisica.estado())
	_traza = [{"paso": 0, "estado": fisica.estado(), "accion": null}]

	if modo == "grabar":
		Engine.time_scale = maxf(velocidad, 1.0)

	await _pedir_accion()


func _pedir_accion() -> void:
	var estados := _historia.slice(_historia.size() - PushTConst.N_OBS_STEPS)
	var posiciones := []
	for e in estados:
		posiciones.append([e[0], e[1]])
	var peticion := {
		"cmd": "act",
		"estado": estados,
		"agent_pos": posiciones,
	}
	# Parar antes de capturar: con la fisica detenida, la imagen que se dibuja
	# corresponde exactamente al estado historico y no a lo que hubiera avanzado
	# la simulacion mientras tanto.
	fisica.parar()
	if modo_obs == "godot":
		if vista == null:
			_fallar("la condicion B necesita ventana: no se puede capturar sin renderizador")
			return
		peticion["imagen"] = await vista.observacion_base64(estados)
	if not cliente.enviar(peticion):
		_fallar("se perdio la conexion con el servidor")


func _process(_delta: float) -> void:
	if _fin or cliente == null or not cliente.esperando():
		return
	var respuesta = cliente.recibir()
	if respuesta == null:
		return
	if not respuesta.get("ok", false):
		_fallar(str(respuesta.get("error", "el servidor devolvio un error")))
		return
	_acciones = respuesta["accion"]
	_ms_ultima = float(respuesta.get("ms", 0.0))
	_decisiones += 1
	fisica.arrancar()
	await _siguiente_accion()


func _siguiente_accion() -> void:
	if _acciones.is_empty():
		await _pedir_accion()
		return
	var accion = _acciones.pop_front()
	fisica.objetivo_agente = Vector2(float(accion[0]), float(accion[1]))


# --- avance comun -----------------------------------------------------------

func _al_paso_control() -> void:
	_pasos += 1
	var estado := fisica.estado()
	_historia.append(estado)
	if _historia.size() > PushTConst.N_OBS_STEPS:
		_historia = _historia.slice(_historia.size() - PushTConst.N_OBS_STEPS)

	var cobertura := fisica.cobertura()
	_cobertura_max = maxf(_cobertura_max, cobertura)
	_traza.append({
		"paso": _pasos,
		"estado": estado,
		"objetivo": [fisica.objetivo_agente.x, fisica.objetivo_agente.y],
		"cobertura": cobertura,
	})

	if hud != null:
		hud.actualizar(_pasos, cobertura, _cobertura_max, _decisiones, _ms_ultima, modo_obs)

	if cobertura > PushTConst.UMBRAL_EXITO or _pasos >= max_pasos:
		_cerrar_episodio()
		return

	if modo == "comparar":
		fisica.objetivo_agente = _accion_del_guion(_pasos)
	else:
		await _siguiente_accion()


func _cerrar_episodio() -> void:
	fisica.parar()
	Engine.time_scale = 1.0
	var resumen := {
		"modo": modo,
		"seed": seed_episodio,
		"obs": modo_obs,
		"pasos": _pasos,
		"decisiones": _decisiones,
		"cobertura_max": _cobertura_max,
		"recompensa_max": PushTConst.recompensa(_cobertura_max),
		"traza": _traza,
	}
	print("fin | %d pasos | cobertura maxima %.4f | recompensa %.4f | %d decisiones"
		% [_pasos, _cobertura_max, resumen["recompensa_max"], _decisiones])
	if ruta_salida != "" or modo in ["grabar", "comparar"]:
		_escribir(resumen)
	# `grabar` cierra siempre: es el modo de lote, y la condicion B lo ejecuta con
	# ventana porque necesita renderizador. Solo `vivo` deja el resultado a la
	# vista, que para eso es el modo de la defensa.
	if _sin_ventana() or modo == "grabar" or modo in MODOS_SIN_SERVIDOR:
		_terminar()
	else:
		_fin = true
		if hud != null:
			hud.anunciar("episodio terminado | cobertura maxima %.3f" % _cobertura_max)


# --- modo observacion: verificacion 6 ---------------------------------------

## Vuelca a PNG lo que la camara de observacion ve para una condicion inicial.
##
## Necesita renderizador, asi que no corre sin ventana. `comparar_observacion.py`
## pone ese PNG al lado del fotograma del rasterizador para el mismo estado; solo
## cuando los dos se parecen tiene sentido cerrar el bucle con obs=godot.
func _modo_observacion() -> void:
	_montar_vista()
	cliente = ClientePolitica.new()
	var error := cliente.conectar(host, puerto)
	if error != "":
		_fallar(error)
		return
	var inicio := cliente.pedir({"cmd": "reset", "seed": seed_episodio})
	if not inicio.get("ok", false):
		_fallar(str(inicio.get("error", "reset rechazado")))
		return
	var estado0: Array = inicio["estado0"]
	fisica.colocar(estado0)
	var ruta := ruta_salida if ruta_salida != "" 		else "res://grabaciones/observacion_godot_seed%d.png" % seed_episodio
	# Dos fotogramas de margen: la escena 3D se construye en _ready y sus nodos
	# no tienen transformada valida hasta que _process ha corrido una vez.
	await get_tree().process_frame
	await get_tree().process_frame
	await vista.guardar_observacion(estado0, ruta)
	var json_ruta := ruta.get_basename() + ".json"
	var archivo := FileAccess.open(json_ruta, FileAccess.WRITE)
	archivo.store_string(JSON.stringify({"seed": seed_episodio, "estado": estado0}, "  "))
	archivo.close()
	print("estado escrito en ", ProjectSettings.globalize_path(json_ruta))
	_terminar()


# --- vista ------------------------------------------------------------------

func _montar_vista() -> void:
	vista = load("res://scripts/vista3d.gd").new()
	vista.name = "Vista3D"
	vista.fisica = fisica
	add_child(vista)
	hud = load("res://scripts/hud.gd").new()
	hud.name = "HUD"
	add_child(hud)


# --- utilidades -------------------------------------------------------------

func _ruta_por_defecto() -> String:
	if ruta_salida != "":
		return ruta_salida
	# Los modos de verificacion no dependen ni de la observacion ni de la semilla,
	# y su nombre esta acordado con `servidor/verificar_port.py`, que los lee sin
	# que haya que pasarle ninguna ruta.
	if modo in MODOS_SIN_SERVIDOR:
		return "res://grabaciones/%s_godot.json" % modo
	return "res://grabaciones/%s_%s_seed%d.json" % [modo, modo_obs, seed_episodio]


func _escribir(datos: Dictionary) -> void:
	var ruta := _ruta_por_defecto()
	DirAccess.make_dir_recursive_absolute(ruta.get_base_dir())
	var archivo := FileAccess.open(ruta, FileAccess.WRITE)
	if archivo == null:
		push_error("no se pudo escribir " + ruta)
		return
	archivo.store_string(JSON.stringify(datos, "  "))
	archivo.close()
	print("escrito en ", ProjectSettings.globalize_path(ruta))


func _arrancar_reproduccion() -> void:
	# Lo que se reproduce lo escribio el modo `grabar`, asi que la ruta por
	# defecto lleva su nombre y no el de este modo.
	var ruta := ruta_salida if ruta_salida != "" \
		else "res://grabaciones/grabar_%s_seed%d.json" % [modo_obs, seed_episodio]
	var archivo := FileAccess.open(ruta, FileAccess.READ)
	if archivo == null:
		_fallar("no existe la grabacion " + ruta)
		return
	var datos = JSON.parse_string(archivo.get_as_text())
	archivo.close()
	if typeof(datos) != TYPE_DICTIONARY:
		_fallar("grabacion ilegible: " + ruta)
		return
	if not _sin_ventana():
		_montar_vista()
	_traza = datos["traza"]
	fisica.colocar(_traza[0]["estado"])
	# Sin servidor ni GPU: la fisica no vuelve a correr, solo se recolocan los
	# cuerpos en las poses grabadas. Es la red de seguridad de la defensa.
	_reproducir()


func _reproducir() -> void:
	for cuadro in _traza:
		fisica.colocar(cuadro["estado"])
		var cobertura := float(cuadro.get("cobertura", 0.0))
		_cobertura_max = maxf(_cobertura_max, cobertura)
		if hud != null:
			hud.actualizar(int(cuadro["paso"]), cobertura, _cobertura_max, 0, 0.0, "grabado")
		await get_tree().create_timer(1.0 / float(PushTConst.CONTROL_HZ) / velocidad).timeout
	var texto := "reproduccion terminada | %d cuadros | cobertura maxima %.3f" \
		% [_traza.size(), _cobertura_max]
	print(texto)
	if hud != null:
		hud.anunciar(texto)


func _fallar(mensaje: String) -> void:
	if _fin:
		return
	_fin = true
	printerr(mensaje)
	if hud != null:
		# Deja el motivo a la vista un momento antes de cerrar. Con ventana y sin
		# esta salida, un servidor apagado dejaba el proceso colgado para siempre.
		hud.anunciar(mensaje)
		await get_tree().create_timer(4.0).timeout
	quit_con(1)


func _terminar() -> void:
	quit_con(0)


func quit_con(codigo: int) -> void:
	if cliente != null:
		cliente.cerrar()
	get_tree().quit(codigo)
