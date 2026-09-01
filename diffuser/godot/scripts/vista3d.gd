extends Node3D

## La escena 3D y la camara de observacion.
##
## Dos camaras miran el mismo mundo con dos propositos incompatibles:
##
##   - La camara de la demostracion es una perspectiva inclinada, con luz y
##     sombras. Es lo que ve el publico.
##   - La camara de observacion es ortografica y estrictamente cenital, dentro
##     de un SubViewport de 512 x 512 sin antialiasing, con fondo blanco y
##     materiales sin sombreado en los colores exactos del entrenamiento. Es lo
##     que ve la politica en la condicion B.
##
## Cada elemento existe dos veces, en dos capas de visibilidad distintas, con la
## misma transformada: la version bonita en la capa 1 y la version plana en la
## capa 2. Sin esa separacion, la condicion B fracasaria por motivos triviales
## (una sombra, un brillo especular, la perspectiva) y no se aprenderia nada de
## ella.
##
## Un pixel del mundo Push-T es una unidad de Godot: x va a x, y va a z, y la
## altura se reserva para apilar los elementos en el orden en que el
## renderizador original los dibuja.

const CAPA_DEMO: int = 1
const CAPA_OBS: int = 2

# Alturas de apilado. El objetivo va debajo de todo, igual que en
# `_render_frame`, y el agente encima.
const Y_MESA: float = -1.0
const Y_OBJETIVO: float = 0.5
const Y_PARED: float = 1.0
const Y_BLOQUE: float = 2.0
const Y_AGENTE: float = 6.0
const GROSOR: float = 20.0
## Correccion de medio pixel de la camara ortografica. Godot muestrea en el
## centro del pixel, de modo que la coordenada de mundo w cae en el pixel
## w - 0,5; pygame rellena el poligono con el pixel entero. Desplazar la
## camara devuelve las dos imagenes a la misma rejilla. Se fijo midiendo, con
## `comparar_observacion.py`, no razonando.
const DESPLAZAMIENTO_OBS: float = -0.5

var fisica: FisicaPushT
## Que se le ha cambiado a la escena. Se asigna antes de entrar al arbol; si nadie
## la fija, la escena es la original.
var perturbacion: Perturbacion = Perturbacion.crear(Perturbacion.NINGUNA)

var _nodo_agente: Node3D
var _nodo_bloque: Node3D
var _subviewport: SubViewport
var _camara_obs: Camera3D
## Mientras se capturan las observaciones, `_process` no debe tocar las poses.
## Los callbacks de proceso corren antes del dibujado, asi que sin este cerrojo
## reescribian la pose historica por la actual y las dos observaciones salian
## iguales: la politica veia un historial congelado.
var _congelado: bool = false


func _ready() -> void:
	_construir_mesa()
	_construir_paredes()
	_construir_objetivo()
	_construir_bloque()
	_construir_agente()
	_construir_camara_demo()
	_construir_camara_obs()
	_construir_luz()


func _process(_delta: float) -> void:
	if fisica == null or _congelado:
		return
	_nodo_bloque.position = _a_3d(fisica.bloque.global_position, Y_BLOQUE)
	_nodo_bloque.rotation.y = -fisica.bloque.global_rotation
	_nodo_agente.position = _a_3d(fisica.agente.global_position, Y_AGENTE)


## Del plano de Push-T al espacio de Godot. El giro cambia de signo porque en el
## plano el angulo crece de x hacia y, y aqui de x hacia z, que es la mano
## contraria.
func _a_3d(punto: Vector2, altura: float) -> Vector3:
	return Vector3(punto.x, altura, punto.y)


# --- construccion -----------------------------------------------------------

## Una pieza con sus dos encarnaciones: la iluminada y la plana.
func _pieza(malla: Mesh, color_demo: Color, color_obs: Color) -> Node3D:
	var raiz := Node3D.new()

	var demo := MeshInstance3D.new()
	demo.mesh = malla
	demo.layers = CAPA_DEMO
	var material_demo := StandardMaterial3D.new()
	material_demo.albedo_color = color_demo
	material_demo.roughness = 0.65
	material_demo.metallic = 0.0
	demo.material_override = material_demo
	raiz.add_child(demo)

	var obs := MeshInstance3D.new()
	obs.mesh = malla
	obs.layers = CAPA_OBS
	if perturbacion.sombreado:
		# Ablacion deliberada: la camara de observacion pasa a ver lo mismo que la
		# de la demostracion, iluminado y con sombras. Ningun color nominal
		# cambia; cambian todos los pixeles.
		obs.material_override = material_demo
	else:
		obs.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		var material_obs := StandardMaterial3D.new()
		material_obs.albedo_color = color_obs
		# Sin sombreado el color sale tal cual, sin luz ni ambiente que lo desvie.
		material_obs.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		obs.material_override = material_obs
	raiz.add_child(obs)

	return raiz


func _caja(ancho: float, alto: float, fondo: float) -> BoxMesh:
	var malla := BoxMesh.new()
	malla.size = Vector3(ancho, alto, fondo)
	return malla


## Un rectangulo con el borde de 4 px que dibuja `draw_polygon`: el borde va
## centrado en la arista, asi que sobresale 2 px y come 2 px hacia dentro.
func _rectangulo_con_borde(
	centro: Vector2, ancho: float, fondo: float, altura: float,
	color_relleno: Color, color_borde: Color
) -> Node3D:
	var raiz := Node3D.new()
	var mitad := PushTConst.GROSOR_BORDE * 0.5

	var borde := _pieza(
		_caja(ancho + PushTConst.GROSOR_BORDE, GROSOR, fondo + PushTConst.GROSOR_BORDE),
		color_borde, color_borde
	)
	borde.position = Vector3(centro.x, altura, centro.y)
	raiz.add_child(borde)

	var relleno := _pieza(
		_caja(ancho - PushTConst.GROSOR_BORDE, GROSOR, fondo - PushTConst.GROSOR_BORDE),
		color_relleno, color_relleno
	)
	relleno.position = Vector3(centro.x, altura + mitad, centro.y)
	raiz.add_child(relleno)

	return raiz


func _construir_mesa() -> void:
	# Solo en la capa de la demostracion: la camara de observacion ve el fondo
	# blanco de su propio entorno, que es lo que hay bajo el lienzo original.
	var mesa := MeshInstance3D.new()
	mesa.mesh = _caja(PushTConst.MUNDO, 2.0, PushTConst.MUNDO)
	mesa.layers = CAPA_DEMO
	mesa.position = Vector3(PushTConst.MUNDO * 0.5, Y_MESA, PushTConst.MUNDO * 0.5)
	var material := StandardMaterial3D.new()
	material.albedo_color = perturbacion.color_fondo
	material.roughness = 0.9
	mesa.material_override = material
	add_child(mesa)


func _construir_paredes() -> void:
	var lo := PushTConst.PARED_MIN
	var hi := PushTConst.PARED_MAX
	var r := PushTConst.PARED_RADIO
	var largo := hi - lo + 2.0 * r
	var medio := (lo + hi) * 0.5
	var paredes := [
		[Vector2(lo, medio), 2.0 * r, largo],
		[Vector2(hi, medio), 2.0 * r, largo],
		[Vector2(medio, lo), largo, 2.0 * r],
		[Vector2(medio, hi), largo, 2.0 * r],
	]
	for pared in paredes:
		var centro: Vector2 = pared[0]
		var nodo := _pieza(
			_caja(pared[1], GROSOR, pared[2]),
			PushTConst.COLOR_PARED, PushTConst.COLOR_PARED
		)
		nodo.position = Vector3(centro.x, Y_PARED, centro.y)
		add_child(nodo)


func _construir_objetivo() -> void:
	# El objetivo se dibuja plano y sin borde, debajo de todo lo demas.
	var nodo := Node3D.new()
	nodo.position = Vector3(PushTConst.OBJETIVO_POS.x, Y_OBJETIVO, PushTConst.OBJETIVO_POS.y)
	nodo.rotation.y = -PushTConst.OBJETIVO_ANG
	for parte in _partes_de_la_t():
		var caja := _pieza(
			_caja(parte[1], 1.0, parte[2]),
			perturbacion.color_objetivo, perturbacion.color_objetivo
		)
		caja.position = Vector3(parte[0].x, 0.0, parte[0].y)
		nodo.add_child(caja)
	add_child(nodo)


## Centro local, ancho y fondo de las dos partes de la T, deducidos de los
## vertices para no repetir numeros que ya estan en las constantes.
func _partes_de_la_t() -> Array:
	var partes := []
	for vertices: Array[Vector2] in [PushTConst.VERTICES_BARRA, PushTConst.VERTICES_VASTAGO]:
		var minimo: Vector2 = vertices[0]
		var maximo: Vector2 = vertices[0]
		for v in vertices:
			minimo = minimo.min(v)
			maximo = maximo.max(v)
		partes.append([(minimo + maximo) * 0.5, maximo.x - minimo.x, maximo.y - minimo.y])
	return partes


func _construir_bloque() -> void:
	_nodo_bloque = Node3D.new()
	_nodo_bloque.name = "Bloque"
	var altura := 0.0
	for parte in _partes_de_la_t():
		# La barra primero y el vastago encima, como en el orden de dibujo del
		# original, donde el relleno del segundo tapa parte del borde del primero.
		var caja := _rectangulo_con_borde(
			parte[0], parte[1], parte[2], altura,
			perturbacion.color_t_relleno, perturbacion.color_t_borde
		)
		_nodo_bloque.add_child(caja)
		altura += 0.5
	add_child(_nodo_bloque)


func _construir_agente() -> void:
	_nodo_agente = Node3D.new()
	_nodo_agente.name = "Agente"
	var borde := _pieza(
		_cilindro(PushTConst.RADIO_AGENTE, GROSOR),
		perturbacion.color_agente_borde, perturbacion.color_agente_borde
	)
	_nodo_agente.add_child(borde)
	var relleno := _pieza(
		_cilindro(PushTConst.RADIO_AGENTE - PushTConst.GROSOR_BORDE, GROSOR),
		perturbacion.color_agente_relleno, perturbacion.color_agente_relleno
	)
	relleno.position.y = 1.0
	_nodo_agente.add_child(relleno)
	add_child(_nodo_agente)


func _cilindro(radio: float, alto: float) -> CylinderMesh:
	var malla := CylinderMesh.new()
	malla.top_radius = radio
	malla.bottom_radius = radio
	malla.height = alto
	malla.radial_segments = 48
	return malla


func _construir_camara_demo() -> void:
	var camara := Camera3D.new()
	camara.name = "CamaraDemo"
	camara.cull_mask = CAPA_DEMO
	camara.position = Vector3(PushTConst.MUNDO * 0.5, 620.0, PushTConst.MUNDO * 1.15)
	camara.look_at_from_position(camara.position, Vector3(256.0, 0.0, 300.0), Vector3.UP)
	camara.fov = 45.0
	camara.far = 4000.0
	var entorno := Environment.new()
	entorno.background_mode = Environment.BG_COLOR
	entorno.background_color = Color(0.92, 0.93, 0.95)
	entorno.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	entorno.ambient_light_color = Color(0.85, 0.88, 0.92)
	entorno.ambient_light_energy = 0.6
	camara.environment = entorno
	add_child(camara)
	camara.current = true


func _construir_luz() -> void:
	var luz := DirectionalLight3D.new()
	luz.name = "Luz"
	luz.rotation_degrees = Vector3(-55.0, -35.0, 0.0)
	luz.light_energy = 1.1
	luz.shadow_enabled = true
	add_child(luz)


func _construir_camara_obs() -> void:
	_subviewport = SubViewport.new()
	_subviewport.name = "ViewportObservacion"
	_subviewport.size = Vector2i(PushTConst.LADO_DIBUJO, PushTConst.LADO_DIBUJO)
	# El original dibuja a 512 sin antialiasing y reduce con interpolacion
	# bilineal. Ese aliasing forma parte de la textura del conjunto de datos, asi
	# que aqui tampoco se suaviza nada.
	_subviewport.msaa_3d = Viewport.MSAA_DISABLED
	_subviewport.screen_space_aa = Viewport.SCREEN_SPACE_AA_DISABLED
	_subviewport.use_taa = false
	_subviewport.use_debanding = false
	_subviewport.transparent_bg = false
	# Sin mundo propio: la camara de observacion mira la misma escena que la de
	# la demostracion, y las separa la mascara de capas y no la geometria.
	_subviewport.own_world_3d = false
	# El viewport de observacion no se muestra en ninguna parte: solo dibuja
	# cuando se le pide una captura.
	_subviewport.render_target_update_mode = SubViewport.UPDATE_DISABLED
	add_child(_subviewport)

	_camara_obs = Camera3D.new()
	_camara_obs.name = "CamaraObservacion"
	_camara_obs.cull_mask = CAPA_OBS
	_camara_obs.projection = Camera3D.PROJECTION_ORTHOGONAL
	_camara_obs.size = PushTConst.MUNDO
	_camara_obs.near = 1.0
	_camara_obs.far = 2000.0
	# Cenital estricta. Con esta rotacion el eje y del plano crece hacia abajo en
	# la imagen, que es el convenio de pygame y del conjunto de entrenamiento.
	var centro := PushTConst.MUNDO * 0.5 + DESPLAZAMIENTO_OBS
	_camara_obs.position = Vector3(centro, 600.0, centro)
	_camara_obs.rotation = Vector3(-PI * 0.5, 0.0, 0.0)
	var entorno := Environment.new()
	entorno.background_mode = Environment.BG_COLOR
	entorno.background_color = perturbacion.color_fondo
	if perturbacion.sombreado:
		entorno.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
		entorno.ambient_light_color = Color(0.85, 0.88, 0.92)
		entorno.ambient_light_energy = 0.6
	else:
		entorno.ambient_light_source = Environment.AMBIENT_SOURCE_DISABLED
	entorno.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	_camara_obs.environment = entorno
	_subviewport.add_child(_camara_obs)
	_camara_obs.current = true


# --- observacion ------------------------------------------------------------

## Las dos observaciones de 96 x 96 en base64, una por estado del historial.
##
## Se dibujan a demanda moviendo los nodos a cada pose historica. Se puede
## porque la fisica esta detenida mientras se pide la accion, de modo que la
## imagen corresponde exactamente al estado y no a lo que hubiera en pantalla.
func observacion_base64(estados: Array) -> Array:
	var pose_bloque := _nodo_bloque.position
	var giro_bloque := _nodo_bloque.rotation.y
	var pose_agente := _nodo_agente.position
	_congelado = true

	var salida := []
	for estado in estados:
		_nodo_bloque.position = Vector3(float(estado[2]), Y_BLOQUE, float(estado[3]))
		_nodo_bloque.rotation.y = -float(estado[4])
		_nodo_agente.position = Vector3(float(estado[0]), Y_AGENTE, float(estado[1]))
		salida.append(await _capturar())

	_nodo_bloque.position = pose_bloque
	_nodo_bloque.rotation.y = giro_bloque
	_nodo_agente.position = pose_agente
	_congelado = false
	return salida


## Captura la vista de la demostracion en una pose dada, para las figuras.
##
## A diferencia de `observacion_base64`, que lee el SubViewport ortografico y
## plano de la politica, esto lee el viewport raiz: perspectiva, luz y sombras,
## es decir lo que ve el publico. No interviene en ningun bucle de control.
func vista_demo_base64(estado: Array) -> String:
	var pose_bloque := _nodo_bloque.position
	var giro_bloque := _nodo_bloque.rotation.y
	var pose_agente := _nodo_agente.position
	_congelado = true

	_nodo_bloque.position = Vector3(float(estado[2]), Y_BLOQUE, float(estado[3]))
	_nodo_bloque.rotation.y = -float(estado[4])
	_nodo_agente.position = Vector3(float(estado[0]), Y_AGENTE, float(estado[1]))
	# Dos pasadas: la primera aplica las transformadas y la segunda las dibuja
	# con el mapa de sombras ya recalculado para la pose nueva.
	await RenderingServer.frame_post_draw
	await RenderingServer.frame_post_draw
	var imagen := get_viewport().get_texture().get_image()

	_nodo_bloque.position = pose_bloque
	_nodo_bloque.rotation.y = giro_bloque
	_nodo_agente.position = pose_agente
	_congelado = false
	return Marshalls.raw_to_base64(imagen.save_png_to_buffer())


func guardar_vista_demo(estado: Array, ruta: String) -> void:
	var b64: String = await vista_demo_base64(estado)
	var archivo := FileAccess.open(ruta, FileAccess.WRITE)
	if archivo == null:
		push_error("no se pudo escribir " + ruta)
		return
	archivo.store_buffer(Marshalls.base64_to_raw(b64))
	archivo.close()
	print("vista escrita en ", ProjectSettings.globalize_path(ruta))


func _capturar() -> String:
	_subviewport.render_target_update_mode = SubViewport.UPDATE_ONCE
	await RenderingServer.frame_post_draw
	var imagen := _subviewport.get_texture().get_image()
	imagen.resize(PushTConst.LADO_OBS, PushTConst.LADO_OBS, Image.INTERPOLATE_BILINEAR)
	imagen.convert(Image.FORMAT_RGB8)
	return Marshalls.raw_to_base64(imagen.save_png_to_buffer())


## Vuelca una observacion a disco, para el paso 6 de verificacion: compararla
## con el fotograma del rasterizador para el mismo estado.
func guardar_observacion(estado: Array, ruta: String) -> void:
	var b64: Array = await observacion_base64([estado])
	var archivo := FileAccess.open(ruta, FileAccess.WRITE)
	if archivo == null:
		push_error("no se pudo escribir " + ruta)
		return
	archivo.store_buffer(Marshalls.base64_to_raw(b64[0]))
	archivo.close()
	print("observacion escrita en ", ProjectSettings.globalize_path(ruta))
