class_name PushTConst
extends RefCounted

## Constantes de Push-T, copiadas del entorno original.
##
## Todas salen de `diffuser/repo/diffusion_policy/diffusion_policy/env/pusht/`.
## Ninguna es un valor de diseno: cambiar cualquiera de ellas cambia la tarea, y
## la politica se entreno sobre esta y no sobre otra. Las referencias de linea
## apuntan a `pusht_env.py` salvo indicacion contraria.

# --- mundo y tiempo ---------------------------------------------------------

## Lado del mundo en pixeles (`window_size`, :41). El eje y crece hacia abajo.
const MUNDO: float = 512.0
## Frecuencia de simulacion, en hercios (`sim_hz`, :43).
const SIM_HZ: int = 100
## Frecuencia de control, en hercios (`control_hz`, :46).
const CONTROL_HZ: int = 10
## Subpasos de fisica por paso de control (`n_steps`, :112).
const SUBPASOS: int = SIM_HZ / CONTROL_HZ
## Paso de integracion, en segundos.
const DT: float = 1.0 / float(SIM_HZ)
## Longitud maxima del episodio, en pasos de control (`max_steps` de
## `pusht_image.yaml`, y del preregistro de la prueba final).
const MAX_PASOS_CONTROL: int = 300

# --- control PD del agente --------------------------------------------------

## Ganancias del controlador (`k_p`, `k_v`, :45). Criticamente amortiguado:
## omega_n = 10 rad/s, zeta = 1.
const K_P: float = 100.0
const K_V: float = 20.0
## Radio del agente, en pixeles (:305).
const RADIO_AGENTE: float = 15.0

# --- la pieza en T ----------------------------------------------------------

## Vertices locales de las dos partes (`add_tee`, :342-367) con scale = 30 y
## length = 4. El origen local es el punto medio del borde superior de la barra.
const VERTICES_BARRA: Array[Vector2] = [
	Vector2(-60.0, 30.0), Vector2(60.0, 30.0), Vector2(60.0, 0.0), Vector2(-60.0, 0.0),
]
const VERTICES_VASTAGO: Array[Vector2] = [
	Vector2(-15.0, 30.0), Vector2(-15.0, 120.0), Vector2(15.0, 120.0), Vector2(15.0, 30.0),
]
const MASA: float = 1.0
## Momento de inercia. El original calcula el segundo momento con los vertices
## del primero (:354), asi que sale 2 x 1500 y no 1500 + 6375. Se replica el
## fallo: es el numero con el que se genero el conjunto de demostraciones.
const INERCIA: float = 3000.0
## Centro de gravedad local (:363). Es la media simple de los dos centroides, no
## la ponderada por area. Que no coincida con el origen es la razon de que fijar
## el angulo desplace la posicion.
const CENTRO_GRAVEDAD: Vector2 = Vector2(0.0, 45.0)
## Los contactos no tienen ni rozamiento ni rebote: `body.friction = 1` de :365
## no hace nada porque en Chipmunk la friccion vive en la forma, y ahi nunca se
## asigna. Ver el modulo 06 del manual.
const FRICCION: float = 0.0
const REBOTE: float = 0.0

# --- objetivo y puntuacion --------------------------------------------------

## Pose objetivo: x, y, theta (:308).
const OBJETIVO_POS: Vector2 = Vector2(256.0, 256.0)
const OBJETIVO_ANG: float = PI / 4.0
## Area de la T: 120x30 + 30x90.
const AREA_OBJETIVO: float = 6300.0
## `success_threshold` (:316). Una cobertura de 0,95 ya puntua 1.
const UMBRAL_EXITO: float = 0.95

# --- paredes ----------------------------------------------------------------

## Los cuatro segmentos estaticos van de (5,5) a (506,506) con radio 2 (:295),
## de modo que la cara interior queda a 7 y a 504.
const PARED_MIN: float = 5.0
const PARED_MAX: float = 506.0
const PARED_RADIO: float = 2.0

# --- politica ---------------------------------------------------------------

## Pasos de observacion y de accion (`n_obs_steps`, `n_action_steps` de
## `train_diffusion_unet_image_workspace.yaml`).
const N_OBS_STEPS: int = 2
const N_ACTION_STEPS: int = 8
## Lado de la observacion que consume el codificador.
const LADO_OBS: int = 96
## Lado al que se dibuja antes de reducir. El original dibuja a 512 sin
## antialiasing y baja a 96 con interpolacion bilineal.
const LADO_DIBUJO: int = 512

# --- colores del renderizador original --------------------------------------
#
# De `pymunk_override.py`. `light_color` multiplica por 1,2 y satura en 255, de
# ahi que el relleno sea mas claro que el borde.

const COLOR_FONDO: Color = Color8(255, 255, 255)
const COLOR_OBJETIVO: Color = Color8(144, 238, 144)
const COLOR_T_RELLENO: Color = Color8(142, 163, 183)
const COLOR_T_BORDE: Color = Color8(119, 136, 153)
const COLOR_AGENTE_RELLENO: Color = Color8(78, 126, 255)
const COLOR_AGENTE_BORDE: Color = Color8(65, 105, 225)
const COLOR_PARED: Color = Color8(211, 211, 211)
## Grosor del borde que `draw_polygon` y `draw_circle` pintan encima, en pixeles
## del lienzo de 512.
const GROSOR_BORDE: float = 4.0


## Vertices de la T, en mundo, para una pose dada.
##
## Chipmunk situa `body.position` en el origen del cuerpo, no en su centro de
## gravedad, y `local_to_world` rota alrededor de ese origen. La dinamica si gira
## alrededor del centro de gravedad, pero eso ya esta contenido en la pose.
static func vertices_mundo(posicion: Vector2, angulo: float) -> Array:
	var salida: Array = []
	for locales in [VERTICES_BARRA, VERTICES_VASTAGO]:
		var poligono := PackedVector2Array()
		for v in locales:
			poligono.append(posicion + v.rotated(angulo))
		salida.append(poligono)
	return salida


## Vertices de la T objetivo, en mundo.
static func vertices_objetivo() -> Array:
	return vertices_mundo(OBJETIVO_POS, OBJETIVO_ANG)


## Recompensa por paso, con el mismo recorte que `pusht_env.step` (:132).
static func recompensa(cobertura: float) -> float:
	return clampf(cobertura / UMBRAL_EXITO, 0.0, 1.0)
