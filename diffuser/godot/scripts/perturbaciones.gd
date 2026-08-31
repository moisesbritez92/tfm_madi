class_name Perturbacion
extends RefCounted

## Cambios deliberados de la escena, para ver cuanto aguanta la politica.
##
## La pregunta es hasta que punto una politica entrenada sobre un renderizador
## concreto depende de la apariencia y no de la geometria. Cada preset toca una
## cosa y deja el resto quieto.
##
## Solo afectan a la **condicion B**. En la condicion A la imagen la dibuja
## `rasterizador_pusht.py` en Python con el codigo del entrenamiento, y ese no
## sabe nada de esto: la condicion A es, por construccion, la escena sin
## perturbar.
##
## La fisica tampoco cambia nunca. Se perturba lo que la politica ve, no lo que
## el mundo hace, para que cualquier degradacion sea atribuible a la imagen.

const NINGUNA := "ninguna"
const T_ROJA := "t_roja"
const SOMBRAS := "sombras"
## Prefijo de la familia parametrica: `t_color_b23222` pinta la pieza de ese
## color. Existe para poder barrer el tono de forma continua y ver si la
## degradacion es gradual o hay un acantilado. Se usa guion bajo y no dos puntos
## porque el nombre acaba en un nombre de fichero, y Windows no admite dos puntos.
const T_COLOR := "t_color_"

## Rojo base de `t_roja`, el `firebrick` de las listas de colores con nombre. Se
## usa como borde, y el relleno sale de aclararlo igual que hace el original.
const ROJO_BASE: Color = Color8(178, 34, 34)

var nombre: String = NINGUNA
var color_t_relleno: Color = PushTConst.COLOR_T_RELLENO
var color_t_borde: Color = PushTConst.COLOR_T_BORDE
var color_agente_relleno: Color = PushTConst.COLOR_AGENTE_RELLENO
var color_agente_borde: Color = PushTConst.COLOR_AGENTE_BORDE
var color_objetivo: Color = PushTConst.COLOR_OBJETIVO
var color_fondo: Color = PushTConst.COLOR_FONDO
## Si es cierto, la camara de observacion deja de usar materiales sin sombreado y
## ve la escena iluminada, con sombras y brillo especular.
var sombreado: bool = false
## Descripcion corta, para el panel y para la bitacora.
var descripcion: String = "escena sin perturbar"


static func nombres() -> Array:
	return [NINGUNA, T_ROJA, SOMBRAS]


## De "b23222" al color. Devuelve un color invalido si el texto no lo es.
static func _de_hexadecimal(texto: String) -> Color:
	if texto.length() != 6 or not texto.is_valid_hex_number():
		return Color(-1.0, -1.0, -1.0)
	return Color8(
		("0x" + texto.substr(0, 2)).hex_to_int(),
		("0x" + texto.substr(2, 2)).hex_to_int(),
		("0x" + texto.substr(4, 2)).hex_to_int()
	)


static func crear(nombre_pedido: String) -> Perturbacion:
	var p := Perturbacion.new()
	p.nombre = nombre_pedido
	match nombre_pedido:
		NINGUNA:
			pass
		T_ROJA:
			# Solo cambia el tono de la pieza. El relleno se deriva del borde con
			# el mismo aclarado que el renderizador original, de modo que la
			# relacion entre los dos tonos es la misma que la politica vio
			# durante el entrenamiento y lo unico distinto es el color.
			p.color_t_borde = ROJO_BASE
			p.color_t_relleno = PushTConst.aclarar(ROJO_BASE)
			p.descripcion = "la pieza pasa de gris azulado a rojo"
		SOMBRAS:
			# Ningun color nominal cambia; cambian todos los pixeles. Es la
			# ablacion de la separacion en dos capas de visibilidad que hizo
			# viable la condicion B.
			p.sombreado = true
			p.descripcion = "la observacion se renderiza iluminada, con sombras"
		_:
			if nombre_pedido.begins_with(T_COLOR):
				var color := _de_hexadecimal(nombre_pedido.trim_prefix(T_COLOR))
				if color.r < 0.0:
					push_error("color mal formado en: " + nombre_pedido)
					p.nombre = NINGUNA
				else:
					# Mismo trato que `t_roja`: el color pedido es el borde y el
					# relleno se aclara igual que en el renderizador original, de
					# modo que lo unico que varia a lo largo del barrido es el tono.
					p.color_t_borde = color
					p.color_t_relleno = PushTConst.aclarar(color)
					p.descripcion = "la pieza se pinta de #" + nombre_pedido.trim_prefix(T_COLOR)
			else:
				push_error("perturbacion desconocida: " + nombre_pedido)
				p.nombre = NINGUNA
	return p
