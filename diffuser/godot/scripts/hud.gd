extends CanvasLayer

## Panel de la demostracion.
##
## Muestra la cobertura, el paso de control y la latencia de la ultima decision.
## Lleva fija una linea que recuerda que la cifra es ilustrativa: el resultado
## del TFM es la pasada preregistrada sobre el bloque 200000-200199, no lo que
## pase en esta ventana.

const AVISO := "cifra ilustrativa | el resultado del TFM es la pasada preregistrada sobre 200000-200199"

var _lineas: Label
var _aviso: Label
var _anuncio: Label


func _ready() -> void:
	var panel := PanelContainer.new()
	panel.position = Vector2(16, 16)
	panel.custom_minimum_size = Vector2(430, 0)
	add_child(panel)

	var caja := VBoxContainer.new()
	caja.add_theme_constant_override("separation", 6)
	panel.add_child(caja)

	var titulo := Label.new()
	titulo.text = "Push-T en Godot | V0, ResNet-18 desde cero"
	titulo.add_theme_font_size_override("font_size", 18)
	caja.add_child(titulo)

	_lineas = Label.new()
	_lineas.add_theme_font_size_override("font_size", 15)
	caja.add_child(_lineas)

	_anuncio = Label.new()
	_anuncio.add_theme_font_size_override("font_size", 15)
	_anuncio.add_theme_color_override("font_color", Color(0.95, 0.75, 0.25))
	caja.add_child(_anuncio)

	_aviso = Label.new()
	_aviso.text = AVISO
	_aviso.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_aviso.custom_minimum_size = Vector2(400, 0)
	_aviso.add_theme_font_size_override("font_size", 12)
	_aviso.add_theme_color_override("font_color", Color(0.65, 0.65, 0.68))
	caja.add_child(_aviso)

	actualizar(0, 0.0, 0.0, 0, 0.0, "-")


func actualizar(
	paso: int, cobertura: float, cobertura_max: float,
	decisiones: int, ms: float, obs: String
) -> void:
	# El factor de tiempo real sale de comparar lo que cuesta pensar con lo que
	# cubre la decision: ocho pasos de control a 10 Hz son 0,8 s de simulacion.
	var simulado := float(PushTConst.N_ACTION_STEPS) / float(PushTConst.CONTROL_HZ)
	var factor := simulado / (ms / 1000.0) if ms > 0.0 else 0.0
	_lineas.text = "\n".join([
		"observacion: %s" % obs,
		"paso de control: %d / %d" % [paso, PushTConst.MAX_PASOS_CONTROL],
		"cobertura: %.3f (maxima %.3f, umbral %.2f)"
			% [cobertura, cobertura_max, PushTConst.UMBRAL_EXITO],
		"recompensa: %.3f" % PushTConst.recompensa(cobertura_max),
		"decisiones: %d | ultima %.0f ms | %.2f x tiempo real" % [decisiones, ms, factor],
	])


func anunciar(texto: String) -> void:
	_anuncio.text = texto
