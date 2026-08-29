class_name ClientePolitica
extends RefCounted

## Cliente del servidor de politica: JSON por lineas sobre TCP.
##
## No hay hilos. La peticion se envia y se sondea la respuesta en `_process`, de
## modo que la ventana sigue viva mientras V0 piensa. La simulacion, en cambio,
## si se detiene: no tiene acciones que ejecutar hasta que llegue la respuesta.
##
## Esa espera no es un defecto de la implementacion. V0 tarda una mediana de
## 1743,9 ms por llamada en este equipo (`memoria/datos/latencia_inferencia.csv`)
## y cada llamada cubre 0,8 s de simulacion, asi que el techo del bucle cerrado
## es ~0,46 veces el tiempo real. La demostracion lo rotula en lugar de
## disimularlo.

const TIEMPO_CONEXION_MS: int = 5000

var _tcp: StreamPeerTCP = StreamPeerTCP.new()
var _buffer: PackedByteArray = PackedByteArray()
var _esperando: bool = false

var host: String = "127.0.0.1"
var puerto: int = 5555


func conectado() -> bool:
	return _tcp.get_status() == StreamPeerTCP.STATUS_CONNECTED


func esperando() -> bool:
	return _esperando


func conectar(a_host: String, a_puerto: int) -> String:
	host = a_host
	puerto = a_puerto
	var error := _tcp.connect_to_host(host, puerto)
	if error != OK:
		return "no se pudo abrir el socket hacia %s:%d (%d)" % [host, puerto, error]

	var limite := Time.get_ticks_msec() + TIEMPO_CONEXION_MS
	while Time.get_ticks_msec() < limite:
		_tcp.poll()
		var estado := _tcp.get_status()
		if estado == StreamPeerTCP.STATUS_CONNECTED:
			_tcp.set_no_delay(true)
			return ""
		if estado == StreamPeerTCP.STATUS_ERROR:
			break
		OS.delay_msec(10)
	return "sin respuesta de %s:%d. Arranca antes servidor_politica.py" % [host, puerto]


func enviar(mensaje: Dictionary) -> bool:
	if not conectado():
		return false
	var linea := (JSON.stringify(mensaje) + "\n").to_utf8_buffer()
	if _tcp.put_data(linea) != OK:
		return false
	_esperando = true
	return true


## Devuelve la respuesta si hay una completa, o null si todavia no ha llegado.
func recibir() -> Variant:
	if not conectado():
		return null
	_tcp.poll()
	var disponibles := _tcp.get_available_bytes()
	if disponibles > 0:
		var lectura := _tcp.get_data(disponibles)
		if lectura[0] == OK:
			_buffer.append_array(lectura[1])

	var corte := _buffer.find(10)   # \n
	if corte < 0:
		return null
	var linea := _buffer.slice(0, corte).get_string_from_utf8()
	_buffer = _buffer.slice(corte + 1)
	_esperando = false

	var analizado = JSON.parse_string(linea)
	if typeof(analizado) != TYPE_DICTIONARY:
		return {"ok": false, "error": "respuesta ilegible: " + linea}
	return analizado


## Envia y espera. Solo para el saludo y el reinicio, que ocurren una vez por
## episodio y cuestan milisegundos; el comando `act` nunca debe pasar por aqui.
func pedir(mensaje: Dictionary, tiempo_ms: int = 30000) -> Dictionary:
	if not enviar(mensaje):
		return {"ok": false, "error": "no hay conexion con el servidor"}
	var limite := Time.get_ticks_msec() + tiempo_ms
	while Time.get_ticks_msec() < limite:
		var respuesta = recibir()
		if respuesta != null:
			return respuesta
		OS.delay_msec(5)
	_esperando = false
	return {"ok": false, "error": "el servidor no respondio a " + str(mensaje.get("cmd"))}


func cerrar() -> void:
	if conectado():
		enviar({"cmd": "adios"})
		_tcp.poll()
	_tcp.disconnect_from_host()
