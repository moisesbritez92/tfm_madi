class_name Cobertura
extends RefCounted

## Solape entre la pieza y la pose objetivo, que es la puntuacion de Push-T.
##
## El original lo hace con shapely: `coverage = area(interseccion) / area(objetivo)`
## sobre las dos partes de la T (`pusht_env.py:124-133`). Aqui se hace con
## `Geometry2D.intersect_polygons`, que es Clipper. Sumar las cuatro
## intersecciones cruzadas es correcto porque las dos partes de la T solo
## comparten una arista, y una arista no tiene area.
##
## El servidor recalcula lo mismo con shapely bajo el comando `cobertura`: si las
## dos cifras se separan mas de 1e-3, el error esta en los vertices o en el
## convenio de angulo, y conviene mirarlo antes que cualquier otra cosa.


## Area de un poligono por la formula del cordon de zapato.
static func area(poligono: PackedVector2Array) -> float:
	var n := poligono.size()
	if n < 3:
		return 0.0
	var doble := 0.0
	for i in n:
		var a := poligono[i]
		var b := poligono[(i + 1) % n]
		doble += a.x * b.y - b.x * a.y
	return absf(doble) * 0.5


## Cobertura de la pieza en la pose dada, en [0, 1].
static func de_pose(posicion: Vector2, angulo: float) -> float:
	var pieza := PushTConst.vertices_mundo(posicion, angulo)
	var objetivo := PushTConst.vertices_objetivo()
	var solape := 0.0
	for parte_pieza in pieza:
		for parte_objetivo in objetivo:
			for trozo in Geometry2D.intersect_polygons(
				PackedVector2Array(parte_pieza), PackedVector2Array(parte_objetivo)
			):
				solape += area(trozo)
	return solape / PushTConst.AREA_OBJETIVO
