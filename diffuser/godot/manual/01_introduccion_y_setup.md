# 01. Introduccion y Setup

## Objetivo del modulo

Dejar Godot instalado, crear el primer proyecto 3D y entender el recorrido minimo para abrir el editor, ejecutar una escena y no perderse en la interfaz.

## Por que Godot para este TFM

Godot no es solo un motor para videojuegos. Para este trabajo interesa por cuatro motivos practicos:

- Permite iterar rapido con escenas 2D y 3D desde una unica interfaz.
- Tiene una estructura de proyecto clara y un editor ligero, util para prototipos rapidos.
- GDScript reduce friccion al probar ideas sin demasiada infraestructura.
- El pipeline de escena, fisica, camaras e interaccion es suficiente para construir entornos experimentales antes de decidir si hace falta algo mas pesado.

## Que conviene tener claro desde el principio

- Godot es un motor generalista, no un simulador robótico especializado.
- Si mas adelante necesitas dinamicas fisicas mas exactas, sensores especificos o integracion avanzada con stacks de robotica, probablemente Godot sera una capa de prototipado o visualizacion, no todo el entorno final.
- Aun asi, aprender Godot ahora tiene valor: te obliga a dominar escenas, eventos, assets, colisiones y flujo de herramientas, que son competencias transferibles.

## Version recomendada

Trabaja con una version estable de Godot 4.x y consulta siempre la documentacion `stable`.

Evita dos errores comunes:

1. Mezclar tutoriales de Godot 3 con Godot 4.
2. Seguir ejemplos de la rama `latest` de la documentacion cuando el editor instalado es estable.

## Instalacion basica en Windows

1. Ir a la pagina oficial de descargas de Godot.
2. Descargar la version estable del editor.
3. Descomprimir o instalar el ejecutable segun el formato ofrecido.
4. Abrir Godot y confirmar que aparece el gestor de proyectos.

En esta etapa no necesitas configuraciones avanzadas ni plantillas de exportacion.

## Crear el primer proyecto

Propuesta para el primer proyecto del manual:

- Nombre: `godot_3d_lab`
- Ruta: una carpeta separada y limpia dentro de tu espacio de trabajo o en otra ubicacion destinada a pruebas.
- Render: dejar los ajustes por defecto de Godot 4 salvo que el hardware te obligue a bajar calidad.

Pasos:

1. En el gestor de proyectos, crear un proyecto nuevo.
2. Elegir carpeta vacia.
3. Confirmar la creacion y abrir el editor.
4. Guardar una escena inicial en cuanto el editor abra por primera vez.

## Primer recorrido del editor

En una primera lectura no necesitas dominar todo. Solo ubicar cinco zonas:

| Zona | Para que sirve |
|---|---|
| Scene | Muestra la jerarquia de nodos de la escena actual. |
| Inspector | Edita propiedades del nodo seleccionado. |
| FileSystem | Muestra archivos, escenas, scripts y recursos del proyecto. |
| Viewport | Es la zona de trabajo visual donde construyes la escena. |
| Script | Editor de codigo integrado para GDScript y consulta de API. |

La idea central es esta: en Godot casi todo lo que haces se reduce a editar nodos, sus propiedades y sus relaciones dentro de una escena.

## Crear una primera escena 3D vacia

Para empezar con el pie correcto, no uses una escena 2D si tu objetivo es simulacion 3D.

Pasos recomendados:

1. Crear una nueva escena.
2. Elegir `Node3D` como nodo raiz.
3. Anadir un `Camera3D`.
4. Anadir una `DirectionalLight3D`.
5. Anadir un `MeshInstance3D` con un cubo simple para tener una referencia visual.
6. Guardar la escena como `Main.tscn`.

Con esto ya tienes el esqueleto minimo de una escena 3D visible.

## Configurar la escena principal

Una vez guardada `Main.tscn`, configúrala como escena principal del proyecto para poder lanzar el juego sin pasos extra.

La razon es sencilla: a partir de ahora todos los modulos del manual asumirán que puedes ejecutar el proyecto con una sola accion.

## Primer resultado observable

Al pulsar ejecutar deberias conseguir como minimo lo siguiente:

- Se abre una ventana del proyecto.
- Aparece un cubo o malla basica visible.
- La camara encuadra la escena.
- La luz evita que todo quede negro.

Si eso no ocurre, el problema casi siempre esta en uno de estos puntos:

- La camara no apunta a la escena.
- La malla no existe o no esta visible.
- No hay luz en la escena.
- La escena principal no esta configurada.

## Convenciones de trabajo recomendadas

Empieza con disciplina desde el principio:

- Usa nombres de escenas y nodos descriptivos.
- Guarda pronto y guarda a menudo.
- No metas scripts, escenas y recursos en la raiz del proyecto sin orden.
- Trabaja por iteraciones pequenas: escena visible primero, logica despues.

## Hito del modulo

Al cerrar este modulo deberias tener:

- Godot estable instalado.
- Un proyecto llamado `godot_3d_lab` o equivalente.
- Una escena `Main.tscn` funcional.
- Claridad basica sobre Scene, Inspector, FileSystem y Viewport.

## Que sigue

El siguiente modulo fija el paradigma que domina todo Godot: nodos, escenas, arbol de escena, recursos e instanciacion.

## Referencias oficiales

- Introduccion general: https://docs.godotengine.org/en/stable/about/introduction.html
- Getting Started: https://docs.godotengine.org/en/stable/getting_started/introduction/index.html