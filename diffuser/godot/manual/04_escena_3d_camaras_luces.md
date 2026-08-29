# 04. Escena 3D, Camaras y Luces

## Objetivo del modulo

Pasar de una escena apenas visible a una escena 3D que puedas recorrer, entender espacialmente y usar como base para los siguientes modulos de fisica, assets e interaccion.

## Lo primero que cambia al trabajar en 3D

En 3D ya no basta con colocar elementos en una superficie. Ahora importan:

- la escala,
- la orientacion,
- la relacion entre ejes,
- la camara desde la que observas,
- y la iluminacion que permite leer la escena.

Si esto no queda bien montado desde el principio, mas adelante sera dificil saber si un problema viene del entorno, del script o de la fisica.

## Sistema de coordenadas de Godot

Godot usa un sistema de coordenadas diestro con estas convenciones practicas:

- `X`: izquierda y derecha.
- `Y`: arriba y abajo.
- `Z`: profundidad.

Dos recordatorios importantes:

- `Y` es el eje vertical.
- Para muchos nodos orientados, `-Z` actua como direccion de avance visible.

Ademas, Godot trabaja en escala metrica: `1 unidad = 1 metro`.

Esto no es un detalle menor. Si construyes escenas fuera de escala, tarde o temprano tendras friccion con fisica, camaras o assets importados.

## Navegacion del viewport 3D

Antes de seguir, merece la pena dominar tres habitos:

1. Orbitar alrededor de la escena.
2. Desplazar la vista.
3. Entrar en modo de libre navegacion para inspeccionar rapido.

Atajos utiles al inicio:

- `Q`: seleccion.
- `W`: mover.
- `E`: rotar.
- `R`: escalar.
- `F`: centrar la vista en el objeto seleccionado.

En el viewport 3D, moverte con soltura te ahorra mucho tiempo. Si vienes de Blender, Godot ofrece una experiencia cercana y permite ajustar el esquema de navegacion en la configuracion del editor.

## Node3D y transformaciones

Todo objeto 3D relevante cuelga de `Node3D` o de una clase derivada.

Cada `Node3D` tiene una transformacion local compuesta por:

- posicion,
- rotacion,
- escala.

La clave es que esa transformacion es relativa al padre. Por tanto, cuando cambias la posicion de un nodo contenedor, arrastras consigo a sus hijos.

Eso te conviene para:

- pivotes de camara,
- personajes con varios componentes,
- entornos con piezas agrupadas,
- o conjuntos de objetos que quieras mover como bloque.

## Advertencia sobre rotaciones

En Godot conviene no abusar de la propiedad `rotation` como si el 3D fuese solo 2D con un eje extra.

Para edicion simple en el Inspector no hay problema. Pero cuando programes comportamiento 3D sostenido, el modelo mental correcto pasa por transformaciones, vectores y ejes locales, no por encadenar angulos sin control.

Regla practica para esta fase:

- usa el Inspector para orientar objetos simples,
- usa nodos pivote para separar movimientos,
- y cuando programes camaras o personajes, piensa en ejes locales y direccion de avance.

## Construir una escena base mas util

La escena 3D minima recomendable para este manual ya no es solo un cubo flotando. Debe incluir una superficie, una camara y una luz con las que puedas leer profundidad y escala.

Estructura sugerida:

```text
Main (Node3D)
|- Ground (StaticBody3D)
|  |- CollisionShape3D
|  `- MeshInstance3D
|- WorldLight (DirectionalLight3D)
|- Environment (WorldEnvironment)
`- CameraRig (Node3D)
   `- Camera3D
```

## Suelo: version visible y version fisica

Una leccion importante aparece muy pronto: lo visible y lo fisico no son necesariamente lo mismo, aunque muchas veces deban ir juntos.

Para el suelo:

- `StaticBody3D` aporta colision estable.
- `CollisionShape3D` define la forma fisica.
- `MeshInstance3D` aporta la parte visible.

Una configuracion sencilla y robusta es usar una caja ancha y baja para ambas partes.

## Luces y lectura de escena

Si no anades luz real a la escena, puedes engañarte con las luces previas del editor. Eso funciona para editar, pero no representa el resultado al ejecutar el proyecto.

Dos ideas practicas:

- usa una `DirectionalLight3D` para iluminar la escena base,
- y activa sombras si quieres empezar a leer volumen y separacion entre objetos.

La previsualizacion del editor puede mostrar un entorno o una luz provisionales, pero esos elementos no sustituyen a una configuracion real cuando ejecutas la escena.

## WorldEnvironment

Aunque no sea estrictamente obligatorio en el primer minuto, `WorldEnvironment` mejora mucho la legibilidad del prototipo.

Te interesa especialmente para:

- definir un color o cielo de fondo,
- evitar escenas negras o lavadas,
- y acostumbrarte a que el entorno visual tambien forma parte de la configuracion del mundo.

No hace falta entrar aqui en postprocesado avanzado. Con un entorno simple basta.

## Camara: primera configuracion razonable

En 3D no se ve nada si no hay una `Camera3D` activa.

Configuracion minima recomendada:

- una `Camera3D` colgando de un `CameraRig` o pivote,
- colocada a cierta altura y distancia del suelo,
- encuadrando el centro de la escena.

Usar un pivote aunque parezca innecesario ahora te ayuda despues si quieres:

- girar alrededor de un punto,
- seguir a un personaje,
- separar yaw y pitch,
- o cambiar la logica de la camara sin rehacer la jerarquia.

## Hito practico: escena 3D navegable

Construye esta escena con pasos deliberados:

1. En `Main`, crea un `Ground` basado en `StaticBody3D`.
2. Anade una `CollisionShape3D` y asigna una `BoxShape3D` amplia.
3. Anade una `MeshInstance3D` con una malla de caja o plano visible.
4. Baja ligeramente el suelo para dejar visible la cuadricula del editor si te ayuda a orientarte.
5. Anade una `DirectionalLight3D` y orientala hasta que el suelo reciba luz clara.
6. Anade una `WorldEnvironment` simple.
7. Crea `CameraRig` y cuelga de el una `Camera3D`.
8. Coloca la camara de modo que puedas ver el suelo y varios objetos de referencia.
9. Anade dos o tres cubos mas para leer escala y profundidad.

## Resultado esperado

Al ejecutar la escena deberias tener:

- un suelo visible,
- una camara encuadrando el entorno,
- una luz real funcionando,
- y suficiente geometria para percibir profundidad.

Si la escena se ve mal, revisa en este orden:

1. existencia de una `Camera3D` activa,
2. direccion y presencia de la luz,
3. escala y posicion de los objetos,
4. si estas confundiendo vista del editor con resultado de ejecucion.

## Mini practica recomendada

Sin escribir aun el controlador definitivo, crea una escena con varios cubos a distintas distancias y mueve la camara manualmente desde el editor para responder a estas preguntas:

- que tan grande parece un objeto de 1 metro,
- a que altura te gusta situar la camara para inspeccion,
- y que configuracion base te resulta mas legible para trabajar despues.

Esto parece trivial, pero fija un criterio espacial util para todo el manual.

## Errores tipicos

- Trabajar sin `Camera3D` real porque en el editor todo parecia verse bien.
- Usar escalas arbitrarias que luego rompen intuicion de fisica.
- Poner todos los nodos al mismo nivel sin pivotes ni jerarquia.
- Querer resolver con scripts un problema de escena mal montada.

## Hito del modulo

Al cerrar este modulo deberias tener una escena 3D base que puedas reutilizar como laboratorio: suelo fisico, elementos visibles, camara, luz y entorno minimamente coherente.

## Que sigue

El siguiente bloque convierte este laboratorio en un entorno interactivo: cuerpos fisicos, colisiones y movimiento controlado.

## Referencias oficiales

- Introduccion a 3D: https://docs.godotengine.org/en/stable/tutorials/3d/introduction_to_3d.html
- Using 3D transforms: https://docs.godotengine.org/en/stable/tutorials/3d/using_transforms.html
- First 3D game, setup: https://docs.godotengine.org/en/stable/getting_started/first_3d_game/01.game_setup.html