# 02. Fundamentos: Nodos, Escenas y Editor

## Objetivo del modulo

Entender el modelo mental correcto de Godot. Si esta parte queda floja, el resto del trabajo en 3D, fisica o scripting se vuelve confuso muy rapido.

## La idea central de Godot

Godot organiza el desarrollo alrededor de dos conceptos: `nodo` y `escena`.

- Un nodo representa una pieza de comportamiento, transformacion, visualizacion, audio, interfaz o logica.
- Una escena es una composicion de nodos organizada en forma jerarquica.

La escena no es solo una pantalla. Puede ser cualquier unidad reutilizable:

- un personaje,
- una camara,
- una puerta,
- una interfaz,
- un nivel,
- un enemigo,
- o un mundo completo.

Este enfoque es importante porque te obliga a pensar en composicion y reutilizacion, no en archivos monoliticos.

## Nodo, escena y arbol de escena

Conviene distinguir tres niveles:

| Concepto | Significado practico |
|---|---|
| Nodo | Unidad individual con propiedades y comportamiento. |
| Escena | Conjunto guardable de nodos que forman una unidad reutilizable. |
| Scene Tree | Arbol jerarquico que existe en ejecucion y organiza todo lo activo. |

Una escena guardada en disco puede instanciarse dentro de otra. Esa idea de composicion es uno de los pilares del motor.

## Ejemplo minimo de escena 3D

Una escena inicial razonable para 3D podria verse asi:

```text
Main (Node3D)
|- WorldLight (DirectionalLight3D)
|- CameraPivot (Node3D)
|  `- Camera3D
`- Floor (MeshInstance3D)
```

De esta jerarquia ya puedes sacar varias conclusiones utiles:

- `Main` actua como contenedor raiz.
- La camara puede colgar de un pivote para rotar o desplazarse sin tocar directamente la camara.
- El suelo es otro nodo independiente con su propia transformacion.

## Tipos de nodos que debes reconocer pronto

| Nodo | Uso habitual |
|---|---|
| `Node` | Contenedor generico sin transformacion espacial. |
| `Node3D` | Base para nodos con posicion, rotacion y escala en 3D. |
| `Camera3D` | Vista de la escena. |
| `DirectionalLight3D` | Luz direccional global. |
| `MeshInstance3D` | Geometria visible en 3D. |
| `CharacterBody3D` | Personaje controlado con logica cinemática. |
| `RigidBody3D` | Objeto gobernado por la fisica. |
| `CollisionShape3D` | Forma de colision usada por cuerpos fisicos. |
| `Area3D` | Zona de deteccion o trigger. |
| `Timer` | Eventos temporizados. |

No hace falta memorizar todo de golpe. Lo importante es entender que el tipo de nodo condiciona su papel dentro de la escena.

## Propiedades y herencia practica

Cada nodo tiene propiedades editables en el Inspector. En 3D, las mas criticas al principio suelen ser:

- Posicion.
- Rotacion.
- Escala.
- Visibilidad.
- Recursos asociados como mallas, materiales o scripts.

En la practica, editar propiedades en el Inspector es tan importante como escribir codigo. La mitad del flujo de trabajo en Godot consiste en combinar configuracion visual y scripting.

## Recursos

Un recurso es un dato reutilizable que vive en el proyecto y que un nodo puede consumir. Ejemplos habituales:

- una malla,
- un material,
- una textura,
- un script,
- una animacion,
- o una forma de colision.

La regla practica es esta: un nodo define comportamiento y presencia en la escena; un recurso define datos configurables que pueden compartirse.

## Escenas reutilizables e instanciacion

La instanciacion es una capacidad clave. Permite crear una escena una vez y reutilizarla varias veces dentro de otras escenas.

Ejemplos tipicos:

- una escena `Enemy.tscn` instanciada varias veces en un nivel,
- una escena `Door.tscn` usada en distintas habitaciones,
- una escena `Pickup.tscn` repetida por todo el entorno.

Este principio evita duplicar trabajo y te acerca a una estructura mantenible.

## El editor como herramienta de composicion

No conviene ver el editor como un sitio donde solo colocas objetos. En Godot el editor es una parte central del desarrollo.

### Scene

Es donde construyes la jerarquia. Casi cualquier cambio importante en estructura empieza aqui.

### Inspector

Es donde ajustas propiedades del nodo actual. Si un objeto no se ve, no colisiona o no responde, el Inspector suele ser el primer lugar donde mirar.

### FileSystem

Es la vista del proyecto. Aqui organizas escenas, scripts, materiales y recursos. Mantener ordenado este panel te ahorra problemas posteriores.

### Viewport

Es el espacio de construccion visual. En 3D importa especialmente aprender a moverte por el viewport con soltura.

## Modelo mental recomendado

Si vienes de otras herramientas, adopta esta secuencia mental:

1. Piensa que objeto quieres en la escena.
2. Elige el tipo de nodo correcto.
3. Decide si sera una escena independiente reutilizable.
4. Ajusta sus propiedades en el Inspector.
5. Solo despues añade script si hace falta comportamiento dinamico.

Muchos errores vienen de escribir codigo demasiado pronto para resolver problemas que en realidad eran de jerarquia o configuracion.

## Ejercicio guiado

Construye una escena base con esta estructura:

```text
Main (Node3D)
|- Camera3D
|- DirectionalLight3D
`- Cube (MeshInstance3D)
```

Tareas:

1. Cambia el nombre del cubo a `ReferenceCube`.
2. Mueve la camara hasta que el cubo se vea con claridad.
3. Rota la luz para cambiar la sombra o la iluminacion.
4. Duplica el cubo y cambia su posicion para confirmar que entiendes la diferencia entre nodo y recurso visual.

## Errores tipicos

- Crear escenas demasiado grandes desde el primer dia.
- No guardar escenas reutilizables por separado.
- Poner logica compleja en cualquier nodo sin pensar si ese nodo es el adecuado.
- Confundir un cambio de estructura con un problema de script.

## Hito del modulo

Al terminar este modulo deberias poder explicar con tus palabras:

- que es un nodo,
- que es una escena,
- como se relacionan dentro del Scene Tree,
- cuando conviene separar algo como escena propia,
- y donde tocar propiedades dentro del editor.

## Que sigue

El siguiente modulo entra en GDScript, input y senales. A partir de ahi el proyecto deja de ser solo una escena visible y empieza a reaccionar.

## Referencias oficiales

- Step by step: https://docs.godotengine.org/en/stable/getting_started/step_by_step/index.html
- Nodes and Scenes: https://docs.godotengine.org/en/stable/getting_started/step_by_step/nodes_and_scenes.html
- Instancing: https://docs.godotengine.org/en/stable/getting_started/step_by_step/instancing.html