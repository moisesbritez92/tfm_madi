# Manual de Godot para 3D y Simulacion

## Objetivo

Este manual esta pensado como una guia de trabajo para aprender Godot con un fin practico: construir escenas 3D, entender el flujo del editor y preparar el terreno para usar el motor como entorno de prototipado dentro del TFM.

No intenta reemplazar la documentacion oficial. Su funcion es ordenar el camino de aprendizaje, reducir ruido y dejar claro que estudiar primero para llegar antes a un entorno 3D funcional.

## Alcance de la v1

Incluido en esta primera version:

- Instalacion y puesta en marcha.
- Editor, nodos, escenas y organizacion del proyecto.
- GDScript basico, input y senales.
- Escena 3D, camaras, luces y nociones espaciales.
- Fisica basica, colisiones y movimiento.
- Importacion de assets y organizacion del entorno.
- Navegacion e interaccion sencilla.
- Proyecto integrador pequeno.

Fuera de esta primera version:

- Shaders y rendering avanzado.
- Networking y multijugador.
- XR.
- Optimizacion profunda.
- Plugins o extensiones de editor.
- Exportacion multiplataforma detallada.

## Como usar este manual

La ruta recomendada es lineal. Cada modulo prepara el siguiente.

1. Leer [01_introduccion_y_setup.md](01_introduccion_y_setup.md) y dejar Godot instalado y un proyecto 3D vacio funcionando.
2. Completar [02_fundamentos_nodos_escenas_editor.md](02_fundamentos_nodos_escenas_editor.md) para entender el paradigma central del motor.
3. Completar [03_gdscript_input_senales.md](03_gdscript_input_senales.md) para poder escribir y conectar logica.
4. Pasar despues al bloque 3D, fisica e integracion.

## Ruta de estudio

| Modulo | Estado | Objetivo |
|---|---|---|
| [01_introduccion_y_setup.md](01_introduccion_y_setup.md) | Implementado | Instalar Godot, crear el primer proyecto y entender el flujo basico del editor. |
| [02_fundamentos_nodos_escenas_editor.md](02_fundamentos_nodos_escenas_editor.md) | Implementado | Dominar nodos, escenas, arbol de escena, recursos e instanciacion. |
| [03_gdscript_input_senales.md](03_gdscript_input_senales.md) | Implementado | Empezar a programar en GDScript y conectar comportamiento con input y senales. |
| [04_escena_3d_camaras_luces.md](04_escena_3d_camaras_luces.md) | Implementado | Construir una escena 3D minima navegable. |
| [05_fisica_colisiones_movimiento.md](05_fisica_colisiones_movimiento.md) | Implementado | Introducir CharacterBody3D, RigidBody3D y colisiones. |
| 06_assets_y_organizacion.md | Pendiente | Importar assets y ordenar el entorno 3D. |
| 07_navegacion_e_interaccion.md | Pendiente | NavigationMesh, agentes y triggers simples. |
| 08_proyecto_integrador.md | Pendiente | Consolidar el flujo completo en un mini proyecto. |
| 09_siguientes_pasos.md | Pendiente | Marcar rutas de ampliacion segun el TFM. |

## Resultado esperado al terminar la v1

Al finalizar este manual deberias poder:

- Crear un proyecto Godot 4 estable desde cero.
- Construir una escena 3D simple y navegar por ella.
- Entender como se organiza la logica en nodos y escenas.
- Escribir scripts basicos en GDScript.
- Capturar input, lanzar eventos y usar senales.
- Montar un prototipo 3D con colisiones, movimiento e interaccion elemental.

## Referencia base

La referencia principal es la documentacion oficial estable de Godot:

- https://docs.godotengine.org/en/stable/about/introduction.html
- https://docs.godotengine.org/en/stable/getting_started/step_by_step/index.html
- https://docs.godotengine.org/en/stable/getting_started/first_3d_game/index.html

Usa siempre la rama `stable` de la documentacion para evitar mezclar funciones o pantallas de versiones en desarrollo.