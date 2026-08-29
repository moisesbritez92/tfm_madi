# Manual de Godot para 3D y Simulacion

## Objetivo

Este manual empezó como una guía para aprender Godot con un fin práctico:
construir escenas 3D, entender el flujo del editor y preparar el terreno para
usar el motor como entorno de prototipado dentro del TFM.

Ese terreno ya está usado. La carpeta que contiene este manual **es** un proyecto
Godot: una reimplementación de Push-T sobre Godot Physics 2D, con vista 3D
cenital, en la que el punto de control congelado de V0 cierra el bucle a través
de un servidor de política en Python. Los módulos 01 a 05 siguen siendo el camino
de aprendizaje; los módulos 06 y 07 documentan lo que se construyó con él.

No intenta reemplazar la documentación oficial. Su función es ordenar el camino,
reducir ruido y dejar por escrito las decisiones que costaron tiempo.

## Cómo usar este manual

Si vienes a aprender Godot, la ruta es lineal y empieza en el módulo 01. Si
vienes a entender o a tocar la demostración de Push-T, ve directo al 06.

1. [01_introduccion_y_setup.md](01_introduccion_y_setup.md), y deja Godot
   instalado con un proyecto 3D vacío funcionando.
2. [02_fundamentos_nodos_escenas_editor.md](02_fundamentos_nodos_escenas_editor.md),
   para entender el paradigma central del motor.
3. [03_gdscript_input_senales.md](03_gdscript_input_senales.md), para poder
   escribir y conectar lógica.
4. Después, el bloque 3D y de física.

## Ruta

| Módulo | Estado | Objetivo |
|---|---|---|
| [01_introduccion_y_setup.md](01_introduccion_y_setup.md) | Implementado | Instalar Godot, crear el primer proyecto y entender el flujo básico del editor. |
| [02_fundamentos_nodos_escenas_editor.md](02_fundamentos_nodos_escenas_editor.md) | Implementado | Dominar nodos, escenas, árbol de escena, recursos e instanciación. |
| [03_gdscript_input_senales.md](03_gdscript_input_senales.md) | Implementado | Empezar a programar en GDScript y conectar comportamiento con input y señales. |
| [04_escena_3d_camaras_luces.md](04_escena_3d_camaras_luces.md) | Implementado | Construir una escena 3D mínima navegable. |
| [05_fisica_colisiones_movimiento.md](05_fisica_colisiones_movimiento.md) | Implementado | Introducir CharacterBody3D, RigidBody3D y colisiones. |
| [06_puente_python_godot.md](06_puente_python_godot.md) | Implementado | El protocolo con el servidor de política, el port de la física de Push-T y sus cuatro trampas. |
| [07_escena_pusht.md](07_escena_pusht.md) | Implementado | La escena, los modos de ejecución y cómo se lanza la demostración. |

Los módulos que la primera versión dejaba pendientes —assets, navegación,
proyecto integrador— se han quedado sin objeto: el proyecto integrador acabó
siendo Push-T, y ni los assets importados ni la navegación entraban en él.

## Qué hay fuera de `manual/`

```
diffuser/godot/
├── project.godot        physics_ticks_per_second = 100
├── lanzar.ps1           arranca servidor y Godot, y limpia al salir
├── escenas/Main.tscn    raíz mínima; la escena se construye por código
├── scripts/             GDScript: física, cobertura, cliente, vista 3D, panel
├── servidor/            Python: servidor de política, rasterizador, verificación
└── grabaciones/         trayectorias y capturas (ignorado por git)
```

## Resultado esperado

Al terminar los cinco primeros módulos deberías poder crear un proyecto Godot 4
estable, construir y navegar una escena 3D, organizar la lógica en nodos y
escenas, escribir GDScript básico, capturar input, usar señales y montar un
prototipo con colisiones y movimiento.

Al terminar los dos últimos deberías poder lanzar la demostración, saber qué se
está enseñando y qué no, y reproducir las cuatro comprobaciones que sostienen el
port.

## Referencia base

Usa siempre la rama `stable` de la documentación oficial, para no mezclar
funciones o pantallas de versiones en desarrollo:

- https://docs.godotengine.org/en/stable/about/introduction.html
- https://docs.godotengine.org/en/stable/getting_started/step_by_step/index.html
- https://docs.godotengine.org/en/stable/getting_started/first_3d_game/index.html
