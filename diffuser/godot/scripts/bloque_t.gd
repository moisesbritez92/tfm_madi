class_name BloqueT
extends RigidBody2D

## La pieza en T, con la amortiguacion total del entorno original.
##
## `space.damping = 0` (`pusht_env.py:291`) no es un rozamiento suave: en
## Chipmunk la amortiguacion es la fraccion de velocidad que se conserva por
## segundo, de modo que `v *= 0 ^ dt` anula la velocidad al principio de cada
## subpaso. La pieza solo se mueve por el impulso de colision generado dentro de
## ese mismo subpaso, y nada desliza por inercia.
##
## Sin esto la pieza de Godot conserva la inercia, sigue viajando despues de que
## el agente la suelte, y la tarea deja de ser la tarea. Es el detalle que mas
## facil se pasa por alto de todo el port.
##
## Anular la velocidad en `_integrate_forces` da el mismo resultado neto que
## anularla al principio del paso: la velocidad con la que se integra la posicion
## es, en ambos casos, solo la que los impulsos de ese paso han producido.


func _integrate_forces(estado: PhysicsDirectBodyState2D) -> void:
	estado.linear_velocity = Vector2.ZERO
	estado.angular_velocity = 0.0
