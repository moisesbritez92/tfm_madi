#!/usr/bin/env python3
"""Exercise the policy server without Godot, against the original simulator.

This is verification step 1 and 2 of the plan. It speaks the same protocol the
Godot client will speak, but the environment on this side is the pymunk
``PushTImageEnv``. So if a closed loop does not solve seed 10000 here, the fault
is in the server or in the protocol, and there is no point looking at the port.

It also checks that the initial state ``reset`` returns is the one the
environment samples on its own, which is the whole reason the sampling was not
moved into GDScript.

Usage:
    .venv_diffuser_infer\\Scripts\\python.exe diffuser/godot/servidor/cliente_prueba.py
    ... --seeds 10000 100000 --max-pasos 300
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections import deque
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

DIFFUSER = Path(__file__).resolve().parents[2]
REPO_ROOT = DIFFUSER / "repo" / "diffusion_policy"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv  # noqa: E402

N_OBS_STEPS = 2
N_ACTION_STEPS = 8
MAX_PASOS = 300
UMBRAL_EXITO = 0.999


class Cliente:
    def __init__(self, host="127.0.0.1", puerto=5555):
        self.sock = socket.create_connection((host, puerto))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.canal = self.sock.makefile("rwb")

    def pedir(self, msg):
        self.canal.write(json.dumps(msg).encode("utf-8") + b"\n")
        self.canal.flush()
        linea = self.canal.readline()
        if not linea:
            raise ConnectionError("el servidor cerro la conexion")
        respuesta = json.loads(linea)
        if not respuesta.get("ok"):
            raise RuntimeError(respuesta.get("error", "error sin descripcion"))
        return respuesta

    def cerrar(self):
        try:
            self.pedir({"cmd": "adios"})
        finally:
            self.canal.close()
            self.sock.close()


def estado_de(env) -> list:
    return [
        float(env.agent.position[0]),
        float(env.agent.position[1]),
        float(env.block.position[0]),
        float(env.block.position[1]),
        float(env.block.angle),
    ]


def comprobar_estado_inicial(cliente, seeds):
    """Step 2: reset over the wire must match reset here, exactly."""
    env = PushTImageEnv(legacy=True, render_size=96)
    print("estados iniciales")
    todo_bien = True
    for seed in seeds:
        remoto = cliente.pedir({"cmd": "reset", "seed": seed})["estado0"]
        env.seed(seed)
        env.reset()
        local = estado_de(env)
        delta = float(np.max(np.abs(np.asarray(remoto) - np.asarray(local))))
        estado = "ok" if delta == 0.0 else "DIFIERE"
        todo_bien &= delta == 0.0
        print(f"  seed {seed}: delta {delta:.3e}  {estado}")
    return todo_bien


def episodio(cliente, seed, max_pasos=MAX_PASOS, verboso=True):
    """Step 1: closed loop, pymunk environment, actions from the server."""
    env = PushTImageEnv(legacy=True, render_size=96)
    env.seed(seed)
    env.reset()
    cliente.pedir({"cmd": "reset", "seed": seed})

    # MultiStepWrapper pads the history by repeating the first observation, so
    # the two observation steps coincide on the very first decision.
    historia = deque([estado_de(env)] * N_OBS_STEPS, maxlen=N_OBS_STEPS)

    recompensas = []
    latencias = []
    pasos = 0
    hecho = False
    while pasos < max_pasos and not hecho:
        estados = list(historia)
        respuesta = cliente.pedir({
            "cmd": "act",
            "estado": estados,
            "agent_pos": [[e[0], e[1]] for e in estados],
        })
        latencias.append(respuesta["ms"])
        for accion in respuesta["accion"]:
            _, recompensa, hecho, _ = env.step(np.asarray(accion, dtype=np.float64))
            recompensas.append(float(recompensa))
            historia.append(estado_de(env))
            pasos += 1
            if hecho or pasos >= max_pasos:
                break

    maxima = max(recompensas) if recompensas else 0.0
    exito = maxima >= UMBRAL_EXITO
    if verboso:
        print(
            f"  seed {seed}: recompensa maxima {maxima:.4f} | "
            f"{'exito' if exito else 'fallo'} | {pasos} pasos | "
            f"{len(latencias)} decisiones | mediana {np.median(latencias):.0f} ms"
        )
    return {"seed": seed, "max_reward": maxima, "exito": exito, "pasos": pasos,
            "ms_mediana": float(np.median(latencias)) if latencias else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--puerto", type=int, default=5555)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10000])
    parser.add_argument("--semillas-estado", type=int, nargs="+",
                        default=[100000, 100001, 100002, 100003, 100004])
    parser.add_argument("--max-pasos", type=int, default=MAX_PASOS)
    parser.add_argument("--solo-estado", action="store_true",
                        help="solo la comprobacion de estados iniciales")
    args = parser.parse_args()

    cliente = Cliente(args.host, args.puerto)
    try:
        info = cliente.pedir({"cmd": "hola"})
        print(f"servidor: {info['variante']} | {info['punto_control']} | "
              f"{info['dispositivo']} | obs={info['modo_obs']}")

        ok = comprobar_estado_inicial(cliente, args.semillas_estado)
        if args.solo_estado:
            return 0 if ok else 1

        print("episodios en bucle cerrado (simulador original)")
        filas = [episodio(cliente, s, args.max_pasos) for s in args.seeds]
        exitos = sum(1 for f in filas if f["exito"])
        print(f"resumen: {exitos}/{len(filas)} con exito | "
              f"media {np.mean([f['max_reward'] for f in filas]):.4f}")
        return 0 if ok else 1
    finally:
        cliente.cerrar()


if __name__ == "__main__":
    raise SystemExit(main())
