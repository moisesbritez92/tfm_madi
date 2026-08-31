#!/usr/bin/env python3
"""TCP policy server: Godot simulates, this answers with actions.

The demo puts a frozen checkpoint of the experiment -- V0 by default, any of the
five with ``--variante`` -- in the loop of a Push-T reimplemented in Godot 4.
Godot owns the environment; this process owns the policy and nothing
else about the simulation, save one detail it cannot delegate: the initial
state. Reproducing the legacy Mersenne Twister of numpy in GDScript would be a
pointless risk, so ``reset`` samples the condition with the real ``PushTEnv``
and hands Godot the five numbers that come out.

Two observation modes, one flag, no change on the Godot side:

  --obs estado   condition A. Godot sends the state, the frame is drawn here by
                 rasterizador_pusht with the same code that produced the
                 demonstrations. Only the physics engine has changed.
  --obs godot    condition B. Godot sends its own 96x96 render. Both the
                 physics and the pixels have changed.

Nothing measured here is a result. The scores this makes visible are
illustrative; the numbers of the memoir are the ones in
``logs_entrenamiento/prueba_final/``.

Usage (Windows, from the repository root):
    .venv_diffuser_infer\\Scripts\\python.exe diffuser/godot/servidor/servidor_politica.py --obs estado
    ... servidor_politica.py --variante v3 --obs godot --puerto 5556
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

DIFFUSER = Path(__file__).resolve().parents[2]
if str(DIFFUSER) not in sys.path:
    sys.path.insert(0, str(DIFFUSER))

from rasterizador_pusht import RasterizadorPushT, recompensa  # noqa: E402

PUERTO = 5555
HOST = "127.0.0.1"
# Same base as the preregistered pass, so a seed always yields the same video.
BASE_SEED_DIFUSION = 20260827

VARIANTES = ("v0", "v1", "v2", "v3", "v4")


def cargar_politica(variante="v0", checkpoint=None, dispositivo=None):
    """Rebuild a frozen checkpoint, reusing the loader of the notebooks.

    Only ``v0_inference_utils`` exports ``load_policy_bundle``; the other four
    modules are thin shims that redefine three constants and re-export the rest.
    So the checkpoint and the artifact directory come from the module of the
    variant, and the loader is always the one of V0. It is generic: it reads the
    config out of the checkpoint, and it already sets ``rgb_model.pretrained =
    False`` when the encoder target lives in ``pretrained_encoders``, which is
    exactly the case of the DINOv2 of V3 and the CLIP of V4. Nothing is
    downloaded.

    Passing the artifact directory of the variant keeps V3 from writing under
    ``artifacts/v0_inference/``, which is where a previous session left four
    stray folders.
    """
    import importlib

    import v0_inference_utils as v0

    modulo = importlib.import_module(f"{variante}_inference_utils")
    ruta = Path(checkpoint) if checkpoint else modulo.default_checkpoint()
    bundle = v0.load_policy_bundle(
        ruta, device=dispositivo, artifact_dir=modulo.ARTIFACT_DIR
    )
    return bundle, ruta


class Sesion:
    """State of one episode: the diffusion noise counter."""

    def __init__(self, base_seed: int):
        self.base_seed = base_seed
        self.seed_episodio = 0
        self.paso = 0

    def nuevo_episodio(self, seed: int) -> None:
        self.seed_episodio = int(seed)
        self.paso = 0

    def sembrar(self) -> None:
        """Common random numbers, same scheme as evaluar_bloque_test.py."""
        import torch

        torch.manual_seed(
            self.base_seed * 1000003 + self.seed_episodio * 1000 + self.paso
        )
        self.paso += 1


def png_a_rgb(b64: str) -> np.ndarray:
    """Decode one base64 PNG from Godot into HWC uint8 RGB."""
    from PIL import Image

    crudo = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(crudo)).convert("RGB"), dtype=np.uint8)


def construir_obs(imagenes, agent_pos, dispositivo):
    """Pack the observation exactly as training did.

    image     (1, 2, 3, 96, 96) float32 in [0, 1], CHW, RGB
    agent_pos (1, 2, 2)         float32 in world pixels 0-512, not 0-96
    """
    import torch

    pila = np.stack(imagenes, axis=0).astype(np.float32) / 255.0   # (2, 96, 96, 3)
    pila = np.moveaxis(pila, -1, 1)                                # (2, 3, 96, 96)
    pos = np.asarray(agent_pos, dtype=np.float32)                  # (2, 2)
    return {
        "image": torch.from_numpy(pila).unsqueeze(0).to(dispositivo),
        "agent_pos": torch.from_numpy(pos).unsqueeze(0).to(dispositivo),
    }


class Servidor:
    def __init__(self, args):
        self.modo_obs = args.obs
        self.rasterizador = RasterizadorPushT()
        self.sesion = Sesion(args.base_seed)
        self._aviso_dado = False

        self.variante = args.variante
        print(f"cargando {self.variante.upper()} ...", flush=True)
        bundle, ruta = cargar_politica(self.variante, args.checkpoint, args.dispositivo)
        self.politica = bundle["policy"]
        self.dispositivo = bundle["device"]
        self.ckpt = ruta.name
        print(f"{self.variante.upper()} listo | {self.ckpt} | {self.dispositivo} | "
              f"obs={self.modo_obs}", flush=True)

        # A second environment, used only to sample initial conditions. Keeping
        # it apart from the rasteriser avoids one command disturbing the other.
        from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

        self.muestreador = PushTImageEnv(legacy=True, render_size=96)

    # ---- commands -------------------------------------------------------

    def hola(self, _):
        return {
            "ok": True,
            "variante": self.variante.upper(),
            "punto_control": self.ckpt,
            "modo_obs": self.modo_obs,
            "dispositivo": str(self.dispositivo),
        }

    def reset(self, msg):
        seed = int(msg["seed"])
        self.muestreador.seed(seed)
        self.muestreador.reset()
        estado = [
            float(self.muestreador.agent.position[0]),
            float(self.muestreador.agent.position[1]),
            float(self.muestreador.block.position[0]),
            float(self.muestreador.block.position[1]),
            float(self.muestreador.block.angle),
        ]
        self.sesion.nuevo_episodio(seed)
        if hasattr(self.politica, "reset"):
            self.politica.reset()
        return {"ok": True, "estado0": estado}

    def act(self, msg):
        import torch

        agent_pos = msg["agent_pos"]
        if self.modo_obs == "estado":
            imagenes = [self.rasterizador.imagen(e) for e in msg["estado"]]
        else:
            imagenes = [png_a_rgb(b) for b in msg["imagen"]]
            for img in imagenes:
                if img.shape != (96, 96, 3):
                    raise ValueError(
                        f"la imagen de Godot es {img.shape}, se esperaba (96, 96, 3)"
                    )
            self._avisar_historial_plano(imagenes, msg.get("estado"))

        obs = construir_obs(imagenes, agent_pos, self.dispositivo)
        self.sesion.sembrar()
        t0 = time.perf_counter()
        with torch.no_grad():
            salida = self.politica.predict_action(obs)
        if self.dispositivo.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0

        accion = salida["action"][0].detach().cpu().numpy()   # (8, 2), pixels 0-512
        return {"ok": True, "accion": accion.tolist(), "ms": round(ms, 1)}

    def _avisar_historial_plano(self, imagenes, estados) -> None:
        """Avisa si las dos observaciones son iguales pero los estados no.

        Los callbacks de proceso de Godot corren antes del dibujado, de modo que
        un descuido en la captura hace que la pose historica se sobrescriba por
        la actual y las dos imagenes salgan identicas. La politica veria un
        historial congelado y seguiria funcionando, algo peor: funcionaria un
        poco peor y sin sintoma. Este aviso es barato y sale una sola vez.
        """
        if self._aviso_dado or estados is None or len(imagenes) < 2:
            return
        estados_iguales = np.allclose(
            np.asarray(estados[0], dtype=np.float64),
            np.asarray(estados[1], dtype=np.float64),
        )
        if estados_iguales or not np.array_equal(imagenes[0], imagenes[1]):
            return
        self._aviso_dado = True
        print(
            "AVISO: las dos observaciones son identicas pese a que los estados "
            "difieren.\n       La captura de Godot esta tomando dos veces la pose "
            "actual; revisa el\n       cerrojo de vista3d.observacion_base64.",
            flush=True,
        )

    def cobertura(self, msg):
        """Reference coverage, to cross-check the geometry port of Godot."""
        cob = self.rasterizador.cobertura(msg["estado"])
        return {"ok": True, "cobertura": cob, "recompensa": recompensa(cob)}

    def adios(self, _):
        return {"ok": True, "adios": True}

    # ---- loop -----------------------------------------------------------

    def despachar(self, msg):
        cmd = msg.get("cmd")
        manejador = {
            "hola": self.hola,
            "reset": self.reset,
            "act": self.act,
            "cobertura": self.cobertura,
            "adios": self.adios,
        }.get(cmd)
        if manejador is None:
            return {"ok": False, "error": f"comando desconocido: {cmd!r}"}
        return manejador(msg)

    def servir(self, host: str, puerto: int):
        escucha = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        escucha.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        escucha.bind((host, puerto))
        escucha.listen(1)
        print(f"escuchando en {host}:{puerto} | Ctrl-C para parar", flush=True)
        try:
            while True:
                cliente, direccion = escucha.accept()
                print(f"conectado {direccion}", flush=True)
                self.atender(cliente)
                print("cliente desconectado", flush=True)
        except KeyboardInterrupt:
            print("\nparando")
        finally:
            escucha.close()

    def atender(self, cliente: socket.socket):
        # TCP_NODELAY matters: the messages are small and each one blocks the
        # simulation until it is answered.
        cliente.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        canal = cliente.makefile("rwb")
        try:
            self._conversar(canal)
        except (ConnectionError, OSError) as error:
            # Un cliente que se va a mitad de una respuesta no es motivo para
            # tirar el servidor: cargar V0 cuesta medio minuto y una sesion de
            # trabajo encadena varias ejecuciones de Godot.
            print(f"conexion interrumpida: {type(error).__name__}: {error}", flush=True)
        finally:
            canal.close()
            cliente.close()

    def _conversar(self, canal) -> None:
        for linea in canal:
            if not linea.strip():
                continue
            try:
                msg = json.loads(linea)
                respuesta = self.despachar(msg)
            except Exception as error:   # a bad frame must not kill the server
                respuesta = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            canal.write(json.dumps(respuesta).encode("utf-8") + b"\n")
            canal.flush()
            if respuesta.get("adios"):
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs", choices=("estado", "godot"), default="estado",
                        help="estado: la imagen se dibuja aqui (condicion A). "
                             "godot: la imagen llega renderizada por Godot (condicion B)")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--puerto", type=int, default=PUERTO)
    parser.add_argument("--variante", choices=VARIANTES, default="v0",
                        help="que punto de control congelado se pone en el bucle")
    parser.add_argument("--checkpoint", default=None,
                        help="por defecto, el punto de control congelado de la variante")
    parser.add_argument("--dispositivo", default=None)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED_DIFUSION)
    args = parser.parse_args()

    Servidor(args).servir(args.host, args.puerto)


if __name__ == "__main__":
    main()
