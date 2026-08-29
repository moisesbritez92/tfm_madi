#!/usr/bin/env python3
"""Draw the 96x96 observation from a Push-T state, with the training code.

The Godot port replaces the physics engine, not the camera. So condition A of
the demo needs the frame to come out of the very same drawing path that made
the demonstrations: white canvas of 512x512, the goal drawn underneath, pymunk's
overridden ``DrawOptions`` on top, and a bilinear ``cv2.resize`` down to 96.
Anything reimplemented by hand here would change the pixel distribution and
confound the one factor the condition is meant to isolate.

The trick is to keep a live ``PushTImageEnv`` around and never step it. Bodies
are placed by assignment and ``_render_frame`` is called directly, so no physics
runs and the state Godot sent is exactly the state that gets drawn.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Headless by default: the rasteriser draws on an offscreen Surface and never
# opens a window. Set before pygame is imported by the environment module.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

DIFFUSER = Path(__file__).resolve().parents[2]
REPO_ROOT = DIFFUSER / "repo" / "diffusion_policy"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv  # noqa: E402
from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely  # noqa: E402

# Both the evaluation config and the demonstrations use the legacy set_state
# order, so the environment that draws has to be built the same way.
LEGACY = True
RENDER_SIZE = 96
UMBRAL_EXITO = 0.95


class RasterizadorPushT:
    """A Push-T environment used only as a canvas."""

    def __init__(self, render_size: int = RENDER_SIZE, legacy: bool = LEGACY):
        self.env = PushTImageEnv(legacy=legacy, render_size=render_size)
        # reset() is what calls _setup(), which builds the space, the walls, the
        # agent, the block and the goal pose. Without it there is nothing to draw.
        self.env.seed(0)
        self.env.reset()
        self._vaciar_contactos()

    def _vaciar_contactos(self) -> None:
        """Leave the space with no live arbiters, once and for all.

        ``space.debug_draw`` paints the contact points of the last step, and
        ``reset`` samples a random condition that may well start in contact. The
        frames would then carry a few red dots inherited from a condition that
        has nothing to do with the state being drawn, and they would come and go
        between runs. Pulling the bodies apart and stepping twice drops every
        arbiter; from then on nothing is ever stepped, so none is created again.

        The price is that this rasteriser never draws contact points, while the
        real environment does draw them where bodies touch. It is one or two
        pixels of a 96x96 frame, and it is the same one or two every time.
        """
        self.env.agent.position = (60.0, 60.0)
        self.env.block.angle = 0.0
        self.env.block.position = (440.0, 300.0)
        for _ in range(2):
            self.env.space.step(1.0 / self.env.sim_hz)

    def colocar(self, estado) -> None:
        """Place agent and block from a 5-vector, without advancing physics."""
        ax, ay, bx, by, theta = (float(v) for v in estado)
        self.env.agent.position = (ax, ay)
        self.env.agent.velocity = (0.0, 0.0)
        # Angle first, then position. Setting the angle rotates the body about
        # its centre of gravity, which is (0, 45) and not the origin, so doing it
        # afterwards would drag the origin away from (bx, by). The legacy order
        # of _set_state is the opposite, but that belongs to how a state is
        # sampled at reset, not to how a known state is reproduced here.
        self.env.block.angle = theta
        self.env.block.position = (bx, by)
        # The shapes cache their world transform; without this the drawing would
        # lag one assignment behind.
        self.env.space.reindex_shapes_for_body(self.env.block)
        self.env.space.reindex_shapes_for_body(self.env.agent)

    def imagen(self, estado) -> np.ndarray:
        """HWC uint8 RGB frame of ``render_size``, no action marker."""
        self.colocar(estado)
        return self.env._render_frame(mode="rgb_array")

    def cobertura(self, estado) -> float:
        """Reference coverage with shapely, to cross-check the Godot geometry."""
        self.colocar(estado)
        objetivo = self.env._get_goal_pose_body(self.env.goal_pose)
        geom_objetivo = pymunk_to_shapely(objetivo, self.env.block.shapes)
        geom_bloque = pymunk_to_shapely(self.env.block, self.env.block.shapes)
        return float(geom_objetivo.intersection(geom_bloque).area / geom_objetivo.area)


def recompensa(cobertura: float) -> float:
    """Same clip as pusht_env.step: a coverage of 0.95 already scores 1."""
    return float(np.clip(cobertura / UMBRAL_EXITO, 0.0, 1.0))


if __name__ == "__main__":
    # Sanity print: the goal pose drawn on itself must cover exactly 1.
    r = RasterizadorPushT()
    estado_objetivo = [256.0, 400.0, 256.0, 256.0, np.pi / 4]
    print("cobertura en la pose objetivo:", round(r.cobertura(estado_objetivo), 6))
    img = r.imagen(estado_objetivo)
    print("imagen:", img.shape, img.dtype, "min", img.min(), "max", img.max())
