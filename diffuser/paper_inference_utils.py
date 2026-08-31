"""Inferencia sobre el punto de control publicado por los autores del articulo.

No es una variante del experimento: su encoder es el de robomimic (ResNet-18 +
SpatialSoftmax de 32 keypoints + Linear, con recorte 84x84), mientras que V0-V4
usan ``MultiImageObsEncoder``. La politica es ``DiffusionUnetHybridImagePolicy``
y necesita ``robomimic`` instalado en el entorno.

El nombre del modulo no sigue el patron ``v{n}_inference_utils`` a proposito:
``godot/servidor/servidor_politica.py`` resuelve las variantes por ese patron y
la demo no debe mezclar el modelo del articulo con las cinco del TFM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from v0_inference_utils import (
    default_checkpoint as _default_checkpoint_v0,
    list_checkpoints as _list_checkpoints,
    load_policy_bundle,
    make_env,
    plot_rollout,
    resolve_device,
    run_rollout as _run_rollout,
    save_gif,
)

THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR / "models" / "V_Paper"
ARTIFACT_DIR = THIS_DIR / "artifacts" / "paper_inference"
PREFERRED_CHECKPOINT = "epoch=0500-test_mean_score=0.884.ckpt"


def list_checkpoints(model_dir: Path = MODEL_DIR) -> list[Path]:
    return _list_checkpoints(model_dir=model_dir)


def default_checkpoint(model_dir: Path = MODEL_DIR) -> Path:
    return _default_checkpoint_v0(
        model_dir=model_dir,
        preferred_checkpoint=PREFERRED_CHECKPOINT,
    )


def run_rollout(
    checkpoint_path: Optional[Path] = None,
    seed: int = 10049,
    device: Optional[Any] = None,
    max_decisions: Optional[int] = None,
    render_size: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> dict[str, Any]:
    return _run_rollout(
        checkpoint_path=checkpoint_path,
        seed=seed,
        device=device,
        max_decisions=max_decisions,
        render_size=render_size,
        max_steps=max_steps,
        model_dir=MODEL_DIR,
        artifact_dir=ARTIFACT_DIR,
        preferred_checkpoint=PREFERRED_CHECKPOINT,
    )


def smoke_test() -> dict[str, Any]:
    result = run_rollout(default_checkpoint(), max_decisions=1)
    return {
        "checkpoint": result["checkpoint_name"],
        "device": result["device"],
        "frames": len(result["frames"]),
        "env_steps": result["n_env_steps"],
        "max_reward": result["max_reward"],
    }


if __name__ == "__main__":
    print(smoke_test())
