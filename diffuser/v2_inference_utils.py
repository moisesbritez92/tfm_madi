from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from v0_inference_utils import (
    compare_checkpoints as _compare_checkpoints,
    default_checkpoint as _default_checkpoint_v0,
    list_checkpoints as _list_checkpoints,
    plot_checkpoint_comparison as _plot_checkpoint_comparison,
    plot_rollout,
    resolve_device,
    run_rollout as _run_rollout,
    save_gif,
)

THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR / "models" / "V2"
ARTIFACT_DIR = THIS_DIR / "artifacts" / "v2_inference"
PREFERRED_CHECKPOINT = "epoch=0150-test_mean_score=0.648.ckpt"


def list_checkpoints(model_dir: Path = MODEL_DIR) -> list[Path]:
    return _list_checkpoints(model_dir=model_dir)


def default_checkpoint(model_dir: Path = MODEL_DIR) -> Path:
    return _default_checkpoint_v0(
        model_dir=model_dir,
        preferred_checkpoint=PREFERRED_CHECKPOINT,
    )


def run_rollout(
    checkpoint_path: Optional[Path] = None,
    seed: int = 10000,
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


def compare_checkpoints(
    model_dir: Path = MODEL_DIR,
    seed: int = 10000,
    device: Optional[Any] = None,
    max_checkpoints: Optional[int] = None,
    seeds: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    return _compare_checkpoints(
        model_dir=model_dir,
        seed=seed,
        device=device,
        max_checkpoints=max_checkpoints,
        artifact_dir=ARTIFACT_DIR,
        seeds=seeds,
    )


def plot_checkpoint_comparison(rows: list[dict[str, Any]], figsize: tuple[int, int] = (9, 4)):
    return _plot_checkpoint_comparison(
        rows,
        figsize=figsize,
        title="V2 checkpoints across test seeds",
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
