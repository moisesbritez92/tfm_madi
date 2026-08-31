from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Optional

import dill
import hydra
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import open_dict
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR / "repo" / "diffusion_policy"
MODEL_DIR = THIS_DIR / "models" / "V0"
ARTIFACT_DIR = THIS_DIR / "artifacts" / "v0_inference"
PREFERRED_CHECKPOINT = "epoch=0350-test_mean_score=0.865.ckpt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


def resolve_device(preferred: Optional[Any] = None) -> torch.device:
    if isinstance(preferred, torch.device):
        if preferred.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return preferred

    if preferred is None:
        preferred = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(str(preferred))
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def list_checkpoints(model_dir: Path = MODEL_DIR) -> list[Path]:
    model_dir = Path(model_dir)
    checkpoints = sorted(model_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return checkpoints


def default_checkpoint(
    model_dir: Path = MODEL_DIR,
    preferred_checkpoint: Optional[str] = PREFERRED_CHECKPOINT,
) -> Path:
    return _default_checkpoint(
        model_dir=model_dir,
        preferred_checkpoint=preferred_checkpoint,
    )


def _default_checkpoint(
    model_dir: Path = MODEL_DIR,
    preferred_checkpoint: Optional[str] = PREFERRED_CHECKPOINT,
) -> Path:
    model_dir = Path(model_dir)
    if preferred_checkpoint:
        preferred = model_dir / preferred_checkpoint
        if preferred.exists():
            return preferred

    latest = model_dir / "latest.ckpt"
    if latest.exists():
        return latest

    return list_checkpoints(model_dir)[0]


def load_policy_bundle(
    checkpoint_path: Path,
    device: Optional[Any] = None,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    device = resolve_device(device)

    payload = torch.load(
        checkpoint_path.open("rb"),
        map_location="cpu",
        pickle_module=dill,
    )
    cfg = payload["cfg"]

    # La politica hibrida del articulo (V_Paper) construye su encoder dentro de
    # robomimic y no tiene la clave obs_encoder, asi que el acceso va con guarda.
    obs_encoder_cfg = getattr(cfg.policy, "obs_encoder", None)
    rgb_model_cfg = (
        getattr(obs_encoder_cfg, "rgb_model", None) if obs_encoder_cfg is not None else None
    )
    if rgb_model_cfg is not None:
        target = str(getattr(rgb_model_cfg, "_target_", ""))
        if "pretrained_encoders" in target:
            with open_dict(rgb_model_cfg):
                rgb_model_cfg.pretrained = False
        elif target == "diffusion_policy.model.vision.model_getter.get_resnet":
            with open_dict(rgb_model_cfg):
                rgb_model_cfg.weights = None

    optimizer_cfg = getattr(cfg, "optimizer", None)
    if optimizer_cfg is not None:
        with open_dict(optimizer_cfg):
            optimizer_cfg.pop("low_lr", None)
            optimizer_cfg.pop("low_lr_scope", None)

    workspace_cls = hydra.utils.get_class(cfg._target_)

    output_dir = Path(artifact_dir) / checkpoint_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = workspace_cls(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=("optimizer",), include_keys=None)

    policy = workspace.ema_model if getattr(cfg.training, "use_ema", False) else workspace.model
    policy.eval()
    policy.to(device)

    return {
        "cfg": cfg,
        "device": device,
        "policy": policy,
        "workspace": workspace,
    }


def make_env(cfg: Any, seed: int = 10000, render_size: Optional[int] = None, max_steps: Optional[int] = None) -> MultiStepWrapper:
    runner_cfg = cfg.task.env_runner
    render_size = int(render_size or getattr(runner_cfg, "render_size", 96))
    max_steps = int(max_steps or getattr(runner_cfg, "max_steps", 200))
    legacy_test = bool(getattr(runner_cfg, "legacy_test", False))

    env = MultiStepWrapper(
        PushTImageEnv(
            legacy=legacy_test,
            render_size=render_size,
        ),
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),
        max_episode_steps=max_steps,
    )
    env.seed(seed)
    return env


def _obs_to_torch(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    def convert(x: np.ndarray) -> torch.Tensor:
        array = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(array).unsqueeze(0).to(device=device)

    return dict_apply(obs, convert)


def run_rollout(
    checkpoint_path: Optional[Path] = None,
    seed: int = 10000,
    device: Optional[Any] = None,
    max_decisions: Optional[int] = None,
    render_size: Optional[int] = None,
    max_steps: Optional[int] = None,
    model_dir: Path = MODEL_DIR,
    artifact_dir: Path = ARTIFACT_DIR,
    preferred_checkpoint: Optional[str] = PREFERRED_CHECKPOINT,
) -> dict[str, Any]:
    checkpoint_path = Path(
        checkpoint_path or _default_checkpoint(
            model_dir=model_dir,
            preferred_checkpoint=preferred_checkpoint,
        )
    )
    bundle = load_policy_bundle(
        checkpoint_path,
        device=device,
        artifact_dir=artifact_dir,
    )
    policy = bundle["policy"]
    cfg = bundle["cfg"]
    device = bundle["device"]

    env = make_env(cfg, seed=seed, render_size=render_size, max_steps=max_steps)
    frames: list[np.ndarray] = []
    decision_rewards: list[float] = []

    try:
        obs = env.reset()
        frames.append(env.render(mode="rgb_array"))

        if hasattr(policy, "reset"):
            policy.reset()

        done = False
        n_decisions = 0
        while not done:
            obs_dict = _obs_to_torch(obs, device)
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            action = action_dict["action"][0].detach().cpu().numpy()
            obs, reward, done, _ = env.step(action)
            frames.append(env.render(mode="rgb_array"))
            decision_rewards.append(float(reward))
            n_decisions += 1

            if max_decisions is not None and n_decisions >= max_decisions:
                break

        step_rewards = np.asarray(env.get_rewards(), dtype=np.float32)
        max_reward = float(step_rewards.max()) if step_rewards.size else 0.0
        result = {
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_path": checkpoint_path,
            "artifact_dir": Path(artifact_dir),
            "seed": seed,
            "device": str(device),
            "frames": frames,
            "decision_rewards": np.asarray(decision_rewards, dtype=np.float32),
            "step_rewards": step_rewards,
            "max_reward": max_reward,
            "n_decisions": n_decisions,
            "n_env_steps": int(step_rewards.size),
            "success": bool(max_reward >= 0.999),
        }
    finally:
        env.close()

    del bundle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def save_gif(result: dict[str, Any], output_path: Optional[Path] = None, fps: int = 4) -> Path:
    gif_dir = Path(result.get("artifact_dir", ARTIFACT_DIR)) / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        stem = Path(result["checkpoint_name"]).stem.replace("=", "_")
        output_path = gif_dir / f"{stem}_seed{result['seed']}.gif"

    output_path = Path(output_path)
    imageio.mimsave(output_path, result["frames"], duration=1.0 / fps, loop=0)
    return output_path


def plot_rollout(result: dict[str, Any], figsize: tuple[int, int] = (14, 4)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(result["frames"][0])
    axes[0].set_title("Start")
    axes[0].axis("off")

    axes[1].imshow(result["frames"][-1])
    axes[1].set_title("End")
    axes[1].axis("off")

    axes[2].plot(result["step_rewards"], color="tab:blue", linewidth=2)
    axes[2].axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xlabel("Environment step")
    axes[2].set_ylabel("Reward")
    axes[2].set_title("Reward curve")

    fig.suptitle(
        f"{result['checkpoint_name']} | seed={result['seed']} | max_reward={result['max_reward']:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def compare_checkpoints(
    model_dir: Path = MODEL_DIR,
    seed: int = 10000,
    device: Optional[Any] = None,
    max_checkpoints: Optional[int] = None,
    artifact_dir: Path = ARTIFACT_DIR,
    seeds: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    checkpoints = list_checkpoints(model_dir)
    if max_checkpoints is not None:
        checkpoints = checkpoints[:max_checkpoints]

    if seeds is None:
        seeds = [seed]
    else:
        seeds = list(seeds)
    if len(seeds) == 0:
        raise ValueError("seeds must contain at least one value")

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        rewards: list[float] = []
        step_counts: list[int] = []
        for this_seed in seeds:
            rollout = run_rollout(
                checkpoint_path=checkpoint,
                seed=this_seed,
                device=device,
                model_dir=model_dir,
                artifact_dir=artifact_dir,
            )
            rewards.append(rollout["max_reward"])
            step_counts.append(rollout["n_env_steps"])

        reward_array = np.asarray(rewards, dtype=np.float32)
        rows.append(
            {
                "checkpoint": checkpoint.name,
                "max_reward": float(reward_array.mean()),
                "std_reward": float(reward_array.std()),
                "success": bool(np.any(reward_array >= 0.999)),
                "success_rate": float(np.mean(reward_array >= 0.999)),
                "n_env_steps": int(np.mean(step_counts)),
                "n_seeds": len(seeds),
                "seeds": seeds,
                "per_seed_rewards": rewards,
            }
        )
    return rows


def plot_checkpoint_comparison(
    rows: list[dict[str, Any]],
    figsize: tuple[int, int] = (9, 4),
    title: str = "V0 checkpoints on the same seed",
):
    labels = [row["checkpoint"].replace(".ckpt", "") for row in rows]
    values = [row["max_reward"] for row in rows]
    errors = [row.get("std_reward", 0.0) for row in rows]
    colors = ["tab:green" if row["success"] else "tab:blue" for row in rows]
    n_seeds = rows[0].get("n_seeds") if rows else None

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(rows)), values, color=colors, yerr=errors, capsize=5)
    ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean max reward")
    if n_seeds is not None:
        ax.set_title(f"{title} ({n_seeds} seeds)")
    else:
        ax.set_title(title)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()
    return fig


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
