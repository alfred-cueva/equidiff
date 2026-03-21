#!/usr/bin/env python3
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import argparse
import pathlib
import sys
from typing import Optional, Set

import imageio
import torch
from omegaconf import OmegaConf
import hydra.utils as hyu

# --- match resolvers from train.py ---
_MAX_STEPS = {
    'stack_d1': 400,
    'stack_three_d1': 400,
    'square_d2': 400,
    'threading_d2': 400,
    'coffee_d2': 400,
    'three_piece_assembly_d2': 500,
    'hammer_cleanup_d1': 500,
    'mug_cleanup_d1': 500,
    'kitchen_d1': 800,
    'nut_assembly_d0': 500,
    'pick_place_d0': 1000,
    'coffee_preparation_d1': 800,
    'tool_hang': 700,
    'can': 400,
    'lift': 400,
    'square': 400,
}
def _ws_x_center(task_name: str) -> float:
    return -0.2 if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_') else 0.0

OmegaConf.register_new_resolver("get_max_steps", lambda x: _MAX_STEPS[x], replace=True)
OmegaConf.register_new_resolver("get_ws_x_center", _ws_x_center, replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", lambda _: 0.0, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Ensure EMA class is importable for pickled references
from equi_diffpo.model.diffusion.ema_model import EMAModel  # noqa: F401
from equi_diffpo.workspace.train_equi_workspace import TrainEquiWorkspace

def _combine_new_videos(media_dir: pathlib.Path, existing: Set[pathlib.Path], fps: int = 20) -> Optional[pathlib.Path]:
    """
    Combine any new mp4 files under media_dir (those not present in `existing`)
    into a single rollouts_combined.mp4. Removes the individual new files after
    successfully writing the combined video.
    """
    if not media_dir.exists():
        return None

    new_videos = [p for p in media_dir.glob("*.mp4") if p not in existing]
    if not new_videos:
        return None

    # sort by modification time to preserve rollout order
    new_videos.sort(key=lambda p: p.stat().st_mtime)

    combined_path = media_dir / "rollouts_combined.mp4"
    if combined_path.exists():
        combined_path.unlink()

    # Fast path: if there's only one video, just rename it
    if len(new_videos) == 1:
        new_videos[0].rename(combined_path)
        return combined_path

    writer = imageio.get_writer(combined_path, fps=fps)
    first_shape = None
    try:
        for video_path in new_videos:
            reader = imageio.get_reader(video_path)
            try:
                for frame in reader:
                    if first_shape is None:
                        first_shape = frame.shape
                    elif frame.shape != first_shape:
                        raise ValueError(
                            f"Video {video_path.name} has shape {frame.shape}, expected {first_shape}"
                        )
                    writer.append_data(frame)
            finally:
                reader.close()
    except Exception:
        writer.close()
        try:
            combined_path.unlink()
        except FileNotFoundError:
            pass
        raise
    writer.close()

    # Clean up individual files now that we have a combined one
    for video_path in new_videos:
        try:
            video_path.unlink()
        except Exception:
            pass

    return combined_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to workspace checkpoint (*.ckpt)")
    ap.add_argument("--dataset", required=True, help="Path to robomimic dataset hdf5")
    ap.add_argument("--task", default=None, help="Optional task name override (e.g., stack_d1)")
    ap.add_argument("--out", default="rollouts_out", help="Output dir for videos/logs")
    ap.add_argument("--n_test", type=int, default=20, help="Number of test rollouts to run")
    ap.add_argument("--device", default=None, help="Override device, e.g., cuda:0 or cpu")
    ap.add_argument("--video-width", type=int, default=480, help="Video width in pixels")
    ap.add_argument("--video-height", type=int, default=480, help="Video height in pixels")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for test rollouts (randomized if not set)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    existing_videos = set(media_dir.glob("*.mp4")) if media_dir.exists() else set()

    # 1) Restore workspace directly from checkpoint (uses saved cfg + weights)
    ws: TrainEquiWorkspace = TrainEquiWorkspace.create_from_checkpoint(args.ckpt)
    ws._output_dir = str(out_dir)  # use user-specified output dir

    # 2) Apply user overrides to saved cfg, then resolve like training
    if args.task is not None:
        ws.cfg.task_name = args.task
    ws.cfg.dataset_path = args.dataset
    ws.cfg.task.dataset.dataset_path = args.dataset  # Also update nested reference
    if hasattr(ws.cfg.task, "env_runner"):
        env_runner_cfg = ws.cfg.task.env_runner
        target = env_runner_cfg.get("_target_", "")
        if "robomimic_image_runner" in target:
            env_runner_cfg._target_ = "equi_diffpo.env_runner.robomimic_image_runner_rollout_1dir_xaxis.RobomimicImageRunner"
        env_runner_cfg.n_test = int(args.n_test)
        if "n_test_vis" in env_runner_cfg:
            env_runner_cfg.n_test_vis = int(args.n_test)
        if "dataset_path" in env_runner_cfg:
            env_runner_cfg.dataset_path = args.dataset
        if "n_train" in env_runner_cfg:
            env_runner_cfg.n_train = 0
        if "n_train_vis" in env_runner_cfg:
            env_runner_cfg.n_train_vis = 0
        if "n_envs" in env_runner_cfg:
            env_runner_cfg.n_envs = int(args.n_test)
        # ensure only test rollouts are run
        if "run_train_rollouts" in env_runner_cfg:
            env_runner_cfg.run_train_rollouts = False
        # set random seed for test rollouts
        if args.seed is not None:
            env_runner_cfg.test_start_seed = args.seed
        else:
            import time
            env_runner_cfg.test_start_seed = int(time.time() * 1000) % (2**31)
        print(f"Using test_start_seed: {env_runner_cfg.test_start_seed}")
    OmegaConf.resolve(ws.cfg)

    # 3) Dataset & normalizer
    dataset = hyu.instantiate(ws.cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    ws.model.set_normalizer(normalizer)
    if ws.cfg.training.use_ema and ws.ema_model is not None:
        ws.ema_model.set_normalizer(normalizer)

    # 4) Device transfer
    device_str = args.device if args.device is not None else ws.cfg.training.device
    device = torch.device(device_str)
    ws.model.to(device)
    if ws.cfg.training.use_ema and ws.ema_model is not None:
        ws.ema_model.to(device)

    # Choose the eval policy exactly like training does
    policy = ws.ema_model if (ws.cfg.training.use_ema and ws.ema_model is not None) else ws.model
    policy.eval()

    # 5) Env runner from cfg and run once
    env_runner = hyu.instantiate(
        ws.cfg.task.env_runner,
        output_dir=str(out_dir),
        video_width=args.video_width,
        video_height=args.video_height,
    )
    video_fps = getattr(env_runner, "fps", 20)
    try:
        logs = env_runner.run(policy)
    finally:
        # ensure proper shutdown order
        try:
            env_runner.env.close()
        except Exception:
            pass
        del env_runner
        import gc; gc.collect()

    combined_video_path = None
    try:
        combined_video_path = _combine_new_videos(
            media_dir,
            existing_videos,
            fps=video_fps,
        )
    except Exception as exc:
        print(f"Warning: failed to combine rollout videos into one file: {exc}")

    # 6) Print key results
    # print("\n== ROLLOUT RESULTS ==")
    # for k in sorted(logs.keys()):
    #     if k.endswith("mean_score") or k.endswith("max_score"):
    #         print(f"{k}: {logs[k]:.3f}")
    # print(f"\nVideos (if any) saved under: {out_dir / 'media'}")

        # 6) Print key results (as success rate)
    print("\n== ROLLOUT RESULTS ==")

    # Overall stats (match robomimic-style summary)
    ep_returns = logs.get("_episode_returns", [])
    ep_horizons = logs.get("_episode_horizons", [])
    ep_success = logs.get("_episode_success", [])
    if ep_returns and ep_horizons and ep_success:
        avg_return = float(sum(ep_returns) / len(ep_returns))
        avg_horizon = float(sum(ep_horizons) / len(ep_horizons))
        success_rate = float(sum(1.0 if s else 0.0 for s in ep_success) / len(ep_success))
        num_success = int(sum(1 for s in ep_success if s))
        summary = {
            "Return": avg_return,
            "Horizon": avg_horizon,
            "Success_Rate": success_rate,
            "Num_Success": num_success,
        }
        import json as _json
        print("Average Rollout Stats")
        print(_json.dumps(summary, indent=4))
    else:
        for split in ("train/", "test/"):
            keys = [k for k in logs if k.startswith(f"{split}sim_max_reward_")]
            total = len(keys)
            if total == 0:
                continue
            successes = sum(float(logs[k]) > 0 for k in keys)
            rate = successes / total
            print(f"{split}success: {successes}/{total} ({rate*100:.1f}%)")
            print(f"{split}mean_score: {logs.get(f'{split}mean_score', float('nan')):.3f}")

    if combined_video_path is not None:
        print(f"\nCombined rollout video saved to: {combined_video_path}")
    else:
        print(f"\nVideos (if any) saved under: {out_dir / 'media'}")


    # Be nice to multiprocessing
    try:
        env_runner.env.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
