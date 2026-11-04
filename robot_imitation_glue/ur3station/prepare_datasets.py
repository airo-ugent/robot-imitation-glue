"""Prepare a UR3 dataset for a diffusion policy with UMI-style relative actions.

Needs a dataset with the absolute targets collect_data records (`target_pose`, `target_gripper`). The new dataset
has:

- `action`: 10-dim absolute targets [xyz, rot6d (rotation x-axis, y-axis), gripper width] in the robot base frame,
  i.e. the recorded targets. This is the layout lerobot's `SE3RelativeActionsProcessorStep`
  (`policy.use_relative_actions`) turns into UMI-style relative chunks during training, and `LerobotAgent` converts
  the predictions back to absolute.
- `observation.state`: the recorded state [xyz, extrinsic xyz euler, gripper width], the reference pose for that.
- `observation.images.wrist_image`: the 224x224 wrist image. Scene and full-resolution images are dropped.
- no idle frames: frames whose target equals the previous one (e.g. the phone's "hold to move" released) are
  removed, so the policy does not learn to stand still. Use --keep-idle to keep them.

Afterwards it writes the relative action stats (`lerobot_dataset/relative_action_stats.py`) for the policy's
horizon / n_obs_steps / drop_n_last_frames, which the policy needs to normalize the relative chunks.

example:
    python -m robot_imitation_glue.ur3station.prepare_datasets --root datasets/green_cup_pickup
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from robot_imitation_glue.collect_data import TARGET_GRIPPER_KEY, TARGET_POSE_KEY
from robot_imitation_glue.lerobot_dataset.relative_action_stats import (
    DROP_N_LAST_FRAMES,
    HORIZON,
    N_OBS_STEPS,
    write_relative_action_stats,
)
from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset
from robot_imitation_glue.preprocessing import ORIGINAL_IMAGE_KEYS, remap_to_lerobot_keys, repeated_targets

# defaults of the command line arguments: edit them here or override them on the command line.
DATASET_ROOT = "datasets/green_cup_pickup"
NEW_DATASET_SUFFIX = "_umi"
FEATURES_TO_DROP = ORIGINAL_IMAGE_KEYS + [
    "scene_image",
    "robot_pose",
    "gripper_state",
    "joints",
    TARGET_POSE_KEY,
    TARGET_GRIPPER_KEY,
]


def abs_actions_and_keep_mask(root: Path, keep_idle: bool) -> dict[tuple[int, int], np.ndarray | None]:
    """{(episode_index, frame_index): absolute 10-dim action, or None for a frame to drop}, from the parquet files."""
    files = sorted((root / "data").glob("**/*.parquet"))
    if TARGET_POSE_KEY not in pq.read_schema(files[0]).names:
        raise ValueError(f"{root} has no {TARGET_POSE_KEY}: record it with example_recording_observation.")
    columns = ["episode_index", "frame_index", TARGET_POSE_KEY, TARGET_GRIPPER_KEY]
    frames = pd.concat(pd.read_parquet(f, columns=columns) for f in files)
    frames = frames.sort_values(["episode_index", "frame_index"])

    lookup = {}
    for episode, episode_frames in frames.groupby("episode_index"):
        target_pose = np.stack(episode_frames[TARGET_POSE_KEY].to_numpy())
        # lerobot stores the 1-element gripper feature as a scalar per frame
        target_gripper = np.stack(episode_frames[TARGET_GRIPPER_KEY].to_numpy()).reshape(-1, 1)
        absolute = np.concatenate([target_pose, target_gripper], axis=1).astype(np.float32)
        idle = np.zeros(len(absolute), dtype=bool) if keep_idle else repeated_targets(absolute)
        for frame_index, action, drop in zip(episode_frames["frame_index"], absolute, idle):
            lookup[(int(episode), int(frame_index))] = None if drop else action
        print(f"episode {episode}: keeping {int((~idle).sum())} / {len(idle)} frames")
    return lookup


def features_transform(features):
    features = remap_to_lerobot_keys(features)
    features["action"] = {**features["action"], "shape": (10,)}
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(DATASET_ROOT), help="recorded dataset")
    parser.add_argument("--new-root", type=Path, default=None, help=f"default: <root>{NEW_DATASET_SUFFIX}")
    parser.add_argument("--keep-idle", action="store_true", help="keep the frames that do not move the robot")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="policy.horizon of the training config")
    parser.add_argument("--n-obs-steps", type=int, default=N_OBS_STEPS, help="policy.n_obs_steps")
    parser.add_argument("--drop-n-last-frames", type=int, default=DROP_N_LAST_FRAMES, help="policy.drop_n_last_frames")
    args = parser.parse_args()
    new_root = args.new_root or args.root.with_name(args.root.name + NEW_DATASET_SUFFIX)

    lookup = abs_actions_and_keep_mask(args.root, args.keep_idle)

    def frame_transform(frame):
        action = lookup[(frame["episode_index"], frame["frame_index"])]
        if action is None:
            return None
        new_frame = remap_to_lerobot_keys(frame)
        new_frame["action"] = action
        return new_frame

    transform_dataset(
        root_dir=str(args.root),
        new_root_dir=str(new_root),
        new_repo_id=new_root.name,
        transform_fn=frame_transform,
        transform_features_fn=features_transform,
        features_to_drop=FEATURES_TO_DROP,
    )
    write_relative_action_stats(
        new_root, horizon=args.horizon, n_obs_steps=args.n_obs_steps, drop_n_last_frames=args.drop_n_last_frames
    )
    print(f"prepared dataset: {new_root}")


if __name__ == "__main__":
    main()
