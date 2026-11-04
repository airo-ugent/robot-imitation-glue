"""Prepare a recorded dataset for training: choose the action space, the state and the cameras.

collect_data records the absolute target of every step (`target_pose` [xyz, rot6d] or `target_joints`, and
`target_gripper`) next to the observations (`state` [xyz, extrinsic xyz euler, gripper], `joints`,
`gripper_state`, images). This builds a new dataset from those, with:

`--action` (each action ends with the gripper):
    abs_eef       [xyz, rot6d, gripper width] absolute target in the robot base frame (10)
    umi           abs_eef + the relative action stats of a `policy.use_relative_actions` policy (UMI-style, the
                  relative chunks are made during training); needs --state eef
    delta_eef     [dxyz, d-euler xyz, d-gripper] w.r.t. the previous target, in its frame (7); the encoding of
                  `abs_se3_to_relative_policy_action_converter` (UR3 / OpenVLA)
    abs_joints    [joints, gripper width] absolute target; from `target_joints`, or from `target_pose` with --ik
    delta_joints  [d-joints, d-gripper] w.r.t. the previous target
`--state`: eef (the recorded `state`) or joints ([joints, gripper width]), as `observation.state`
`--cameras`: the images to keep, as `observation.images.<name>`; all other images are dropped

Frames whose target equals the previous one (e.g. the phone's "hold to move" released) are dropped, so a policy does
not learn to stand still; --keep-idle keeps them. For the delta actions the first frame of an episode is relative to
the measured pose / joints.

The arguments default to the constants below: edit them here, or override them on the command line.

example:
    python -m robot_imitation_glue.lerobot_dataset.prepare_dataset --root datasets/green_cup_pickup --action umi
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from robot_imitation_glue.collect_data import TARGET_GRIPPER_KEY, TARGET_JOINTS_KEY, TARGET_POSE_KEY
from robot_imitation_glue.lerobot_dataset.relative_action_stats import (
    DROP_N_LAST_FRAMES,
    HORIZON,
    N_OBS_STEPS,
    write_relative_action_stats,
)
from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset
from robot_imitation_glue.preprocessing import (
    abs_se3_to_relative_policy_action_converter,
    concatenate_state,
    euler_state_to_se3,
    policy_action_to_abs_se3_converter,
    repeated_targets,
)

# defaults of the command line arguments: edit them here or override them on the command line.
DATASET_ROOT = "datasets/green_cup_pickup"
NEW_DATASET_ROOT = None  # None: <root>_<action>
ACTION = "umi"
STATE = "eef"
CAMERAS = ["wrist_image"]
KEEP_IDLE = False
IK = None  # for abs_joints / delta_joints from EEF targets, see IK_SOLVERS
# for --action umi: the values of the policy config
UMI_HORIZON = HORIZON
UMI_N_OBS_STEPS = N_OBS_STEPS
UMI_DROP_N_LAST_FRAMES = DROP_N_LAST_FRAMES

# recorder features kept as they are, everything else is rebuilt or dropped
PASSED_THROUGH = ("next.reward", "next.success", "seed")
ACTIONS = ["abs_eef", "umi", "delta_eef", "abs_joints", "delta_joints"]
STATES = ["eef", "joints"]


def _ur5e_schunk_ik(target_pose: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
    """UR5e with the Schunk gripper: the IK solution closest to the measured joints."""
    from ur_analytic_ik import ur5e

    from robot_imitation_glue.preprocessing import tcp_offset_transform

    return np.array(ur5e.inverse_kinematics_closest_with_tcp(target_pose, tcp_offset_transform(), *current_joints)[0])


# --ik: (target 4x4 TCP pose, measured joints) -> joint target
IK_SOLVERS = {"ur5e_schunk": _ur5e_schunk_ik}


def _pose(target_pose) -> np.ndarray:
    """4x4 pose of a recorded [xyz, rot6d] target."""
    return policy_action_to_abs_se3_converter(None, None, np.append(target_pose[:9], 0.0))[0]


def _stack(frames: pd.DataFrame, key: str) -> np.ndarray:
    """(N, D) array of a feature; lerobot stores 1-element features as a scalar per frame."""
    values = np.stack(frames[key].to_numpy()).astype(np.float64)
    return values.reshape(len(frames), -1)


def episode_actions(frames: pd.DataFrame, action: str, ik=None) -> tuple[np.ndarray, np.ndarray]:
    """(the actions (N, D), the absolute targets (N, D') the idle frames are detected on) of one episode."""
    gripper = _stack(frames, TARGET_GRIPPER_KEY)
    if action in ("abs_eef", "umi", "delta_eef"):
        if TARGET_POSE_KEY not in frames:
            raise ValueError(f"--action {action} needs {TARGET_POSE_KEY} (EEF actions) in the dataset")
        targets = np.concatenate([_stack(frames, TARGET_POSE_KEY), gripper], axis=1)
        if action != "delta_eef":
            return targets, targets
        state = _stack(frames, "state")
        previous_pose, previous_gripper = euler_state_to_se3(state[0]), state[0, 6]
        deltas = []
        for target in targets:
            pose = _pose(target)
            delta = abs_se3_to_relative_policy_action_converter(previous_pose, previous_gripper, pose, target[9])
            deltas.append(delta)
            previous_pose, previous_gripper = pose, target[9]
        return np.stack(deltas), targets

    # joint actions
    measured_joints = _stack(frames, "joints")
    if TARGET_JOINTS_KEY in frames:
        joints = _stack(frames, TARGET_JOINTS_KEY)
    elif TARGET_POSE_KEY in frames and ik is not None:
        poses = [_pose(pose) for pose in _stack(frames, TARGET_POSE_KEY)]
        joints = np.stack([ik(pose, current) for pose, current in zip(poses, measured_joints)])
    else:
        raise ValueError(f"--action {action} needs {TARGET_JOINTS_KEY}, or {TARGET_POSE_KEY} and --ik")
    targets = np.concatenate([joints, gripper], axis=1)
    if action == "abs_joints":
        return targets, targets
    previous = np.concatenate([measured_joints[:1], _stack(frames, "gripper_state")[:1]], axis=1)
    return np.diff(np.concatenate([previous, targets]), axis=0), targets


def actions_lookup(root: Path, action: str, keep_idle: bool, ik=None) -> dict[tuple[int, int], np.ndarray | None]:
    """{(episode_index, frame_index): action, or None for a dropped frame}, from the parquet files."""
    files = sorted((root / "data").glob("**/*.parquet"))
    available = set(pq.read_schema(files[0]).names)
    if TARGET_GRIPPER_KEY not in available:
        raise ValueError(f"{root} has no recorded targets, record with collect_data.example_recording_observation")
    wanted = ["state", "joints", "gripper_state", TARGET_POSE_KEY, TARGET_JOINTS_KEY, TARGET_GRIPPER_KEY]
    columns = ["episode_index", "frame_index"] + [key for key in wanted if key in available]
    frames = pd.concat(pd.read_parquet(f, columns=columns) for f in files)
    frames = frames.sort_values(["episode_index", "frame_index"])

    lookup = {}
    for episode, episode_frames in frames.groupby("episode_index"):
        actions, targets = episode_actions(episode_frames, action, ik)
        idle = np.zeros(len(actions), dtype=bool) if keep_idle else repeated_targets(targets)
        for frame_index, frame_action, drop in zip(episode_frames["frame_index"], actions, idle):
            lookup[(int(episode), int(frame_index))] = None if drop else frame_action.astype(np.float32)
        print(f"episode {episode}: keeping {int((~idle).sum())} / {len(idle)} frames")
    return lookup


def prepare_dataset(
    root: Path,
    new_root: Path,
    action: str = ACTION,
    state: str = STATE,
    cameras: list[str] = CAMERAS,
    keep_idle: bool = KEEP_IDLE,
    ik: str | None = IK,
    umi_horizon: int = UMI_HORIZON,
    umi_n_obs_steps: int = UMI_N_OBS_STEPS,
    umi_drop_n_last_frames: int = UMI_DROP_N_LAST_FRAMES,
):
    if action == "umi" and state != "eef":
        raise ValueError(
            "--action umi needs --state eef, the relative actions are expressed in the frame of that pose"
        )
    lookup = actions_lookup(root, action, keep_idle, IK_SOLVERS[ik] if ik else None)
    action_dim = len(next(a for a in lookup.values() if a is not None))
    renamed = {"state": "observation.state", **{camera: f"observation.images.{camera}" for camera in cameras}}

    def features_transform(features):
        images = [key for key, feature in features.items() if feature["dtype"] in ("video", "image")]
        if any(camera not in images for camera in cameras):
            raise ValueError(f"--cameras {cameras}: {root} has the images {images}")
        kept = {key: features[key] for key in PASSED_THROUGH if key in features}
        kept.update({renamed[key]: feature for key, feature in features.items() if key in renamed})
        kept["action"] = {**features["action"], "shape": (action_dim,)}
        if state == "joints":
            kept["observation.state"] = {**features["state"], "shape": (features["joints"]["shape"][0] + 1,)}
        return kept

    def frame_transform(frame):
        frame_action = lookup[(frame["episode_index"], frame["frame_index"])]
        if frame_action is None:
            return None
        new_frame = {renamed[key]: value for key, value in frame.items() if key in renamed}
        new_frame.update({key: frame[key] for key in ("task", *PASSED_THROUGH) if key in frame})
        if state == "joints":
            new_frame["observation.state"] = concatenate_state(frame["joints"].numpy(), frame["gripper_state"].numpy())
        new_frame["action"] = frame_action
        return new_frame

    transform_dataset(
        root_dir=str(root),
        new_root_dir=str(new_root),
        new_repo_id=Path(new_root).name,
        transform_fn=frame_transform,
        transform_features_fn=features_transform,
    )
    if action == "umi":
        write_relative_action_stats(
            new_root, horizon=umi_horizon, n_obs_steps=umi_n_obs_steps, drop_n_last_frames=umi_drop_n_last_frames
        )
    print(f"prepared dataset: {new_root} (action {action}, {action_dim} dims; state {state}; cameras {cameras})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(DATASET_ROOT), help="recorded dataset")
    parser.add_argument("--new-root", type=Path, default=NEW_DATASET_ROOT, help="default: <root>_<action>")
    parser.add_argument("--action", default=ACTION, choices=ACTIONS)
    parser.add_argument("--state", default=STATE, choices=STATES)
    parser.add_argument("--cameras", nargs="+", default=CAMERAS, help="image keys of the recorded dataset to keep")
    parser.add_argument("--keep-idle", action="store_true", default=KEEP_IDLE, help="keep frames without motion")
    parser.add_argument("--ik", default=IK, choices=list(IK_SOLVERS), help="EEF target -> joints, for joint actions")
    parser.add_argument("--horizon", type=int, default=UMI_HORIZON, help="umi: policy.horizon")
    parser.add_argument("--n-obs-steps", type=int, default=UMI_N_OBS_STEPS, help="umi: policy.n_obs_steps")
    parser.add_argument("--drop-n-last-frames", type=int, default=UMI_DROP_N_LAST_FRAMES, help="umi: policy value")
    args = parser.parse_args()
    new_root = args.new_root or args.root.with_name(f"{args.root.name}_{args.action}")
    prepare_dataset(
        args.root,
        new_root,
        args.action,
        args.state,
        args.cameras,
        args.keep_idle,
        args.ik,
        args.horizon,
        args.n_obs_steps,
        args.drop_n_last_frames,
    )


if __name__ == "__main__":
    main()
