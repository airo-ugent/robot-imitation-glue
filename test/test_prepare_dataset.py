"""prepare_dataset builds the chosen action space, state and cameras from the recorded absolute targets."""

import json

import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.lerobot_dataset")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

import robot_imitation_glue.lerobot_dataset.prepare_dataset as prepare  # noqa: E402
from robot_imitation_glue.collect_data import target_observation  # noqa: E402
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder  # noqa: E402
from robot_imitation_glue.preprocessing import (  # noqa: E402
    abs_se3_to_policy_action_converter,
    abs_se3_to_relative_policy_action_converter,
    euler_state_to_se3,
)

# per step: (translation of the target, change of the gripper width); the second step is idle
STEPS = [([0.01, 0, 0], -0.01), ([0, 0, 0], 0.0), ([0, 0.02, -0.01], -0.02), ([0, 0, 0.01], 0.0)]
START_STATE = np.array([0.1, -0.3, 0.2, np.pi, 0.1, -0.2, 0.08], dtype=np.float32)
START_JOINTS = np.array([0.0, -1.5, 1.5, -1.5, -1.5, 0.0], dtype=np.float32)


def _image():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _observation(joints):
    return {
        "wrist_image": _image(),
        "scene_image": _image(),
        "state": START_STATE,
        "joints": joints.astype(np.float32),
        "gripper_state": START_STATE[6:],
    }


def _record(root, eef=True, n_episodes=2):
    """episodes as collect_data records them (EEF: relative action w.r.t. the previous target, joint: absolute)."""
    joints = START_JOINTS
    example = _observation(joints)
    if eef:
        example.update(target_observation(0.08, target_pose=np.eye(4)))
    else:
        example.update(target_observation(0.08, target_joints=joints))
    recorder = LeRobotDatasetRecorder(example, np.zeros(7, dtype=np.float32), root, "test", fps=10, use_videos=False)
    episodes = []
    for _ in range(n_episodes):
        recorder.start_episode()
        target, gripper, target_joints = euler_state_to_se3(START_STATE), float(START_STATE[6]), START_JOINTS
        poses, grippers, joint_targets = [], [], []
        for translation, gripper_change in STEPS:
            previous, previous_gripper = target, gripper
            target = target.copy()
            target[:3, 3] += translation
            gripper += gripper_change
            target_joints = target_joints + np.sum(translation)
            if eef:
                targets = target_observation(gripper, target_pose=target)
                action = abs_se3_to_relative_policy_action_converter(previous, previous_gripper, target, gripper)
            else:
                targets = target_observation(gripper, target_joints=target_joints)
                action = np.append(target_joints, gripper).astype(np.float32)[:7]
            recorder.record_step({**_observation(START_JOINTS), **targets}, action)
            poses.append(target)
            grippers.append(gripper)
            joint_targets.append(target_joints)
        recorder.save_episode()
        episodes.append((poses, grippers, joint_targets))
    recorder.finish_recording()
    return episodes


def _actions(root):
    dataset = LeRobotDataset("prepared", root=root)
    return [(int(dataset[i]["episode_index"]), dataset[i]["action"].numpy()) for i in range(len(dataset))], dataset


def test_abs_eef_drops_idle_frames_and_other_cameras(tmp_path):
    poses, grippers, _ = _record(tmp_path / "raw")[0]
    prepare.prepare_dataset(tmp_path / "raw", tmp_path / "new", action="abs_eef", cameras=["wrist_image"])

    actions, dataset = _actions(tmp_path / "new")
    assert len(actions) == 6 and [episode for episode, _ in actions] == [0, 0, 0, 1, 1, 1]  # step 1 dropped
    expected = [abs_se3_to_policy_action_converter(None, None, poses[i], grippers[i]) for i in (0, 2, 3)]
    np.testing.assert_allclose([a for _, a in actions[:3]], expected, atol=1e-6)
    features = set(dataset.features)
    assert {"observation.images.wrist_image", "observation.state", "action"} <= features
    assert not features & {"observation.images.scene_image", "scene_image", "wrist_image", "joints", "target_pose"}


def test_keep_idle_and_umi_stats(tmp_path):
    _record(tmp_path / "raw")
    prepare.prepare_dataset(tmp_path / "raw", tmp_path / "new", action="umi", keep_idle=True, umi_drop_n_last_frames=1)

    actions, _ = _actions(tmp_path / "new")
    assert len(actions) == 8
    assert json.loads((tmp_path / "new" / "action_representation.json").read_text())["use_relative_actions"]


def test_delta_eef_is_the_recorded_encoding(tmp_path):
    poses, grippers, _ = _record(tmp_path / "raw")[0]
    prepare.prepare_dataset(tmp_path / "raw", tmp_path / "new", action="delta_eef", keep_idle=True)

    actions, _ = _actions(tmp_path / "new")
    start = euler_state_to_se3(START_STATE)
    previous = [(start, START_STATE[6])] + list(zip(poses, grippers))[:-1]
    expected = [
        abs_se3_to_relative_policy_action_converter(p, pg, t, g) for (p, pg), t, g in zip(previous, poses, grippers)
    ]
    np.testing.assert_allclose([a for _, a in actions[:4]], expected, atol=1e-5)


def test_joint_actions_and_state(tmp_path):
    _, grippers, joint_targets = _record(tmp_path / "raw", eef=False)[0]
    prepare.prepare_dataset(tmp_path / "raw", tmp_path / "new", action="abs_joints", state="joints", keep_idle=True)
    actions, dataset = _actions(tmp_path / "new")
    np.testing.assert_allclose(actions[2][1], np.append(joint_targets[2], grippers[2]), atol=1e-6)
    np.testing.assert_allclose(dataset[0]["observation.state"].numpy(), np.append(START_JOINTS, START_STATE[6]))

    prepare.prepare_dataset(tmp_path / "raw", tmp_path / "delta", action="delta_joints", keep_idle=True)
    actions, _ = _actions(tmp_path / "delta")
    np.testing.assert_allclose(actions[2][1], np.append(joint_targets[2] - joint_targets[1], -0.02), atol=1e-6)


def test_joint_actions_from_eef_targets_with_ik(tmp_path, monkeypatch):
    poses, _, _ = _record(tmp_path / "raw", n_episodes=1)[0]
    # a fake solver: the joints are the target position, offset by the measured joints
    monkeypatch.setitem(prepare.IK_SOLVERS, "fake", lambda pose, joints: np.append(pose[:3, 3], joints[:3]))
    prepare.prepare_dataset(tmp_path / "raw", tmp_path / "new", action="abs_joints", keep_idle=True, ik="fake")

    actions, _ = _actions(tmp_path / "new")
    np.testing.assert_allclose(actions[2][1][:3], poses[2][:3, 3], atol=1e-6)
    np.testing.assert_allclose(actions[2][1][3:6], START_JOINTS[:3], atol=1e-6)


def test_invalid_requests_are_refused(tmp_path):
    _record(tmp_path / "raw", n_episodes=1)
    with pytest.raises(ValueError, match="state eef"):
        prepare.prepare_dataset(tmp_path / "raw", tmp_path / "new", action="umi", state="joints")
    with pytest.raises(ValueError, match="--ik"):
        prepare.prepare_dataset(tmp_path / "raw", tmp_path / "new", action="abs_joints")
    with pytest.raises(ValueError, match="cameras"):
        prepare.prepare_dataset(tmp_path / "raw", tmp_path / "new", action="abs_eef", cameras=["top_image"])

    recorder = LeRobotDatasetRecorder(
        _observation(START_JOINTS), np.zeros(7, dtype=np.float32), tmp_path / "old", "old", 10, use_videos=False
    )
    recorder.start_episode()
    recorder.record_step(_observation(START_JOINTS), np.zeros(7, dtype=np.float32))
    recorder.save_episode()
    recorder.finish_recording()
    with pytest.raises(ValueError, match="no recorded targets"):
        prepare.prepare_dataset(tmp_path / "old", tmp_path / "new", action="abs_eef")
