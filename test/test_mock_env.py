import inspect

import numpy as np
import pytest

from robot_imitation_glue.base import ActionType, BaseEnv
from robot_imitation_glue.mock import MockEnv


def test_eef_action():
    env = MockEnv()
    env.reset()

    pose = np.eye(4)
    pose[:2, 3] = [10, 20]
    # keyword args deliberately: the control loops call act() by keyword, so the parameter
    # names must match BaseEnv and the real stations
    env.act(robot_pose_se3=pose, gripper_pose=0.05, timestamp=0)

    np.testing.assert_allclose(env.get_robot_pose_se3()[:2, 3], [10, 20])
    assert env.get_gripper_opening() == 0.05


def test_joint_action():
    env = MockEnv()
    env.reset()

    env.act_joints(robot_joints=np.array([30.0, 40.0]), gripper_pose=0.02, timestamp=0)

    np.testing.assert_allclose(env.get_joint_configuration(), [30, 40])
    np.testing.assert_allclose(env.get_robot_pose_se3()[:2, 3], [30, 40])
    assert env.get_gripper_opening() == 0.02


def test_observation_keys():
    # the mock must emit the same keys as the real stations, the collection & eval loops read them
    env = MockEnv()
    env.reset()
    env.act_joints(robot_joints=np.array([30.0, 40.0]), gripper_pose=0.02, timestamp=0)

    obs = env.get_observations()
    assert set(obs.keys()) == {"scene_image", "wrist_image", "state", "joints", "gripper_state"}

    for key in ("scene_image", "wrist_image"):
        assert obs[key].dtype == np.uint8
        assert obs[key].shape == (env.grid_size, env.grid_size, 3)

    for key in ("state", "joints", "gripper_state"):
        assert obs[key].dtype == np.float32
        assert obs[key].ndim == 1

    np.testing.assert_allclose(obs["joints"], [30, 40])
    np.testing.assert_allclose(obs["gripper_state"], [0.02])
    np.testing.assert_allclose(obs["state"], [30, 40, 0.02])


def test_action_signatures_match_base_env():
    # the control loops pass these by keyword, so the names must match the abstract interface
    # that the real stations implement, not just the arity
    for method in ("act", "act_joints"):
        assert inspect.signature(getattr(MockEnv, method)) == inspect.signature(getattr(BaseEnv, method))


def test_action_spec():
    assert MockEnv.ACTION_SPEC.action_type == ActionType.ABS_EEF


def test_act_joints_raises_by_default():
    # envs that do not implement the joint path should raise instead of silently doing nothing
    with pytest.raises(NotImplementedError):
        BaseEnv.act_joints(MockEnv(), np.zeros(2), 0.0, 0)
