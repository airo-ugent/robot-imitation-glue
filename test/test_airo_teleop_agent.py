"""Tests for the airo-teleop adapter.

These use fake, duck-typed teleop agents, so airo-teleop does not have to be installed.
"""

import numpy as np
import pytest

from robot_imitation_glue.agents.airo_teleop_agent import (
    SE3_ACTION_DIM,
    AiroTeleopAgent,
    airo_teleop_se3_action_to_pose,
)


class FakeTeleopAgent:
    """duck-typed stand-in for an airo_teleop_agents TeleopAgent."""

    def __init__(self, action, teleop_device=None):
        self.action = action
        self.teleop_device = teleop_device
        self.n_reset_calls = 0

    def get_action(self):
        return self.action

    def reset(self):
        self.n_reset_calls += 1


def example_pose():
    # 90 degree rotation around z + a translation
    pose = np.eye(4)
    pose[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose[:3, 3] = np.array([0.1, -0.2, 0.3])
    return pose


def test_joint_action_is_passed_through():
    joint_action = np.array([0.0, -1.5, 1.0, -1.0, -1.5, 0.0, 0.05])
    agent = AiroTeleopAgent(FakeTeleopAgent(joint_action))

    action = agent.get_action({"some": "observation"})

    assert isinstance(action, np.ndarray)
    np.testing.assert_allclose(action, joint_action)


def test_pose_action_is_flattened():
    pose = example_pose()
    agent = AiroTeleopAgent(FakeTeleopAgent(pose))

    action = agent.get_action()

    assert action.shape == (SE3_ACTION_DIM,)
    np.testing.assert_allclose(action[:3], pose[:3, 3])
    np.testing.assert_allclose(action[3:6], pose[:3, 0])
    np.testing.assert_allclose(action[6:9], pose[:3, 1])


def test_pose_and_gripper_action_is_flattened():
    pose = example_pose()
    agent = AiroTeleopAgent(FakeTeleopAgent((pose, 0.042)))

    action = agent.get_action()

    assert action.shape == (SE3_ACTION_DIM + 1,)
    np.testing.assert_allclose(action[:3], pose[:3, 3])
    assert action[SE3_ACTION_DIM] == pytest.approx(0.042)


def test_converter_roundtrips_pose_and_gripper():
    pose = example_pose()
    agent = AiroTeleopAgent(FakeTeleopAgent((pose, 0.042)))

    action = agent.get_action()
    converted_pose, converted_gripper = airo_teleop_se3_action_to_pose(np.eye(4), 0.0, action)

    np.testing.assert_allclose(converted_pose, pose, atol=1e-9)
    assert converted_gripper == pytest.approx(0.042)


def test_converter_keeps_gripper_state_for_pose_only_agents():
    agent = AiroTeleopAgent(FakeTeleopAgent(example_pose()))

    action = agent.get_action()
    _, converted_gripper = airo_teleop_se3_action_to_pose(np.eye(4), 0.07, action)

    assert converted_gripper == pytest.approx(0.07)


def test_reset_is_delegated():
    teleop_agent = FakeTeleopAgent(example_pose())
    agent = AiroTeleopAgent(teleop_agent)

    agent.reset()

    assert teleop_agent.n_reset_calls == 1


def test_close_closes_the_device():
    """airo-teleop agents have no close(), but e.g. the SpaceMouse device runs a reading thread."""

    class FakeDevice:
        closed = False

        def close(self):
            self.closed = True

    device = FakeDevice()
    AiroTeleopAgent(FakeTeleopAgent(example_pose(), teleop_device=device)).close()

    assert device.closed


def test_action_spec_is_exposed():
    action_spec = object()
    agent = AiroTeleopAgent(FakeTeleopAgent(example_pose()), action_spec=action_spec)

    assert agent.action_spec is action_spec


def test_teleop_device_is_exposed():
    """raw device inputs (e.g. the phone buttons) must remain reachable through the adapter."""
    device = object()
    agent = AiroTeleopAgent(FakeTeleopAgent(example_pose(), teleop_device=device))

    assert agent.teleop_device is device
    assert agent.teleop_agent.teleop_device is device


def test_invalid_teleop_agent_raises():
    with pytest.raises(TypeError):
        AiroTeleopAgent(object())


def test_invalid_action_shape_raises():
    agent = AiroTeleopAgent(FakeTeleopAgent(np.zeros((3, 3))))

    with pytest.raises(ValueError):
        agent.get_action()


def test_phone_gripper_agent_integrates_a_b_buttons(monkeypatch):
    """holding A closes and holding B opens the gripper in deltas, clipped to the gripper range."""
    from types import SimpleNamespace

    import robot_imitation_glue.agents.airo_teleop_agent as module
    from robot_imitation_glue.agents.airo_teleop_agent import PhoneGripperAgent, airo_teleop_se3_action_to_pose

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    device = SimpleNamespace(last_raw_inputs={})
    teleop_agent = SimpleNamespace(get_action=lambda: np.eye(4), teleop_device=device)
    agent = PhoneGripperAgent(teleop_agent, lambda: 0.04, min_width=0.0, max_width=0.085, speed=0.05)

    def step(buttons):
        clock[0] += 0.1
        device.last_raw_inputs = buttons
        return agent.get_action()

    # starts from the measured width, no button: unchanged
    assert agent.get_action()[-1] == pytest.approx(0.04)
    assert step({})[-1] == pytest.approx(0.04)
    assert step({"reservedButtonA": True})[-1] == pytest.approx(0.035)
    assert step({"reservedButtonB": True})[-1] == pytest.approx(0.04)
    for _ in range(20):
        action = step({"reservedButtonB": True})
    assert action[-1] == pytest.approx(0.085)
    pose, gripper = airo_teleop_se3_action_to_pose(None, 0.0, action)
    np.testing.assert_allclose(pose, np.eye(4))
    assert gripper == pytest.approx(0.085)

    # a long pause is not one large step
    clock[0] += 10.0
    device.last_raw_inputs = {"reservedButtonA": True}
    assert agent.get_action()[-1] == pytest.approx(0.075)

    # reset restarts from the measured width
    agent.reset()
    assert agent.get_action()[-1] == pytest.approx(0.04)
