"""UMI-style relative actions: LerobotAgent's conversion back to absolute and the dataset stats helper."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
processor = pytest.importorskip("lerobot.processor")
pytest.importorskip("lerobot.processor.se3_relative_action_processor")

from lerobot.processor import absolute_to_relative_pose_actions  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from robot_imitation_glue.agents.lerobot_agent import LerobotAgent  # noqa: E402
from robot_imitation_glue.lerobot_dataset.relative_action_stats import relative_action_chunks  # noqa: E402


def _state(position, euler):
    return np.concatenate([position, euler, [0.04]]).astype(np.float32)  # [xyz, extrinsic xyz euler, gripper]


def _absolute_action(position, rotation: Rotation, gripper=0.02):
    matrix = rotation.as_matrix()
    return np.concatenate([position, matrix[:, 0], matrix[:, 1], [gripper]]).astype(np.float32)


class _FakeRelativePolicy:
    """Queues a chunk of relative actions (computed from the chunk's absolute targets and the state of the
    observation it is predicted from), like DiffusionPolicy with n_action_steps = len(chunk)."""

    def __init__(self, absolute_chunk):
        self.config = SimpleNamespace(use_relative_actions=True)
        self.absolute_chunk = torch.from_numpy(np.stack(absolute_chunk))
        self.reset()

    def reset(self):
        self._queues = {ACTION: deque()}

    def select_action(self, batch):
        if len(self._queues[ACTION]) == 0:
            relative = absolute_to_relative_pose_actions(self.absolute_chunk.unsqueeze(0), batch[OBS_STATE])
            self._queues[ACTION].extend(relative.transpose(0, 1))
        return self._queues[ACTION].popleft()


class _Identity:
    def __call__(self, x):
        return x

    def reset(self):
        pass


def test_agent_converts_a_chunk_back_with_the_pose_it_was_predicted_from():
    targets = [_absolute_action([0.4 + 0.01 * k, -0.1, 0.3], Rotation.from_euler("xyz", [3.0, 0.1 * k, 0.5])) for k in range(3)]
    policy = _FakeRelativePolicy(targets)
    agent = LerobotAgent(
        policy, "cpu", lambda obs: {OBS_STATE: torch.from_numpy(obs["state"]).unsqueeze(0)}, _Identity(), _Identity()
    )

    # the robot moves while the chunk is executed: later actions must still use the pose of the first observation
    states = [_state([0.4, -0.1, 0.3 + 0.05 * k], [3.0, 0.2 * k, 0.4]) for k in range(3)]
    for state, target in zip(states, targets):
        np.testing.assert_allclose(agent.get_action({"state": state}), target, atol=1e-5)

    # after reset a new chunk is predicted from the new observation
    agent.reset()
    np.testing.assert_allclose(agent.get_action({"state": states[2]}), targets[0], atol=1e-5)


def test_relative_action_chunks_follow_lerobot_sampling():
    # 2 episodes of 4 frames, horizon 3, n_obs_steps 2, drop the last frame of each episode
    states = np.zeros((8, 7))
    states[:, 0] = np.arange(8)  # the reference position is the frame index, the rotation is identity
    actions = np.zeros((8, 10))
    actions[:, 0] = 100 + np.arange(8)
    actions[:, 3] = actions[:, 7] = 1.0  # identity rot6d
    episode_index = np.repeat([0, 1], 4)

    relative = relative_action_chunks(actions, states, episode_index, horizon=3, n_obs_steps=2, drop_n_last_frames=1)
    # 3 sample starts per episode, 3 actions each; chunk of t is actions t-1 .. t+1, clipped to the episode
    assert relative.shape == (2 * 3 * 3, 10)
    expected_x = []
    for first in (0, 4):
        for t in range(first, first + 3):
            expected_x += [100 + np.clip(t + o, first, first + 3) - t for o in (-1, 0, 1)]
    np.testing.assert_allclose(relative[:, 0], expected_x)
    np.testing.assert_allclose(relative[:, 3:9], np.tile([1, 0, 0, 0, 1, 0], (18, 1)), atol=1e-12)
