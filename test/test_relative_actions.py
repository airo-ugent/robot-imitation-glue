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
    targets = [
        _absolute_action([0.4 + 0.01 * k, -0.1, 0.3], Rotation.from_euler("xyz", [3.0, 0.1 * k, 0.5]))
        for k in range(3)
    ]
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


def test_too_short_episodes_are_skipped_like_lerobots_sampler():
    states, actions = np.zeros((6, 7)), np.zeros((6, 10))
    actions[:, 3] = actions[:, 7] = 1.0
    episode_index = np.array([0, 0, 0, 0, 1, 1])  # episode 1 has 2 frames <= drop_n_last_frames
    relative = relative_action_chunks(actions, states, episode_index, horizon=2, n_obs_steps=1, drop_n_last_frames=2)
    assert relative.shape == (2 * 2, 10)  # 2 sample starts of episode 0 only


def _record_eef_dataset(root, episode_lengths):
    from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder

    obs = {"scene_image": np.zeros((16, 16, 3), dtype=np.uint8), "observation.state": np.zeros(7, dtype=np.float32)}
    recorder = LeRobotDatasetRecorder(obs, np.zeros(10, dtype=np.float32), root, "test", 10, use_videos=False)
    rng = np.random.default_rng(0)
    for length in episode_lengths:
        # a smooth motion of 1 cm per step from somewhere in the workspace, the target is the next pose
        start = rng.uniform(0.3, 0.6, 3)
        for t in range(length):
            rotation = Rotation.from_euler("xyz", [3.0, 0.02 * t, 0.5])
            position = start + [0.0, 0.0, -0.01 * t]
            state = _state(position, rotation.as_euler("xyz"))
            recorder.record_step(
                {**obs, "observation.state": state}, _absolute_action(position - [0, 0, 0.01], rotation)
            )
        recorder.save_episode()
    recorder.finish_recording()


def test_relative_stats_survive_removing_an_episode(tmp_path):
    import json
    import shutil

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from robot_imitation_glue.lerobot_dataset.relative_action_stats import write_relative_action_stats
    from robot_imitation_glue.lerobot_dataset.remove_lerobot_dataset_episode import remove_episodes

    def action_stats(root):
        return json.loads((root / "meta" / "stats.json").read_text())["action"]

    root = tmp_path / "dataset"
    _record_eef_dataset(root, [12, 10, 14])
    write_relative_action_stats(root, horizon=4, n_obs_steps=2, drop_n_last_frames=2)

    remove_episodes(LeRobotDataset(repo_id="test", root=root), [1])
    assert (root / "action_representation.json").exists()  # not deleted with the old dataset dir

    # lerobot rebuilt stats.json from the per-episode stats: they must be the relative stats of the kept episodes
    rebuilt = action_stats(root)
    recomputed_root = tmp_path / "recomputed"
    shutil.copytree(root, recomputed_root)
    write_relative_action_stats(recomputed_root, horizon=4, n_obs_steps=2, drop_n_last_frames=2)
    expected = action_stats(recomputed_root)
    for key in ("min", "max", "mean"):
        np.testing.assert_allclose(rebuilt[key], expected[key], atol=1e-5)
    np.testing.assert_allclose(rebuilt["std"], expected["std"], atol=1e-4)
    # relative positions (a few cm), not the absolute ones (0.3 .. 0.6 m)
    assert np.abs(np.array(rebuilt["max"][:3])).max() < 0.1


def test_recorder_reopens_a_dataset_without_saved_episodes(tmp_path):
    from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder

    obs = {"scene_image": np.zeros((16, 16, 3), dtype=np.uint8), "state": np.zeros(3, dtype=np.float32)}
    action = np.zeros(3, dtype=np.float32)
    LeRobotDatasetRecorder(obs, action, tmp_path / "dataset", "test", 10, False).finish_recording()  # quit, no episode

    recorder = LeRobotDatasetRecorder(obs, action, tmp_path / "dataset", "test", 10, False)
    assert recorder.n_recorded_episodes == 0
    recorder.record_step(obs, action)
    recorder.save_episode()
    assert recorder.n_recorded_episodes == 1
