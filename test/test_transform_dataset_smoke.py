import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.lerobot_dataset")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset


def test_transformed_dataset_keeps_frames_and_task(tmp_path):
    """transform_dataset must pass the task through to lerobot's add_frame (it is a key of the frame)."""
    obs = {"scene_image": np.zeros((16, 16, 3), dtype=np.uint8), "state": np.array([0.1, 0.2, 0.3], dtype=np.float32)}
    action = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    recorder = LeRobotDatasetRecorder(
        example_obs_dict=obs,
        example_action=action,
        root_dataset_dir=tmp_path / "dataset",
        dataset_name="test_dataset",
        fps=10,
        use_videos=False,
    )
    recorder.start_episode()
    for _ in range(3):
        recorder.record_step(obs, action)
    recorder.save_episode()
    recorder.finish_recording()

    def double_action(frame):
        frame["action"] = frame["action"] * 2
        return frame

    new_dataset = transform_dataset(
        root_dir=tmp_path / "dataset",
        new_root_dir=tmp_path / "transformed",
        transform_fn=double_action,
        use_videos=False,
        verbose=False,
    )

    assert new_dataset.meta.total_frames == 3
    np.testing.assert_allclose(new_dataset[0]["action"].numpy(), 2 * action)
    assert new_dataset[0]["task"] == LeRobotDataset(repo_id="test_dataset", root=tmp_path / "dataset")[0]["task"]


def test_transform_can_drop_frames_and_episodes(tmp_path):
    """a transform returning None drops the frame, an episode without frames left is dropped altogether."""
    obs = {"state": np.array([0.0], dtype=np.float32)}
    recorder = LeRobotDatasetRecorder(
        example_obs_dict=obs,
        example_action=np.zeros(1, dtype=np.float32),
        root_dataset_dir=tmp_path / "dataset",
        dataset_name="test_dataset",
        fps=10,
        use_videos=False,
    )
    # episode 0: actions 0..3, episode 1: only zeros, episode 2: actions 0..2
    for actions in ([0, 1, 2, 3], [0, 0], [0, 1, 2]):
        recorder.start_episode()
        for a in actions:
            recorder.record_step(obs, np.array([a], dtype=np.float32))
        recorder.save_episode()
    recorder.finish_recording()

    seen = []

    def drop_zero_actions(frame):
        seen.append((frame["episode_index"], frame["frame_index"]))
        return None if frame["action"].item() == 0 else frame

    new_dataset = transform_dataset(
        root_dir=tmp_path / "dataset",
        new_root_dir=tmp_path / "transformed",
        transform_fn=drop_zero_actions,
        use_videos=False,
        verbose=False,
    )

    assert seen[:5] == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
    assert new_dataset.meta.total_episodes == 2
    assert new_dataset.meta.total_frames == 5
    assert [new_dataset[i]["action"].item() for i in range(5)] == [1, 2, 3, 1, 2]
    assert [int(new_dataset[i]["episode_index"]) for i in range(5)] == [0, 0, 0, 1, 1]
