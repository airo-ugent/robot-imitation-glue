import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.dataset_tools")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.lerobot_dataset.add_description_lerobot_dataset import add_description
from robot_imitation_glue.lerobot_dataset.remove_lerobot_dataset_episode import (
    parse_episodes_list,
    remove_episodes,
)


@pytest.fixture
def recorded_dataset(tmp_path):
    """3 episodes of 2, 3 and 4 frames, recorded like collect_data does (so with the empty task "")."""
    obs = {"scene_image": np.zeros((16, 16, 3), dtype=np.uint8), "state": np.zeros(3, dtype=np.float32)}
    recorder = LeRobotDatasetRecorder(obs, np.zeros(3, dtype=np.float32), tmp_path / "dataset", "test", 10, False)
    for episode, n_frames in enumerate([2, 3, 4]):
        for _ in range(n_frames):
            recorder.record_step(obs, np.full(3, episode, dtype=np.float32))
        recorder.save_episode()
    recorder.finish_recording()
    return LeRobotDataset(repo_id="test", root=tmp_path / "dataset")


def test_parse_episodes_list():
    assert parse_episodes_list("1-3,7,2") == [1, 2, 3, 7]


def test_remove_episodes_renumbers_and_keeps_a_backup(recorded_dataset, tmp_path):
    dataset = remove_episodes(recorded_dataset, [1], backup=tmp_path / "backup")

    assert dataset.meta.total_episodes == 2
    assert dataset.meta.total_frames == 6
    np.testing.assert_array_equal([int(e) for e in dataset.hf_dataset["episode_index"]], [0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(dataset[2]["action"].numpy(), [2, 2, 2])
    assert LeRobotDataset(repo_id="test", root=tmp_path / "backup").meta.total_episodes == 3


def test_add_description_to_some_episodes_of_a_recorded_dataset(recorded_dataset):
    # the other episodes keep the empty task, which lerobot's modify_tasks refuses unless they are listed
    add_description(recorded_dataset, [0, 2], task_description="pick the cube")

    dataset = LeRobotDataset(repo_id="test", root=recorded_dataset.root)
    tasks = [dataset[dataset.meta.episodes[ep]["dataset_from_index"]]["task"] for ep in range(3)]
    assert tasks == ["pick the cube", "", "pick the cube"]
