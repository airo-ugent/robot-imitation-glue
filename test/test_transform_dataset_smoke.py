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
