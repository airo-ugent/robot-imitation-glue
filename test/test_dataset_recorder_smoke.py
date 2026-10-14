import numpy as np
import pytest

pytest.importorskip("lerobot.common.datasets.lerobot_dataset")

from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder


def test_recorded_episode_is_saved(tmp_path):
    """record_step must match the pinned lerobot's add_frame signature (the task is an argument there)."""
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
    assert recorder.n_recorded_episodes == 1
    assert recorder.lerobot_dataset.meta.total_frames == 3
