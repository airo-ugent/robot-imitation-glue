import numpy as np
import pytest

pytest.importorskip("lerobot.common.datasets.lerobot_dataset")

from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder

EXAMPLE_OBS = {
    "scene_image": np.zeros((16, 16, 3), dtype=np.uint8),
    "state": np.array([0.1, 0.2, 0.3], dtype=np.float32),
}
EXAMPLE_ACTION = np.array([0.1, 0.2, 0.3], dtype=np.float32)


def _make_recorder(root):
    return LeRobotDatasetRecorder(
        example_obs_dict=EXAMPLE_OBS,
        example_action=EXAMPLE_ACTION,
        root_dataset_dir=root,
        dataset_name="test_dataset",
        fps=10,
        use_videos=False,
    )


@pytest.fixture
def recorder(tmp_path):
    return _make_recorder(tmp_path / "dataset")


def _record_steps(recorder, n, value=0.0):
    recorder.start_episode()
    obs = {"scene_image": np.full_like(EXAMPLE_OBS["scene_image"], int(value)), "state": EXAMPLE_OBS["state"] + value}
    for _ in range(n):
        recorder.record_step(obs, EXAMPLE_ACTION)


def _saved_column(recorder, key):
    return np.stack([np.asarray(x) for x in recorder.lerobot_dataset.hf_dataset[key]])


def test_set_episode_success_labels_all_recorded_frames(recorder):
    _record_steps(recorder, 3)
    recorder.set_episode_success(True)

    episode_buffer = recorder.lerobot_dataset.episode_buffer
    assert episode_buffer["size"] == 3
    assert all(bool(value) for value in episode_buffer["next.success"])

    recorder.save_episode()
    assert recorder.n_recorded_episodes == 1


def test_episode_success_label_is_persisted_per_episode(recorder):
    # episode-level label: every frame of a successful episode is True, every frame of a failed one False
    _record_steps(recorder, 3)
    recorder.set_episode_success(True)
    recorder.save_episode()

    _record_steps(recorder, 2)
    recorder.set_episode_success(False)
    recorder.save_episode()

    success = _saved_column(recorder, "next.success").reshape(-1)
    episode_index = _saved_column(recorder, "episode_index").reshape(-1)
    np.testing.assert_array_equal(episode_index, [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(success, [True, True, True, False, False])


def test_record_step_defaults_to_failure(recorder):
    _record_steps(recorder, 2)
    assert not any(bool(value) for value in recorder.lerobot_dataset.episode_buffer["next.success"])


def test_delete_episode_discards_current_episode(recorder):
    _record_steps(recorder, 2)
    recorder.save_episode()

    _record_steps(recorder, 2)
    recorder.delete_episode()

    assert recorder.lerobot_dataset.episode_buffer["size"] == 0
    assert recorder.n_recorded_episodes == 1


def test_episode_after_delete_is_saved_with_next_index(recorder):
    # the discarded episode must leave no trace: the next one gets its index and its own frames/images
    _record_steps(recorder, 2, value=1.0)
    recorder.save_episode()

    _record_steps(recorder, 4, value=2.0)
    recorder.delete_episode()

    _record_steps(recorder, 3, value=3.0)
    recorder.set_episode_success(True)
    recorder.save_episode()

    dataset = recorder.lerobot_dataset
    assert recorder.n_recorded_episodes == 2
    assert dataset.meta.total_episodes == 2
    assert dataset.meta.total_frames == 5
    np.testing.assert_array_equal(_saved_column(recorder, "episode_index").reshape(-1), [0, 0, 1, 1, 1])
    np.testing.assert_allclose(_saved_column(recorder, "state")[2:], np.tile(EXAMPLE_OBS["state"] + 3.0, (3, 1)))
    # images go through the async writer; check that episode 1 contains the images of the kept episode
    for item in (dataset[i] for i in range(2, 5)):
        np.testing.assert_allclose(item["scene_image"].numpy(), 3.0 / 255.0, atol=1e-6)


def test_success_and_delete_are_noops_without_episode_buffer(recorder):
    # a dataset loaded from disk (the recorder appends to an existing dir) has no episode buffer until the
    # first frame is added. Simulated here: reloading needs datasets<4 with the pinned lerobot.
    recorder.lerobot_dataset.episode_buffer = None
    recorder.set_episode_success(True)
    recorder.delete_episode()
    assert recorder.lerobot_dataset.episode_buffer is None

    _record_steps(recorder, 2)
    recorder.set_episode_success(True)
    recorder.save_episode()
    assert recorder.lerobot_dataset.meta.total_frames == 2
    np.testing.assert_array_equal(_saved_column(recorder, "next.success").reshape(-1), [True, True])
