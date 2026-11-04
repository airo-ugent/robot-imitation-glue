"""the UR3 UMI dataset preparation takes the recorded absolute targets as actions and drops the idle frames."""

import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.lerobot_dataset")

from robot_imitation_glue.collect_data import target_observation  # noqa: E402
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder  # noqa: E402
from robot_imitation_glue.preprocessing import abs_se3_to_policy_action_converter  # noqa: E402
from robot_imitation_glue.ur3station.prepare_datasets import abs_actions_and_keep_mask  # noqa: E402


def _record(root, with_targets=True):
    """one episode as collect_data records it: the second step repeats the target of the first."""
    obs = {"state": np.zeros(7, dtype=np.float32)}
    target, gripper = np.eye(4), 0.08
    example = {**obs, **target_observation(gripper, target_pose=target)} if with_targets else obs
    recorder = LeRobotDatasetRecorder(example, np.zeros(7, dtype=np.float32), root, "test", fps=10, use_videos=False)
    recorder.start_episode()
    expected = []
    for step in ([0.01, 0, 0], [0, 0, 0], [0, 0.02, -0.01]):
        target = target.copy()
        target[:3, 3] += step
        gripper = gripper - 0.01 if any(step) else gripper
        recorder.record_step({**obs, **target_observation(gripper, target_pose=target)}, np.zeros(7, np.float32))
        expected.append(abs_se3_to_policy_action_converter(None, None, target, gripper))
    recorder.save_episode()
    recorder.finish_recording()
    return expected


def test_absolute_targets_and_idle_frames(tmp_path):
    expected = _record(tmp_path / "dataset")

    lookup = abs_actions_and_keep_mask(tmp_path / "dataset", keep_idle=False)

    assert lookup[(0, 1)] is None  # the idle step is dropped
    np.testing.assert_allclose(lookup[(0, 0)], expected[0], atol=1e-6)
    np.testing.assert_allclose(lookup[(0, 2)], expected[2], atol=1e-6)
    np.testing.assert_allclose(abs_actions_and_keep_mask(tmp_path / "dataset", keep_idle=True)[(0, 1)], expected[1])


def test_dataset_without_targets_is_refused(tmp_path):
    _record(tmp_path / "dataset", with_targets=False)
    with pytest.raises(ValueError, match="target_pose"):
        abs_actions_and_keep_mask(tmp_path / "dataset", keep_idle=False)
