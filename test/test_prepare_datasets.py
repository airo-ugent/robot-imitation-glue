import numpy as np
import pytest

pytest.importorskip("ur_analytic_ik")
pytest.importorskip("lerobot.datasets.lerobot_dataset")
torch = pytest.importorskip("torch")

from ur_analytic_ik import ur5e  # noqa: E402

from robot_imitation_glue.preprocessing import abs_se3_to_policy_action_converter, tcp_offset_transform  # noqa: E402
from robot_imitation_glue.ur5station.prepare_datasets import joints_frame_transform  # noqa: E402


@pytest.mark.parametrize("q0", [-3.3, -3.1, -2.0, 0.5])
def test_joints_frame_transform(q0):
    """the joint action must be the IK of the recorded EEF action, close to the current joints (no 2pi offset)."""
    current_joints = np.array([q0, -1.4, 1.6, -1.8, -1.5, 0.3])
    target_joints = current_joints + 0.05
    target_pose = ur5e.forward_kinematics_with_tcp(*target_joints, tcp_offset_transform())
    commanded_gripper, measured_gripper = 0.02, 0.035
    frame = {
        "joints": torch.tensor(current_joints, dtype=torch.float32),
        "gripper_state": torch.tensor([measured_gripper], dtype=torch.float32),
        "action": torch.tensor(abs_se3_to_policy_action_converter(None, None, target_pose, commanded_gripper)),
    }

    new_frame = joints_frame_transform(frame)

    np.testing.assert_allclose(new_frame["action"][:6], target_joints, atol=1e-4)
    assert new_frame["action"][6] == pytest.approx(commanded_gripper)
    np.testing.assert_allclose(new_frame["observation.state"][:6], current_joints, atol=1e-6)
    # the state holds the measured gripper width, like the eval preprocessors
    assert new_frame["observation.state"][6] == pytest.approx(measured_gripper)
