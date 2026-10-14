import numpy as np
import pytest

from robot_imitation_glue.preprocessing import (
    LEROBOT_KEY_MAPPING,
    ROBOTIQ_2F85_MAX_WIDTH,
    SCHUNK_TCP_OFFSET,
    abs_se3_to_policy_action_converter,
    concatenate_state,
    policy_action_to_abs_se3_converter,
    remap_to_lerobot_keys,
    tcp_offset_transform,
)


def _random_se3_pose(seed=0):
    rng = np.random.default_rng(seed)
    # orthonormal rotation matrix through QR decomposition
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = rng.standard_normal(3)
    return pose


def test_policy_action_encoding_layout():
    pose = _random_se3_pose()
    action = abs_se3_to_policy_action_converter(None, None, pose, 0.03)

    assert action.shape == (10,)
    assert action.dtype == np.float32
    np.testing.assert_allclose(action[:3], pose[:3, 3], rtol=1e-6)
    np.testing.assert_allclose(action[3:6], pose[:3, 0], rtol=1e-6)
    np.testing.assert_allclose(action[6:9], pose[:3, 1], rtol=1e-6)
    assert action[9] == pytest.approx(0.03)


def test_policy_action_se3_roundtrip():
    """collection encodes an SE3 pose into an action, inference decodes it again: both must agree."""
    pose = _random_se3_pose(seed=1)
    gripper = 0.05

    action = abs_se3_to_policy_action_converter(None, None, pose, gripper)
    decoded_pose, decoded_gripper = policy_action_to_abs_se3_converter(None, None, action)

    np.testing.assert_allclose(decoded_pose, pose, atol=1e-6)
    assert decoded_gripper == pytest.approx(gripper)


def test_policy_action_to_se3_orthonormalizes_rotation():
    """the 6D rotation of a (noisy) policy output should be mapped onto a valid rotation matrix."""
    action = np.array([0.1, 0.2, 0.3, 2.0, 0.0, 0.0, 0.5, 3.0, 0.0, 0.0], dtype=np.float32)

    pose, _ = policy_action_to_abs_se3_converter(None, None, action)
    rotation = pose[:3, :3]

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_concatenate_state():
    joints = np.arange(6, dtype=np.float64)
    gripper = np.array([0.02])

    state = concatenate_state(joints, gripper)

    assert state.shape == (7,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(state, np.array([0, 1, 2, 3, 4, 5, 0.02], dtype=np.float32))


def test_remap_to_lerobot_keys():
    frame = {"state": 1, "scene_image": 2, "wrist_image": 3, "joints": 4}

    remapped = remap_to_lerobot_keys(frame)

    assert remapped == {
        "observation.state": 1,
        "observation.images.scene_image": 2,
        "observation.images.wrist_image": 3,
        "joints": 4,  # keys that are not in the mapping are kept as-is
    }
    assert frame == {"state": 1, "scene_image": 2, "wrist_image": 3, "joints": 4}  # input is not mutated


def test_remap_to_lerobot_keys_with_custom_mapping():
    assert remap_to_lerobot_keys({"joints": 1}, {"joints": "observation.joints"}) == {"observation.joints": 1}


def test_lerobot_key_mapping_targets_are_unique():
    assert len(set(LEROBOT_KEY_MAPPING.values())) == len(LEROBOT_KEY_MAPPING)


def test_lerobot_image_tensor_roundtrip():
    pytest.importorskip("torch")
    from robot_imitation_glue.preprocessing import image_to_lerobot_tensor, lerobot_tensor_to_image

    image = np.arange(16 * 16 * 3, dtype=np.int64).reshape(16, 16, 3).astype(np.uint8)
    tensor = image_to_lerobot_tensor(image, add_batch_dim=False)

    # LeRobotDataset returns images as CHW float in [0,1], which lerobot_tensor_to_image has to scale back by 255
    assert tuple(tensor.shape) == (3, 16, 16)
    assert float(tensor.max()) <= 1.0
    np.testing.assert_array_equal(lerobot_tensor_to_image(tensor), image)


def test_tcp_offset_transform():
    transform = tcp_offset_transform()

    expected = np.eye(4)
    expected[2, 3] = SCHUNK_TCP_OFFSET
    np.testing.assert_allclose(transform, expected)


def test_delta_action_converter_roundtrip():
    pytest.importorskip("scipy")
    from robot_imitation_glue.preprocessing import (
        abs_se3_to_relative_policy_action_converter,
        delta_action_to_abs_se3_converter,
    )

    pose = _random_se3_pose(seed=2)
    gripper_state = np.array([0.04])
    delta_action = np.array([0.01, -0.02, 0.03, 0.05, -0.05, 0.1, 0.01])

    target_pose, target_gripper = delta_action_to_abs_se3_converter(pose, gripper_state, delta_action)
    relative_action = abs_se3_to_relative_policy_action_converter(pose, gripper_state, target_pose, target_gripper)

    assert relative_action.shape == (7,)
    assert relative_action.dtype == np.float32

    # the relative encoding is expressed in the current EEF frame, so re-applying it to the current pose must
    # reproduce the target pose (the delta action itself is base-aligned, so it is not simply returned).
    from scipy.spatial.transform import Rotation as R

    relative_se3 = np.eye(4)
    relative_se3[:3, :3] = R.from_euler("xyz", relative_action[3:6].astype(np.float64)).as_matrix()
    relative_se3[:3, 3] = relative_action[:3]
    np.testing.assert_allclose(pose @ relative_se3, target_pose, atol=1e-5)

    # the delta position is applied in the base frame
    np.testing.assert_allclose(target_pose[:3, 3], pose[:3, 3] + delta_action[:3], atol=1e-9)
    # and the delta rotation is applied in a base-aligned frame at the EEF
    np.testing.assert_allclose(
        target_pose[:3, :3], R.from_euler("xyz", delta_action[3:6]).as_matrix() @ pose[:3, :3], atol=1e-9
    )

    np.testing.assert_allclose(relative_action[6], delta_action[6], atol=1e-6)
    assert target_gripper == pytest.approx(gripper_state + delta_action[6])


def test_delta_action_converter_clips_gripper_width():
    pytest.importorskip("scipy")
    from robot_imitation_glue.preprocessing import delta_action_to_abs_se3_converter

    pose = np.eye(4)
    action = np.zeros(7)
    action[6] = 1.0

    _, gripper = delta_action_to_abs_se3_converter(pose, np.array([0.04]), action)
    assert gripper == pytest.approx(ROBOTIQ_2F85_MAX_WIDTH)
