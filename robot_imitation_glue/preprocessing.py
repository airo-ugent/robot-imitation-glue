"""Preprocessing steps that are shared between data collection, dataset preparation and inference.

Observations and actions pass through a number of conversions before they reach a policy:
the SE3 target pose is encoded into a flat action vector, the env observation keys are renamed to the
LeRobot convention, images are converted to CHW float tensors, ...

These conversions have to be *identical* in all three code paths:

1. data collection      (`collect_data.py` + the station `data_collection.py` scripts)
2. dataset preparation  (`lerobot_dataset/transform_dataset.py` + the station `prepare_datasets.py` scripts)
3. inference            (`eval_agent.py` + the station `eval_*.py` scripts)

A silent mismatch between them (a different key name, a different rotation encoding, an image that is
not scaled to [0,1], ...) does not raise an error but simply degrades the policy, which is very hard to debug.
Therefore this module is the single source of truth: all three paths import these functions instead of
re-implementing them, so changing a convention here changes it everywhere at once.

This module only depends on numpy (torch and scipy are imported lazily) so that it can be imported without
any hardware or deep learning dependencies.
"""

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# robot embodiment constants
# ---------------------------------------------------------------------------------------------------------------

# z-offset of the TCP w.r.t. the robot flange for the Schunk EGK40 gripper.
# lives here because both the online env (forward kinematics) and the offline dataset preparation
# (inverse kinematics) need it, and the offline path cannot import the hardware drivers.
SCHUNK_TCP_OFFSET = 0.184

# stroke of the Robotiq 2F-85 gripper, used to clip absolute gripper widths.
ROBOTIQ_2F85_MAX_WIDTH = 0.085


def tcp_offset_transform(z_offset: float = SCHUNK_TCP_OFFSET) -> np.ndarray:
    """4x4 transform of the TCP frame w.r.t. the robot flange frame."""
    tcp_transform = np.eye(4)
    tcp_transform[2, 3] = z_offset
    return tcp_transform


# ---------------------------------------------------------------------------------------------------------------
# action encoding: SE3 pose + gripper width <-> flat policy action
# ---------------------------------------------------------------------------------------------------------------


def abs_se3_to_policy_action_converter(robot_pose, gripper_pose, abs_se3_action, gripper_action):
    """absolute poses, encoded as position, x-vector of rotation, y-vector of rotation, gripper action"""
    abs_position_target = abs_se3_action[:3, 3]
    abs_rotation_x = abs_se3_action[:3, 0]
    abs_rotation_y = abs_se3_action[:3, 1]

    policy_action = np.zeros(10)
    policy_action[:3] = abs_position_target
    policy_action[3:6] = abs_rotation_x
    policy_action[6:9] = abs_rotation_y
    policy_action[9] = gripper_action
    policy_action = policy_action.astype(np.float32)
    return policy_action


def policy_action_to_abs_se3_converter(robot_pose, gripper_pose, policy_action):
    """inverse of `abs_se3_to_policy_action_converter`, the 6D rotation is orthonormalized (gram-schmidt)."""
    policy_action = policy_action.astype(np.float64)
    abs_position_target = policy_action[:3]
    rot6D = policy_action[3:9]
    gripper_action = policy_action[9]

    # convert to se3
    x = rot6D[:3] / np.linalg.norm(rot6D[:3])
    y = rot6D[3:] - np.dot(rot6D[3:], x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)

    target_pose = np.eye(4)
    target_pose[:3, 3] = abs_position_target
    target_pose[:3, 0] = x
    target_pose[:3, 1] = y
    target_pose[:3, 2] = z
    return target_pose, gripper_action


def delta_action_to_abs_se3_converter(robot_pose_se3, gripper_state, action, max_gripper_width=ROBOTIQ_2F85_MAX_WIDTH):
    """convert a delta (teleop) action [dx,dy,dz, drx,dry,drz, dgripper] to an absolute SE3 pose + gripper width.

    we take the action to consist of a delta position, delta rotation and delta gripper width.
    the delta rotation is interpreted as expressed in a frame with the same orientation as the base frame but with
    the origin at the EEF. in this way, when rotating the teleop device, the robot eef will not move around, while at
    the same time the axes of orientation do not depend on the current orientation of the EEF.

    the delta position is intepreted in the world frame and also applied on the EEF frame.

    CONVENTION: the delta rotation is interpreted as XYZ **euler angles**, not as a rotation vector
    (similar to Diffusion Policy). Rotation vectors seem more principled (related to twist), but recorded datasets
    depend on this convention, so do not change it without re-recording or converting the datasets.
    """
    from scipy.spatial.transform import Rotation as R

    delta_pos = action[:3]
    delta_rot = action[3:6]
    gripper_action = action[6]

    robot_trans = robot_pose_se3[:3, 3]
    robot_SO3 = robot_pose_se3[:3, :3]

    new_robot_trans = robot_trans + delta_pos
    new_robot_SO3 = R.from_euler("xyz", delta_rot).as_matrix() @ robot_SO3

    new_robot_SE3 = np.eye(4)
    new_robot_SE3[:3, :3] = new_robot_SO3
    new_robot_SE3[:3, 3] = new_robot_trans

    new_gripper_state = gripper_state + gripper_action
    new_gripper_state = np.clip(new_gripper_state, 0, max_gripper_width)

    return new_robot_SE3, new_gripper_state


def abs_se3_to_relative_policy_action_converter(robot_pose, gripper_pose, abs_se3_action, gripper_action):
    """encode an absolute SE3 target pose as a relative action w.r.t. the current pose.

    CONVENTION: the relative rotation is encoded as XYZ **euler angles**, matching
    `delta_action_to_abs_se3_converter`. see the note in that function before changing this.
    """
    from scipy.spatial.transform import Rotation as R

    relative_se3 = np.linalg.inv(robot_pose) @ abs_se3_action

    relative_pos = relative_se3[:3, 3]
    relative_euler = R.from_matrix(relative_se3[:3, :3]).as_euler("xyz")
    relative_gripper = gripper_action - gripper_pose

    return np.concatenate((relative_pos, relative_euler, relative_gripper), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# proprioception
# ---------------------------------------------------------------------------------------------------------------


def concatenate_state(*state_components) -> np.ndarray:
    """concatenate proprioception components (e.g. joints + gripper width) into a single float32 state vector."""
    return np.concatenate(state_components).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# key remapping between the env observation dict and the LeRobot naming convention
# ---------------------------------------------------------------------------------------------------------------

LEROBOT_KEY_MAPPING = {
    "state": "observation.state",
    "scene_image": "observation.images.scene_image",
    "wrist_image": "observation.images.wrist_image",
}

# original, full-resolution images that should not end up in the training dataset.
ORIGINAL_IMAGE_KEYS = ["wrist_image_original", "scene_image_original"]


def remap_to_lerobot_keys(dictionary: dict, key_mapping: dict = None) -> dict:
    """rename the env observation keys to the LeRobot naming convention.

    works both on a frame dict and on a features dict, keys that are not in the mapping are kept as-is.
    """
    key_mapping = LEROBOT_KEY_MAPPING if key_mapping is None else key_mapping
    return {key_mapping.get(key, key): value for key, value in dictionary.items()}


# ---------------------------------------------------------------------------------------------------------------
# image preprocessing for (torch) policies
# ---------------------------------------------------------------------------------------------------------------


def image_to_lerobot_tensor(image, add_batch_dim: bool = True):
    """convert a HWC uint8 image to the CHW float32 [0,1] tensor that LeRobot policies expect."""
    import torch

    tensor = torch.tensor(image).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)
    return tensor


def lerobot_tensor_to_image(tensor) -> np.ndarray:
    """inverse of `image_to_lerobot_tensor`: convert a CHW float [0,1] tensor to a HWC uint8 image."""
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = image * 255
    return image.astype(np.uint8)


def lerobot_observation_from_env_observation(scene_image, wrist_image, state) -> dict:
    """build the (batched) observation dict for a LeRobot policy from the raw env observation components."""
    import torch

    return {
        LEROBOT_KEY_MAPPING["scene_image"]: image_to_lerobot_tensor(scene_image),
        LEROBOT_KEY_MAPPING["wrist_image"]: image_to_lerobot_tensor(wrist_image),
        LEROBOT_KEY_MAPPING["state"]: torch.tensor(state).float().unsqueeze(0),
    }
