import numpy as np
from ur_analytic_ik import ur5e

from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset
from robot_imitation_glue.preprocessing import (
    ORIGINAL_IMAGE_KEYS,
    concatenate_state,
    policy_action_to_abs_se3_converter,
    remap_to_lerobot_keys,
    tcp_offset_transform,
)

# scenarios:

# 1. convert to joint actions and joint configuration state with absolute gripper


features_to_drop = ORIGINAL_IMAGE_KEYS


def features_transform(features):
    features = remap_to_lerobot_keys(features)
    features["observation.state"]["shape"] = (7,)
    features["action"]["shape"] = (7,)

    print("processed features:")
    print(features)
    return features


def eef_features_transform(features):
    features = remap_to_lerobot_keys(features)
    features["observation.state"]["shape"] = (7,)
    features["action"]["shape"] = (10,)

    print("processed features:")
    print(features)
    return features


def joints_frame_transform(frame):
    current_joints = frame["joints"].numpy()
    action_rot6d = frame["action"].numpy()

    # convert the action into an SE3 pose
    robot_se3_action, gripper_action = policy_action_to_abs_se3_converter(None, None, action_rot6d)
    gripper_action = np.array([gripper_action])
    # get the pose without TCP transform
    tcp_transform = tcp_offset_transform()

    robot_joint_action = ur5e.inverse_kinematics_closest_with_tcp(robot_se3_action, tcp_transform, *current_joints)[0]
    robot_joint_action[0] -= 2 * np.pi  # dirty hack to fix a bug in ur analytic IK
    robot_joint_action = np.array(robot_joint_action)

    new_frame = remap_to_lerobot_keys(frame)
    new_frame["observation.state"] = concatenate_state(current_joints, gripper_action)
    new_frame["action"] = concatenate_state(robot_joint_action, gripper_action)

    return new_frame


def ee_pose_frame_transform(frame):
    return remap_to_lerobot_keys(frame)


if __name__ == "__main__":
    # rewrites a dataset: edit the paths/transforms below, then run this module as a script.
    transform_dataset(
        root_dir="datasets/pour-cup-v2",
        new_root_dir="datasets/pour-cup-v2-remapped-joints",
        transform_fn=joints_frame_transform,
        transform_features_fn=features_transform,
        features_to_drop=features_to_drop,
        episodes_to_drop=[12],
    )
