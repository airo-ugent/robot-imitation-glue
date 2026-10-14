from datetime import datetime
from pathlib import Path

import numpy as np

from robot_imitation_glue.agents.gello import DynamixelConfig, GelloAgent
from robot_imitation_glue.collect_data import collect_data
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder

# the action encoding is shared with the dataset preparation and the eval scripts,
# see robot_imitation_glue.preprocessing for the inverse conversion.
# both converters used to be defined in this module: keep re-exporting them so that existing
# `from robot_imitation_glue.ur5station.data_collection import ...` imports keep working.
from robot_imitation_glue.preprocessing import abs_se3_to_policy_action_converter  # noqa: F401
from robot_imitation_glue.preprocessing import policy_action_to_abs_se3_converter  # noqa: F401
from robot_imitation_glue.ur5station.ur5_robot_env import (
    GELLO_AGENT_PORT,
    UR5eStation,
    convert_abs_gello_actions_to_se3,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--dataset_name", type=str, default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    args = parser.parse_args()

    env = UR5eStation()

    config = DynamixelConfig(
        joint_ids=[1, 2, 3, 4, 5, 6],
        joint_offsets=(np.array([40, 16, 25, 40, 15, 7]) * np.pi / 16).tolist(),
        joint_signs=[1, 1, -1, 1, 1, 1],
        gripper_config=(7, 194, 152),
    )
    agent = GelloAgent(config, GELLO_AGENT_PORT)

    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros(10),
        root_dataset_dir=Path(f"datasets/{args.dataset_name}/"),
        dataset_name="none",
        fps=10,
    )

    input("Press Enter to start collecting data")
    action = agent.get_action(env.get_observations())
    initial_pose, gripper = convert_abs_gello_actions_to_se3(
        env.get_robot_pose_se3(), env.get_gripper_opening(), action
    )
    # first move robot slowly to the current teleop pose.
    env.robot.move_linear_to_tcp_pose(initial_pose).wait()

    collect_data(
        env,
        agent,
        dataset_recorder,
        frequency=10,
        teleop_to_pose_converter=convert_abs_gello_actions_to_se3,
        abs_pose_to_policy_action=abs_se3_to_policy_action_converter,
    )
