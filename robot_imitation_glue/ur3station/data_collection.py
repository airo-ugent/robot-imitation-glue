import argparse
import os
from pathlib import Path

import numpy as np

from robot_imitation_glue.agents.airo_teleop_agent import (
    airo_teleop_se3_action_to_pose,
    make_phone_agent,
    make_spacemouse_agent,
)
from robot_imitation_glue.collect_data import collect_data, example_recording_observation
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder

# the delta action encoding is shared with the dataset preparation and the eval scripts,
# see robot_imitation_glue.preprocessing.
from robot_imitation_glue.preprocessing import (
    abs_se3_to_relative_policy_action_converter,
    delta_action_to_abs_se3_converter,
)
from robot_imitation_glue.ur3station.ur3_robot_env import UR3eStation

# defaults of the command line arguments: edit them here or override them on the command line.
DATASET_NAME = "test_dataset"
TELEOP_DEVICE = "phone"  # "phone" or "spacemouse"
PHONE_OS = "android"  # "android" or "ios"
# robot base axis the top of the phone points to when the agent starts: "+x", "-x", "+y" or "-y"
PHONE_FORWARD_AXIS = "-x"


def main():
    parser = argparse.ArgumentParser(description="Collect phone or SpaceMouse demonstrations on the UR3 station.")
    parser.add_argument("--dataset-name", default=DATASET_NAME, help="name of the recorded dataset")
    parser.add_argument("--teleop", default=TELEOP_DEVICE, choices=["phone", "spacemouse"], help="teleop device")
    parser.add_argument("--phone-os", default=PHONE_OS, choices=["android", "ios"])
    parser.add_argument(
        "--phone-forward-axis",
        default=PHONE_FORWARD_AXIS,
        choices=["+x", "-x", "+y", "-y"],
        help="robot base axis the top of the phone points to",
    )
    args = parser.parse_args()

    # connects to the robot, the cameras and the teleop device: only run this at the station.
    env = UR3eStation()
    if args.teleop == "phone":
        # airo-teleop's own PhoneConfig, not lerobot's: its Phone compares against airo_teleop_phone.PhoneOS
        from airo_teleop_phone.config_phone import PhoneConfig, PhoneOS

        agent = make_phone_agent(
            env, PhoneConfig(phone_os=PhoneOS(args.phone_os)), phone_forward_axis=args.phone_forward_axis
        )
        # the phone returns absolute TCP poses (and gripper widths, hold A / B to close / open)
        teleop_to_pose_converter = airo_teleop_se3_action_to_pose
    else:
        agent = make_spacemouse_agent(env)
        # the SpaceMouse returns base-aligned deltas
        teleop_to_pose_converter = delta_action_to_abs_se3_converter

    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    dataset_recorder = LeRobotDatasetRecorder(
        # with the absolute targets collect_data records next to the actions
        example_obs_dict=example_recording_observation(env, agent),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir=Path(f"datasets/{args.dataset_name}"),
        dataset_name=args.dataset_name,
        fps=10,
        use_videos=True,
    )

    # recorded actions are the same 7-dim relative EEF actions for both devices
    collect_data(
        env,
        agent,
        dataset_recorder,
        10,
        teleop_to_pose_converter,
        abs_se3_to_relative_policy_action_converter,
    )

    env.close()
    agent.close()


if __name__ == "__main__":
    main()
