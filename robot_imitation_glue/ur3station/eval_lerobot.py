"""Evaluate a LeRobot policy (e.g. Diffusion Policy) on the UR3 station.

The policy sees the wrist camera and the EEF state, as in the dataset of `ur3station/prepare_datasets.py`:
`observation.images.wrist_image` (224x224) and `observation.state` [xyz, extrinsic xyz euler, gripper width]. It
predicts absolute targets [xyz, rot6d, gripper width]; a policy trained with UMI-style relative actions
(`policy.use_relative_actions`) predicts them relative to the observed pose and `LerobotAgent` converts them back.

Between rollouts the scene is reset with the phone (or the SpaceMouse), as in data_collection.py. Keyboard: see
`eval_agent.eval`. The rollouts are recorded as a new dataset.
"""

import argparse
import os
import shutil

import numpy as np
import torch

from robot_imitation_glue.preprocessing import (
    LEROBOT_KEY_MAPPING,
    image_to_lerobot_tensor,
    policy_action_to_abs_se3_converter,
)

# defaults of the command line arguments: edit them here or override them on the command line.
CHECKPOINT_PATH = "outputs/train/green_cup_pickup_umi_diffusion/checkpoints/last/pretrained_model"
EVAL_DATASET_NAME = "green_cup_pickup_eval"
EVAL_SCENARIOS_DATASET_PATH = None  # optional dataset whose first frames show the initial scenes
TELEOP_DEVICE = "phone"  # "phone" or "spacemouse"
PHONE_OS = "android"  # "android" or "ios"
PHONE_FORWARD_AXIS = "-x"  # robot base axis the top of the phone points to: "+x", "-x", "+y" or "-y"
POLICY_ACTION_DIM = 10  # [xyz, rot6d, gripper width]


def wrist_preprocessor(obs_dict):
    """the env observation as the (batched) policy input of a wrist camera + EEF state policy."""
    return {
        LEROBOT_KEY_MAPPING["wrist_image"]: image_to_lerobot_tensor(obs_dict["wrist_image"]),
        LEROBOT_KEY_MAPPING["state"]: torch.from_numpy(obs_dict["state"]).float().unsqueeze(0),
    }


def make_teleop_agent(env, args):
    """the teleop agent to reset the scene between rollouts, and its teleop_to_pose_converter."""
    from robot_imitation_glue.agents.airo_teleop_agent import (
        airo_teleop_se3_action_to_pose,
        make_phone_agent,
        make_spacemouse_agent,
    )
    from robot_imitation_glue.preprocessing import delta_action_to_abs_se3_converter

    if args.teleop == "phone":
        from airo_teleop_phone.config_phone import PhoneConfig, PhoneOS

        phone_config = PhoneConfig(phone_os=PhoneOS(args.phone_os))
        agent = make_phone_agent(env, phone_config, phone_forward_axis=args.phone_forward_axis)
        return agent, airo_teleop_se3_action_to_pose
    return make_spacemouse_agent(env), delta_action_to_abs_se3_converter


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="pretrained_model dir of the checkpoint")
    parser.add_argument("--eval-dataset-name", default=EVAL_DATASET_NAME, help="name of the recorded eval dataset")
    parser.add_argument("--eval-scenarios", default=EVAL_SCENARIOS_DATASET_PATH, help="dataset with initial scenes")
    parser.add_argument("--teleop", default=TELEOP_DEVICE, choices=["phone", "spacemouse"], help="teleop device")
    parser.add_argument("--phone-os", default=PHONE_OS, choices=["android", "ios"])
    parser.add_argument("--phone-forward-axis", default=PHONE_FORWARD_AXIS, choices=["+x", "-x", "+y", "-y"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
    from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
    from robot_imitation_glue.eval_agent import eval
    from robot_imitation_glue.ur3station.ur3_robot_env import UR3eStation

    policy, policy_preprocessor, policy_postprocessor = make_lerobot_policy(args.checkpoint, args.device)
    policy_agent = LerobotAgent(policy, args.device, wrist_preprocessor, policy_preprocessor, policy_postprocessor)

    # connects to the robot, the cameras and the teleop device: only run this at the station.
    env = UR3eStation()
    teleop_agent, teleop_to_pose_converter = make_teleop_agent(env, args)

    eval_scenarios = None
    if args.eval_scenarios is not None:
        eval_scenarios = LeRobotDataset(repo_id="eval_scenarios", root=args.eval_scenarios)

    # the eval dataset is recorded from scratch on every run
    eval_dataset_dir = f"datasets/{args.eval_dataset_name}"
    if os.path.exists(eval_dataset_dir):
        shutil.rmtree(eval_dataset_dir)
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((POLICY_ACTION_DIM,), dtype=np.float32),
        root_dataset_dir=eval_dataset_dir,
        dataset_name=args.eval_dataset_name,
        fps=10,
        use_videos=True,
    )

    try:
        eval(
            env,
            teleop_agent,
            policy_agent,
            dataset_recorder,
            policy_to_pose_converter=policy_action_to_abs_se3_converter,
            teleop_to_pose_converter=teleop_to_pose_converter,
            fps=10,
            eval_dataset=eval_scenarios,
            eval_dataset_image_key=LEROBOT_KEY_MAPPING["wrist_image"],
            env_observation_image_key="wrist_image",
        )
    finally:
        teleop_agent.close()
        env.close()


if __name__ == "__main__":
    main()
