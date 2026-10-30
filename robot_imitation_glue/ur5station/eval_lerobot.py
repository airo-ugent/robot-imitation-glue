"""Evaluate a LeRobot policy (ACT, Diffusion Policy, ...) on the UR5 station.

The policy's state and action space has to match the dataset it was trained on:
- `--state eef`: EEF pose state and actions, as recorded by data_collection.py (e.g. Diffusion Policy)
- `--state joints`: joint state and actions, from the joint dataset of prepare_datasets.py (e.g. ACT)
"""

import argparse

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robot_imitation_glue.agents.airo_teleop_agent import make_gello_agent
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.eval_agent import eval
from robot_imitation_glue.preprocessing import (
    concatenate_state,
    lerobot_observation_from_env_observation,
    policy_action_to_abs_se3_converter,
)
from robot_imitation_glue.ur5station.ur5_robot_env import (
    GELLO_AGENT_PORT,
    GELLO_CONFIG,
    UR5eStation,
    abs_joint_policy_action_to_se3,
    convert_abs_gello_actions_to_se3,
)

# defaults of the command line arguments: edit them here or override them on the command line.
CHECKPOINT_PATH = "outputs/train/pick-cube_diffusion/checkpoints/070000/pretrained_model"
STATE = "eef"
EVAL_SCENARIOS_DATASET_PATH = "datasets/pick-cube-eval-scenarios"
EVAL_DATASET_NAME = "pick-cube-eval-DP"


def eef_preprocessor(obs_dict):
    # eef pose state, as recorded by the env during data collection
    return lerobot_observation_from_env_observation(
        obs_dict["scene_image"], obs_dict["wrist_image"], obs_dict["state"]
    )


def joints_preprocessor(obs_dict):
    # joint state, matching the joint dataset transform in prepare_datasets.py
    state = concatenate_state(obs_dict["joints"], obs_dict["gripper_state"])
    return lerobot_observation_from_env_observation(obs_dict["scene_image"], obs_dict["wrist_image"], state)


# per state space: (observation preprocessor, policy action -> target pose converter, action dim)
STATE_SPACES = {
    "eef": (eef_preprocessor, policy_action_to_abs_se3_converter, 10),
    "joints": (joints_preprocessor, abs_joint_policy_action_to_se3, 7),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="pretrained_model dir of the checkpoint")
    parser.add_argument("--state", default=STATE, choices=STATE_SPACES, help="state and action space of the policy")
    parser.add_argument("--eval-scenarios", default=EVAL_SCENARIOS_DATASET_PATH, help="dataset with initial scenes")
    parser.add_argument("--eval-dataset-name", default=EVAL_DATASET_NAME, help="name of the recorded eval dataset")
    args = parser.parse_args()

    preprocessor, policy_to_pose_converter, action_dim = STATE_SPACES[args.state]

    policy, policy_preprocessor, policy_postprocessor = make_lerobot_policy(args.checkpoint, "cuda")
    lerobot_agent = LerobotAgent(policy, "cuda", preprocessor, policy_preprocessor, policy_postprocessor)

    env = UR5eStation()
    env.reset()

    teleop_agent = make_gello_agent(env, GELLO_AGENT_PORT, GELLO_CONFIG)

    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((action_dim,), dtype=np.float32),
        root_dataset_dir=f"datasets/{args.eval_dataset_name}",
        dataset_name=args.eval_dataset_name,
        fps=10,
        use_videos=True,
    )

    eval_scenarios_dataset = LeRobotDataset(repo_id="", root=args.eval_scenarios)
    input("Press Enter to start evaluation (should hold your teleop in place now!)")
    try:
        eval(
            env,
            teleop_agent,
            lerobot_agent,
            dataset_recorder,
            policy_to_pose_converter=policy_to_pose_converter,
            teleop_to_pose_converter=convert_abs_gello_actions_to_se3,
            fps=10,
            eval_dataset=eval_scenarios_dataset,
            eval_dataset_image_key="scene_image",
            env_observation_image_key="scene_image",
        )
    finally:
        teleop_agent.close()
        env.close()
