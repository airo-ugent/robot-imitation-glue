"""Evaluate pi0 (served by openpi, joint-space state and actions) on the UR5 station."""

import argparse

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robot_imitation_glue.agents.airo_teleop_agent import make_gello_agent
from robot_imitation_glue.agents.openpi_agent import OpenPIAgent
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.eval_agent import eval
from robot_imitation_glue.preprocessing import concatenate_state
from robot_imitation_glue.ur5station.ur5_robot_env import (
    GELLO_AGENT_PORT,
    GELLO_CONFIG,
    UR5eStation,
    abs_joint_policy_action_to_se3,
    convert_abs_gello_actions_to_se3,
)

# defaults of the command line arguments: edit them here or override them on the command line.
TRAIN_DATASET_PATH = "datasets/pour-cup-v2"
EVAL_DATASET_NAME = "pour-cup-eval-pi0"
PROMPT = "pour the white mug in the transparent cup"
OPENPI_HOST = "localhost"
OPENPI_PORT = 8000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", default=TRAIN_DATASET_PATH, help="dataset whose scenes are replayed")
    parser.add_argument("--eval-dataset-name", default=EVAL_DATASET_NAME, help="name of the recorded eval dataset")
    parser.add_argument("--prompt", default=PROMPT, help="language instruction for the policy")
    parser.add_argument("--host", default=OPENPI_HOST, help="host of the openpi policy server")
    parser.add_argument("--port", type=int, default=OPENPI_PORT, help="port of the openpi policy server")
    args = parser.parse_args()

    def preprocessor(obs_dict):
        # joint state, matching the joint dataset transform in prepare_datasets.py
        return {
            "scene_image": obs_dict["scene_image"],
            "wrist_image": obs_dict["wrist_image"],
            "state": concatenate_state(obs_dict["joints"], obs_dict["gripper_state"]),
        }

    env = UR5eStation()
    env.reset()

    teleop_agent = make_gello_agent(env, GELLO_AGENT_PORT, GELLO_CONFIG)

    pi0_agent = OpenPIAgent(
        default_prompt=args.prompt,
        observation_preprocessor=preprocessor,
        host=args.host,
        port=args.port,
        n_action_steps=10,
    )

    # create a dataset recorder
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir=f"datasets/{args.eval_dataset_name}",
        dataset_name=args.eval_dataset_name,
        fps=10,
        use_videos=True,
    )

    train_dataset = LeRobotDataset(repo_id="", root=args.train_dataset)
    input("Press Enter to start evaluation (should hold your teleop in place now!)")
    try:
        eval(
            env,
            teleop_agent,
            pi0_agent,
            dataset_recorder,
            policy_to_pose_converter=abs_joint_policy_action_to_se3,
            teleop_to_pose_converter=convert_abs_gello_actions_to_se3,
            fps=10,
            eval_dataset=train_dataset,
            eval_dataset_image_key="scene_image",
            env_observation_image_key="scene_image",
        )
    finally:
        teleop_agent.close()
        env.close()
