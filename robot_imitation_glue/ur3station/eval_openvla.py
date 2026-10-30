"""Evaluate OpenVLA (served over HTTP, relative EEF actions) on the UR3 station."""

import argparse
import os
import shutil

import numpy as np

from robot_imitation_glue.preprocessing import (
    delta_action_to_abs_se3_converter,
    relative_policy_action_to_abs_se3_converter,
)

# defaults of the command line arguments: edit them here or override them on the command line.
EVAL_SCENARIOS_DATASET_PATH = "datasets"
EVAL_DATASET_NAME = "test_dataset"

if __name__ == "__main__":
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from robot_imitation_glue.agents.airo_teleop_agent import make_spacemouse_agent
    from robot_imitation_glue.agents.openvla_agent import OpenVLAAgent
    from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
    from robot_imitation_glue.eval_agent import eval
    from robot_imitation_glue.ur3station.ur3_robot_env import UR3eStation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-scenarios", default=EVAL_SCENARIOS_DATASET_PATH, help="dataset with initial scenes")
    parser.add_argument("--eval-dataset-name", default=EVAL_DATASET_NAME, help="name of the recorded eval dataset")
    args = parser.parse_args()

    env = UR3eStation()
    env.reset()
    teleop_agent = make_spacemouse_agent(env)
    policy_agent = OpenVLAAgent()

    if os.path.exists(args.eval_scenarios):
        dataset = LeRobotDataset(repo_id="eval_scenarios", root=args.eval_scenarios)
    else:
        dataset = None

    # the eval dataset is recorded from scratch on every run
    eval_dataset_dir = f"datasets/{args.eval_dataset_name}"
    if os.path.exists(eval_dataset_dir):
        shutil.rmtree(eval_dataset_dir)
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
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
            relative_policy_action_to_abs_se3_converter,
            delta_action_to_abs_se3_converter,
            fps=10,
            eval_dataset=dataset,
            eval_dataset_image_key="scene_image",
        )
    finally:
        teleop_agent.close()
        env.close()
