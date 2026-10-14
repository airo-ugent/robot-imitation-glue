import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.gello.gello_agent import GelloAgent
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.eval_agent import eval
from robot_imitation_glue.preprocessing import concatenate_state, lerobot_observation_from_env_observation
from robot_imitation_glue.ur5station.ur5_robot_env import (
    GELLO_AGENT_PORT,
    UR5eStation,
    abs_joint_policy_action_to_se3,
    convert_abs_gello_actions_to_se3,
    dynamixel_config,
)

if __name__ == "__main__":
    checkpoint_path = "/home/tlips/Code/robot-imitation-glue/outputs/train/2025-04-16/15-26-26_pick-block-act/checkpoints/200000/pretrained_model"
    dataset_path = "/home/tlips/Code/robot-imitation-glue/datasets/pick-cube-v2-remapped-joints"
    eval_scenarios_dataset_path = "/home/tlips/Code/robot-imitation-glue/datasets/pick-cube-eval-scenarios"

    eval_dataset_name = "pick-cube-eval-ACT"

    def preprocessor(obs_dict):
        # joint state, matching the joint dataset transform in prepare_datasets.py
        state = concatenate_state(obs_dict["joints"], obs_dict["gripper_state"])
        return lerobot_observation_from_env_observation(obs_dict["scene_image"], obs_dict["wrist_image"], state)

    policy = make_lerobot_policy(checkpoint_path, dataset_path)
    lerobot_agent = LerobotAgent(policy, "cuda", preprocessor)

    env = UR5eStation()
    env.reset()

    teleop_agent = GelloAgent(dynamixel_config, GELLO_AGENT_PORT)

    # create a dataset recorder
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir=f"datasets/{eval_dataset_name}",
        dataset_name=eval_dataset_name,
        fps=10,
        use_videos=True,
    )

    eval_scenarios_dataset = LeRobotDataset(repo_id="", root=eval_scenarios_dataset_path)
    input("Press Enter to start evaluation (should hold your teleop in place now!)")
    eval(
        env,
        teleop_agent,
        lerobot_agent,
        dataset_recorder,
        policy_to_pose_converter=abs_joint_policy_action_to_se3,
        teleop_to_pose_converter=convert_abs_gello_actions_to_se3,
        fps=10,
        eval_dataset=eval_scenarios_dataset,
        eval_dataset_image_key="scene_image",
        env_observation_image_key="scene_image",
    )
