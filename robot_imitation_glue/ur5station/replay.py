import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robot_imitation_glue.lerobot_dataset.replay_episode import replay_episode
from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation, abs_joint_policy_action_to_se3

# defaults of the command line arguments: edit them here or override them on the command line.
DATASET_PATH = "datasets/pick_cube-v2-remapped-joints"
EPISODE = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a recorded episode (joint actions) on the UR5 station.")
    parser.add_argument("--dataset", default=DATASET_PATH, help="dataset to replay from")
    parser.add_argument("--episode", type=int, default=EPISODE, help="index of the episode to replay")
    args = parser.parse_args()

    try:
        env = UR5eStation()
        dataset = LeRobotDataset(repo_id="", root=args.dataset)
        replay_episode(
            env, dataset, abs_joint_policy_action_to_se3, "scene_image", "observation.images.scene_image", args.episode
        )

    finally:
        env.close()
