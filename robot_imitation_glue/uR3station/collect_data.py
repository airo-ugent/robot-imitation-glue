import os
from pathlib import Path

import numpy as np  # Missing import for np

from robot_imitation_glue.agents.spacemouse_agent import SpaceMouseAgent
from robot_imitation_glue.collect_data import collect_data
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder

# the delta action encoding is shared with the dataset preparation and the eval scripts,
# see robot_imitation_glue.preprocessing.
from robot_imitation_glue.preprocessing import (
    abs_se3_to_relative_policy_action_converter,
    delta_action_to_abs_se3_converter,
)
from robot_imitation_glue.uR3station.robot_env import UR3eStation

dataset_name = "test_dataset"


def main():
    # connects to the robot, the cameras and the SpaceMouse: only run this at the station.
    env = UR3eStation()
    agent = SpaceMouseAgent()

    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir=Path(f"datasets/{dataset_name}"),
        dataset_name=dataset_name,
        fps=10,
        use_videos=True,
    )

    collect_data(
        env,
        agent,
        dataset_recorder,
        10,
        delta_action_to_abs_se3_converter,
        abs_se3_to_relative_policy_action_converter,
    )

    env.close()
    agent.close()


if __name__ == "__main__":
    main()
