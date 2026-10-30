"""Set the language instruction (task) of episodes of a local LeRobot dataset, in place.

Thin CLI around `lerobot.datasets.dataset_tools.modify_tasks` (also available as `lerobot-edit-dataset`).

example:
    python -m robot_imitation_glue.lerobot_dataset.add_description_lerobot_dataset \
        --root datasets/pick-cube -e 0-20 --description "pick the cube" -b
"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.dataset_tools import modify_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

from robot_imitation_glue.lerobot_dataset.remove_lerobot_dataset_episode import (
    backup_dataset,
    parse_backup_arg,
    parse_episodes_list,
)


def add_description(
    dataset: LeRobotDataset,
    episodes_to_describe: list[int],
    backup: str | Path | bool = False,
    task_description: str = "",
) -> LeRobotDataset:
    """Set the task of the episodes to `task_description` and return the updated dataset."""
    if not episodes_to_describe:
        return dataset
    backup_dataset(Path(dataset.root), backup)
    # modify_tasks refuses episodes without a task unless they are listed, and the recorder stores the task "" (no
    # task), so list every episode: the new description for the selected ones, the current task for the others.
    episode_tasks = {
        ep_idx: (dataset.meta.episodes[ep_idx]["tasks"] or [""])[0] for ep_idx in range(dataset.num_episodes)
    }
    episode_tasks.update({ep_idx: task_description for ep_idx in episodes_to_describe})
    return modify_tasks(dataset, episode_tasks=episode_tasks)


def main():
    parser = argparse.ArgumentParser(description="Set the task description of episodes of a local LeRobot dataset")
    parser.add_argument("--repo-id", type=str, default="local", help="repo id of the dataset")
    parser.add_argument("--root", type=Path, required=True, help="Root directory of the local dataset")
    parser.add_argument(
        "-e",
        "--episodes",
        type=str,
        required=True,
        help="Episodes to describe. Can be a single index, comma-separated indices, or ranges (e.g., '1-5,7,10-12')",
    )
    parser.add_argument(
        "--description", type=str, required=True, help="Task description to use as language instruction"
    )
    parser.add_argument(
        "-b",
        "--backup",
        nargs="?",
        const=True,
        default=False,
        help="Create a backup before modifying the dataset. Without a value, creates a backup in the default location. "
        "With a value, either 'true'/'false' or a path to store the backup.",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the edited dataset to the Hugging Face hub")
    parser.add_argument("--private", action="store_true", help="Make the repository on the Hub private")
    parser.add_argument("--tags", type=str, nargs="+", help="List of tags to apply to the dataset on the Hub")
    parser.add_argument("--license", type=str, default=None, help="License to use for the dataset on the Hub")
    parser.add_argument("--branch", type=str, default=None, help="Branch to push the dataset to on the Hub")
    args = parser.parse_args()

    episodes_to_describe = parse_episodes_list(args.episodes)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    logging.info(f"Setting the task of episodes {episodes_to_describe} to '{args.description}'")

    updated_dataset = add_description(
        dataset, episodes_to_describe, backup=parse_backup_arg(args.backup), task_description=args.description
    )
    logging.info(f"Dataset now has the following tasks: {updated_dataset.meta.tasks}.")

    if args.push_to_hub:
        updated_dataset.push_to_hub(branch=args.branch, tags=args.tags, license=args.license, private=args.private)


if __name__ == "__main__":
    init_logging()
    main()
