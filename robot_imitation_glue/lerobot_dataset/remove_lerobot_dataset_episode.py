"""Remove episodes from a local LeRobot dataset, in place; the remaining episodes are renumbered.

Thin CLI around `lerobot.datasets.dataset_tools.delete_episodes` (also available as `lerobot-edit-dataset`), which
writes the edited dataset to a new directory. This script writes it next to the dataset and swaps it in.

example:
    python -m robot_imitation_glue.lerobot_dataset.remove_lerobot_dataset_episode --root datasets/pick-cube -e 1-5,7 -b
"""

import argparse
import logging
import shutil
import time
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


def parse_episodes_list(episodes_str: str) -> list[int]:
    """Parse episode indices, ranges and comma-separated lists (e.g. '1-5,7,10-12') into a sorted list."""
    episodes = set()
    for ep in episodes_str.split(","):
        if "-" in ep:
            start, end = ep.split("-")
            episodes.update(range(int(start), int(end) + 1))
        else:
            episodes.add(int(ep))
    return sorted(episodes)


def backup_dataset(root: Path, backup: str | Path | bool) -> Path | None:
    """Copy the dataset dir to `backup` (a path) or, for True, to `<root>_backup_<timestamp>` next to it."""
    if not backup:
        return None
    backup_path = (
        Path(backup) if isinstance(backup, (str, Path)) else root.parent / f"{root.name}_backup_{int(time.time())}"
    )
    shutil.copytree(root, backup_path)
    logging.info(f"Backed up the dataset to {backup_path}")
    return backup_path


def parse_backup_arg(backup_value):
    """--backup without a value is True, 'true'/'false' are booleans, anything else is a path."""
    if isinstance(backup_value, str) and backup_value.lower() in ("true", "false"):
        return backup_value.lower() == "true"
    return backup_value


def remove_episodes(dataset: LeRobotDataset, episodes_to_remove: list[int], backup: str | Path | bool = False):
    """Remove the episodes from the dataset on disk and return the reloaded dataset."""
    root = Path(dataset.root)
    backup_dataset(root, backup)

    tmp_root = root.parent / f"{root.name}_tmp_{int(time.time())}"
    delete_episodes(dataset, episodes_to_remove, output_dir=tmp_root, repo_id=dataset.repo_id)
    shutil.rmtree(root)
    tmp_root.rename(root)
    return LeRobotDataset(repo_id=dataset.repo_id, root=root)


def main():
    parser = argparse.ArgumentParser(description="Remove episodes from a local LeRobot dataset")
    parser.add_argument("--repo-id", type=str, default="local", help="repo id of the dataset")
    parser.add_argument("--root", type=Path, required=True, help="Root directory of the local dataset")
    parser.add_argument(
        "-e",
        "--episodes",
        type=str,
        required=True,
        help="Episodes to remove. Can be a single index, comma-separated indices, or ranges (e.g., '1-5,7,10-12')",
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

    episodes_to_remove = parse_episodes_list(args.episodes)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    logging.info(f"Dataset has {dataset.meta.total_episodes} episodes, removing {episodes_to_remove}")

    updated_dataset = remove_episodes(dataset, episodes_to_remove, backup=parse_backup_arg(args.backup))
    logging.info(f"Dataset now has {updated_dataset.meta.total_episodes} episodes.")

    if args.push_to_hub:
        updated_dataset.push_to_hub(branch=args.branch, tags=args.tags, license=args.license, private=args.private)


if __name__ == "__main__":
    init_logging()
    main()
