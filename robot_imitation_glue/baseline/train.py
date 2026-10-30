"""Train the behavior-cloning baseline policy on a recorded dataset.

The datasets are recorded in the lerobot format (see `robot_imitation_glue.dataset_recorder`), so the lerobot
package is used to read the frames from disk. The model and the training loop below are plain pytorch.

example:
    python -m robot_imitation_glue.baseline.train \
      --dataset-root datasets/pick-cube \
      --image-keys observation.images.scene_image observation.images.wrist_image \
      --output-dir outputs/baseline-bc
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from robot_imitation_glue.baseline.policy import (
    BCPolicy,
    Normalizer,
    resize_image,
    save_checkpoint,
    to_chw_float_image,
)


class BCDataset(Dataset):
    """Wrap a LeRobotDataset to expose (images, proprioception, action) samples for the baseline policy."""

    def __init__(self, lerobot_dataset, image_keys: list[str], state_key: str, action_key: str, image_size: int):
        self.dataset = lerobot_dataset
        self.image_keys = image_keys
        self.state_key = state_key
        self.action_key = action_key
        self.image_size = image_size

        for key in [*image_keys, state_key, action_key]:
            if key not in self.dataset.features:
                available = ", ".join(sorted(self.dataset.features.keys()))
                raise ValueError(f"key '{key}' not found in the dataset. Available keys: {available}")

    def __len__(self) -> int:
        return len(self.dataset)

    def preprocess_images(self, sample: dict) -> dict[str, torch.Tensor]:
        """the camera images of a lerobot sample as (3, image_size, image_size) float tensors in [0,1]."""
        return {
            key: resize_image(to_chw_float_image(sample[key]).unsqueeze(0), self.image_size).squeeze(0)
            for key in self.image_keys
        }

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        images = self.preprocess_images(sample)
        proprio = sample[self.state_key].float()
        action = sample[self.action_key].float()
        # lerobot can return an action chunk, this baseline only predicts a single action
        if action.ndim == 2:
            action = action[0]
        return images, proprio, action


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_normalization_stats(dataloader: DataLoader) -> tuple[Normalizer, Normalizer]:
    """One pass over the dataset to compute the mean/std of the proprioception and the action vectors."""
    proprios, actions = [], []
    for _, proprio, action in tqdm(dataloader, desc="Computing normalization stats", leave=False):
        proprios.append(proprio)
        actions.append(action)
    proprios = torch.cat(proprios)
    actions = torch.cat(actions)
    return Normalizer(proprios.mean(dim=0), proprios.std(dim=0)), Normalizer(actions.mean(dim=0), actions.std(dim=0))


def train(args) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lerobot_dataset = LeRobotDataset(
        repo_id="local", root=str(args.dataset_root), download_videos=False, video_backend=args.video_backend
    )
    dataset = BCDataset(lerobot_dataset, args.image_keys, args.state_key, args.action_key, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Training on {len(dataset)} frames.")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # a separate pass without drop_last, so every frame contributes to the statistics
    stats_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    proprio_normalizer, action_normalizer = compute_normalization_stats(stats_dataloader)
    proprio_normalizer, action_normalizer = proprio_normalizer.to(device), action_normalizer.to(device)

    policy = BCPolicy(
        camera_keys=args.image_keys,
        proprio_dim=proprio_normalizer.mean.shape[0],
        action_dim=action_normalizer.mean.shape[0],
        pretrained_backbone=args.use_imagenet_pretrained,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        policy.train()
        loss_sum, n_batches = 0.0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, proprio, action in pbar:
            images = {key: value.to(device, non_blocking=True) for key, value in images.items()}
            proprio = proprio_normalizer.normalize(proprio.to(device, non_blocking=True))
            action = action_normalizer.normalize(action.to(device, non_blocking=True))

            loss = policy.compute_loss(images, proprio, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch:03d} | train_loss={loss_sum / max(n_batches, 1):.6f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = output_dir / "policy.ckpt"
            save_checkpoint(checkpoint_path, policy, proprio_normalizer, action_normalizer, args.image_size)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the behavior cloning baseline on a LeRobot format dataset")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the dataset root on disk")
    parser.add_argument(
        "--image-keys", type=str, nargs="+", required=True, help="Image feature keys to use as camera inputs"
    )
    parser.add_argument("--state-key", type=str, default="observation.state", help="Proprioception feature key")
    parser.add_argument("--action-key", type=str, default="action", help="Action feature key")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline-bc", help="Directory to save checkpoints")

    parser.add_argument("--image-size", type=int, default=128, help="Images are resized to this square resolution")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=10, help="Save a checkpoint every n epochs")
    parser.add_argument("--video-backend", type=str, default="pyav", choices=["pyav", "torchcodec", "video_reader"])
    parser.add_argument(
        "--use-imagenet-pretrained", action="store_true", help="Initialize the resnet18 encoders from ImageNet weights"
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu, defaults to cuda if available")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
