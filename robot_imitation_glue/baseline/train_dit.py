"""Train the DiT flow-matching policy on a recorded dataset.

Same plumbing as `robot_imitation_glue.baseline.train` (lerobot only reads the frames from disk, the model and
the loop are plain pytorch), but the policy predicts a chunk of future actions instead of a single action, so
the dataset yields action chunks and the loop uses the AdamW + warmup/cosine + gradient clipping recipe that
diffusion transformers need.

example:
    python -m robot_imitation_glue.baseline.train_dit \
      --dataset-root datasets/pick-cube \
      --image-keys observation.images.scene_image observation.images.wrist_image \
      --output-dir outputs/baseline-dit
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from robot_imitation_glue.baseline.dit_policy import DiTPolicy
from robot_imitation_glue.baseline.policy import Normalizer, save_checkpoint
from robot_imitation_glue.baseline.train import BCDataset, LeRobotDataset, set_seed


class DiTDataset(BCDataset):
    """`BCDataset` that returns a chunk of `prediction_horizon` future actions plus a validity mask.

    The future actions are queried through lerobot's `delta_timestamps` mechanism (see `make_dataset`): lerobot
    clamps the queried indices to the boundaries of the episode and flags the clamped entries as padding, so a
    chunk never mixes actions of two episodes and the mask marks exactly the in-episode timesteps.
    """

    def __init__(self, lerobot_dataset, image_keys, state_key, action_key, image_size, prediction_horizon: int):
        super().__init__(lerobot_dataset, image_keys, state_key, action_key, image_size)
        self.prediction_horizon = prediction_horizon

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        images = self.preprocess_images(sample)
        proprio = sample[self.state_key].float()
        actions = sample[self.action_key].float()
        assert actions.shape == (self.prediction_horizon, actions.shape[-1]), (
            f"expected an action chunk of {self.prediction_horizon} steps, got {tuple(actions.shape)}. "
            "Was the LeRobotDataset created with delta_timestamps?"
        )
        is_pad = sample[f"{self.action_key}_is_pad"]
        return images, proprio, actions, (~is_pad).float()


def make_dataset(args) -> DiTDataset:
    """The delta timestamps depend on the fps of the dataset, which is only known once the metadata is read, so
    the dataset is opened twice (the first time only to read the fps, which is cheap)."""
    kwargs = dict(
        repo_id="local", root=str(args.dataset_root), download_videos=False, video_backend=args.video_backend
    )
    fps = LeRobotDataset(**kwargs).fps
    delta_timestamps = {args.action_key: [i / fps for i in range(args.prediction_horizon)]}
    lerobot_dataset = LeRobotDataset(**kwargs, delta_timestamps=delta_timestamps)
    return DiTDataset(
        lerobot_dataset, args.image_keys, args.state_key, args.action_key, args.image_size, args.prediction_horizon
    )


def compute_normalization_stats(dataloader: DataLoader) -> tuple[Normalizer, Normalizer]:
    """One pass over the dataset to compute the mean/std of the proprioception and the action vectors.

    Padded (out-of-episode) actions are dropped, they would bias the statistics towards the last action of every
    episode."""
    proprios, actions = [], []
    for _, proprio, action, mask in tqdm(dataloader, desc="Computing normalization stats", leave=False):
        proprios.append(proprio)
        actions.append(action[mask.bool()])
    proprios = torch.cat(proprios)
    actions = torch.cat(actions)
    return Normalizer(proprios.mean(dim=0), proprios.std(dim=0)), Normalizer(actions.mean(dim=0), actions.std(dim=0))


def make_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
    """Linear warmup followed by a cosine decay to `min_lr_ratio` of the base learning rate."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(args)
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

    policy = DiTPolicy(
        camera_keys=args.image_keys,
        proprio_dim=proprio_normalizer.mean.shape[0],
        action_dim=action_normalizer.mean.shape[0],
        prediction_horizon=args.prediction_horizon,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        pretrained_backbone=args.use_imagenet_pretrained,
    ).to(device)
    print(f"DiT policy with {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M parameters.")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = make_lr_scheduler(optimizer, args.warmup_steps, args.epochs * len(dataloader))

    for epoch in range(1, args.epochs + 1):
        policy.train()
        loss_sum, n_batches = 0.0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, proprio, action, mask in pbar:
            images = {key: value.to(device, non_blocking=True) for key, value in images.items()}
            proprio = proprio_normalizer.normalize(proprio.to(device, non_blocking=True))
            action = action_normalizer.normalize(action.to(device, non_blocking=True))
            mask = mask.to(device, non_blocking=True)

            loss = policy.compute_loss(images, proprio, action, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            loss_sum += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(
            f"Epoch {epoch:03d} | train_loss={loss_sum / max(n_batches, 1):.6f} | lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = output_dir / "policy.ckpt"
            save_checkpoint(checkpoint_path, policy, proprio_normalizer, action_normalizer, args.image_size)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DiT flow-matching policy on a LeRobot format dataset")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the dataset root on disk")
    parser.add_argument(
        "--image-keys", type=str, nargs="+", required=True, help="Image feature keys to use as camera inputs"
    )
    parser.add_argument("--state-key", type=str, default="observation.state", help="Proprioception feature key")
    parser.add_argument("--action-key", type=str, default="action", help="Action feature key")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline-dit", help="Directory to save checkpoints")

    parser.add_argument("--image-size", type=int, default=128, help="Images are resized to this square resolution")
    parser.add_argument("--prediction-horizon", type=int, default=16, help="Number of future actions to predict")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=10.0, help="Max gradient norm")
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
