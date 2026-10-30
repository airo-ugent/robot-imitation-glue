"""A minimal behavior-cloning policy: one resnet18 image encoder per camera + an MLP head.

Plain pytorch, no lerobot / openPI dependency. The policy predicts a single action vector and is trained with an
MSE loss on normalized actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

# resnet18 was trained with these statistics, so we always apply them (also when training from scratch, it does not
# hurt and it keeps the inference code identical for both cases).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# proprio dims with a std below this (m, rad, ...) in the training data are treated as constant: only centered, not
# scaled. Scaling them by their tiny std would turn sensor noise or a slightly different value at inference into
# huge inputs. Not applied to actions: they are only unnormalized at inference (multiplied by the std), where a
# small std is harmless, and real action dims can have stds in this range (e.g. delta positions in m).
PROPRIO_CONSTANT_STD = 1e-3


def to_chw_float_image(image: torch.Tensor) -> torch.Tensor:
    """converts a single HWC or CHW uint8/float image tensor to a CHW float tensor in [0,1]."""
    assert image.ndim == 3, f"expected an image with 3 dims, got shape {tuple(image.shape)}"
    if image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):  # HWC -> CHW
        image = image.permute(2, 0, 1)
    # uint8 images are in [0, 255], float images (e.g. from a LeRobot dataset) already in [0, 1]
    image = image.float() / 255.0 if image.dtype == torch.uint8 else image.float()
    if image.shape[0] == 1:  # grayscale -> RGB
        image = image.repeat(3, 1, 1)
    return image


def resize_image(image: torch.Tensor, image_size: int) -> torch.Tensor:
    """resize a (B,3,H,W) float image tensor. Used both during training and inference to make sure the
    preprocessing is identical."""
    if image.shape[-2:] == (image_size, image_size):
        return image
    return nn.functional.interpolate(image, size=(image_size, image_size), mode="bilinear", antialias=True)


def crop_image(image: torch.Tensor, crop_ratio: float, random_crop: bool) -> torch.Tensor:
    """crop a (..., H, W) image to `crop_ratio` of its height and width, before resizing. As in Diffusion Policy:
    a random crop during training (augmentation) and a center crop at inference, so both see the same scale."""
    if crop_ratio >= 1.0:
        return image
    height, width = image.shape[-2:]
    crop_height, crop_width = round(height * crop_ratio), round(width * crop_ratio)
    if random_crop:
        top = int(torch.randint(0, height - crop_height + 1, ()))
        left = int(torch.randint(0, width - crop_width + 1, ()))
    else:
        top, left = (height - crop_height) // 2, (width - crop_width) // 2
    return image[..., top : top + crop_height, left : left + crop_width]


class Normalizer(nn.Module):
    """(un)normalizes a vector with per-dimension mean/std. Registered as buffers so it is part of the state dict."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, constant_std: float = 0.0):
        """dims with a std below `constant_std` get a std of 1 (only centered), see PROPRIO_CONSTANT_STD."""
        super().__init__()
        self.register_buffer("mean", mean.float())
        std = std.float()
        std = torch.where(std < constant_std, torch.ones_like(std), std)
        # avoid dividing by zero for constant dimensions
        self.register_buffer("std", torch.clamp(std, min=1e-6))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class ImageEncoder(nn.Module):
    """resnet18 backbone that maps a (B,3,H,W) image in [0,1] to a (B, feature_dim) feature vector."""

    def __init__(self, feature_dim: int = 128, pretrained_backbone: bool = False):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained_backbone else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, feature_dim)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        return self.backbone(image)


class BCPolicy(nn.Module):
    """Behavior cloning policy: concatenate the image features of all cameras with the proprioception vector and
    regress the action with an MLP."""

    def __init__(
        self,
        camera_keys: list[str],
        proprio_dim: int,
        action_dim: int,
        image_feature_dim: int = 128,
        hidden_dim: int = 512,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        assert len(camera_keys) > 0, "the policy needs at least one camera"
        # keep the config around so that a checkpoint can be loaded without knowing the hyperparameters
        self.config = {
            "camera_keys": list(camera_keys),
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
            "image_feature_dim": image_feature_dim,
            "hidden_dim": hidden_dim,
        }
        self.camera_keys = list(camera_keys)
        self.action_dim = action_dim

        # a ModuleList instead of a ModuleDict, because the lerobot camera keys contain dots
        self.encoders = nn.ModuleList([ImageEncoder(image_feature_dim, pretrained_backbone) for _ in self.camera_keys])
        mlp_input_dim = len(self.camera_keys) * image_feature_dim + proprio_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, images: dict[str, torch.Tensor], proprio: torch.Tensor) -> torch.Tensor:
        """images: dict of (B,3,H,W) float tensors in [0,1], proprio: (B, proprio_dim) normalized float tensor.

        returns the normalized action as a (B, action_dim) tensor.
        """
        features = [encoder(images[key]) for key, encoder in zip(self.camera_keys, self.encoders)]
        features.append(proprio)
        return self.mlp(torch.cat(features, dim=1))

    def compute_loss(
        self, images: dict[str, torch.Tensor], proprio: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss on the normalized actions."""
        return nn.functional.mse_loss(self.forward(images, proprio), action)


def save_checkpoint(
    path,
    policy: nn.Module,
    proprio_normalizer: Normalizer,
    action_normalizer: Normalizer,
    image_size,
    crop_ratio: float = 1.0,
):
    """save the weights, the model config, the normalization statistics and the image preprocessing in one file."""
    torch.save(
        {
            "config": policy.config,
            "image_size": image_size,
            "crop_ratio": crop_ratio,
            "model_state_dict": policy.state_dict(),
            "proprio_normalizer_state_dict": proprio_normalizer.state_dict(),
            "action_normalizer_state_dict": action_normalizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path, device="cpu", policy_class=BCPolicy):
    """returns (policy, proprio_normalizer, action_normalizer, image_size, crop_ratio), on the requested device.

    `policy_class` is constructed from the config stored in the checkpoint, so other policies with the same
    checkpoint format (e.g. `dit_policy.DiTPolicy`) can reuse this loader."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    policy = policy_class(**config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()

    proprio_dim = config["proprio_dim"]
    action_dim = config["action_dim"]
    proprio_normalizer = Normalizer(torch.zeros(proprio_dim), torch.ones(proprio_dim))
    proprio_normalizer.load_state_dict(checkpoint["proprio_normalizer_state_dict"])
    action_normalizer = Normalizer(torch.zeros(action_dim), torch.ones(action_dim))
    action_normalizer.load_state_dict(checkpoint["action_normalizer_state_dict"])

    # checkpoints from before the crop augmentation were trained on uncropped images
    crop_ratio = checkpoint.get("crop_ratio", 1.0)
    return policy, proprio_normalizer.to(device), action_normalizer.to(device), checkpoint["image_size"], crop_ratio
