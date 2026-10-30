"""Inference agent for the behavior-cloning baseline policy."""

from __future__ import annotations

import time

import numpy as np
import torch
from loguru import logger

from robot_imitation_glue.base import BaseAgent
from robot_imitation_glue.baseline.policy import crop_image, load_checkpoint, resize_image, to_chw_float_image


def preprocess_image(image: np.ndarray, image_size: int, crop_ratio: float = 1.0) -> torch.Tensor:
    """converts a single HWC (or CHW) uint8/float image to a (1,3,image_size,image_size) float tensor in [0,1].

    Center crops to `crop_ratio` first: the inference counterpart of the random crop of training."""
    tensor = to_chw_float_image(torch.from_numpy(np.ascontiguousarray(image)))
    tensor = crop_image(tensor, crop_ratio, random_crop=False)
    return resize_image(tensor.unsqueeze(0), image_size)


class BaselineAgent(BaseAgent):
    """Agent for inference with a policy trained by `robot_imitation_glue.baseline.train`.

    `camera_key_mapping` maps the keys of the environment observation dict to the camera keys of the policy,
    `state_key` is the key of the proprioception vector in the environment observation dict.
    """

    def __init__(self, checkpoint_path, camera_key_mapping: dict[str, str], state_key: str, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.policy, self.proprio_normalizer, self.action_normalizer, self.image_size, self.crop_ratio = (
            load_checkpoint(checkpoint_path, device)
        )
        self.camera_key_mapping = camera_key_mapping
        self.state_key = state_key

        missing = set(self.policy.camera_keys) - set(camera_key_mapping.values())
        assert not missing, f"no observation key mapped to the policy camera keys {missing}"

    def get_action(self, observation: dict) -> np.ndarray:
        images = {
            policy_key: preprocess_image(observation[obs_key], self.image_size, self.crop_ratio).to(self.device)
            for obs_key, policy_key in self.camera_key_mapping.items()
        }
        proprio = torch.from_numpy(np.asarray(observation[self.state_key], dtype=np.float32)).unsqueeze(0)
        proprio = self.proprio_normalizer.normalize(proprio.to(self.device))

        start_time = time.time()
        with torch.no_grad():
            action = self.policy(images, proprio)
        logger.info(f"baseline agent inference took {((time.time() - start_time)*1000):.2f} ms")

        action = self.action_normalizer.unnormalize(action)
        return action.squeeze(0).cpu().numpy()
