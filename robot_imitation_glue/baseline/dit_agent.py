"""Inference agent for the DiT flow-matching policy."""

from __future__ import annotations

import time

import numpy as np
import torch
from loguru import logger

from robot_imitation_glue.base import BaseAgent
from robot_imitation_glue.baseline.agent import preprocess_image
from robot_imitation_glue.baseline.dit_policy import DiTPolicy
from robot_imitation_glue.baseline.policy import load_checkpoint


class DiTAgent(BaseAgent):
    """Agent for inference with a policy trained by `robot_imitation_glue.baseline.train_dit`.

    The policy predicts a chunk of `prediction_horizon` actions at once. We execute only the first
    `action_chunk_size` of them and then replan from the new observation (receding horizon): the actions at the
    end of the horizon are the least accurate ones, and replanning keeps the policy reactive.

    `camera_key_mapping` maps the keys of the environment observation dict to the camera keys of the policy,
    `state_key` is the key of the proprioception vector in the environment observation dict.
    """

    def __init__(
        self,
        checkpoint_path,
        camera_key_mapping: dict[str, str],
        state_key: str,
        device: str = "cpu",
        action_chunk_size: int | None = None,
        n_sampling_steps: int = 10,
    ):
        super().__init__()
        self.device = device
        self.policy, self.proprio_normalizer, self.action_normalizer, self.image_size, self.crop_ratio = (
            load_checkpoint(checkpoint_path, device, policy_class=DiTPolicy)
        )
        self.camera_key_mapping = camera_key_mapping
        self.state_key = state_key
        self.n_sampling_steps = n_sampling_steps
        # execute half of the predicted horizon by default, as in the pusht-slim reference (16 predicted, 8 executed)
        self.action_chunk_size = action_chunk_size or self.policy.prediction_horizon // 2
        assert 0 < self.action_chunk_size <= self.policy.prediction_horizon

        missing = set(self.policy.camera_keys) - set(camera_key_mapping.values())
        assert not missing, f"no observation key mapped to the policy camera keys {missing}"

        self.reset()

    def reset(self) -> None:
        """Drop the cached action chunk. Must be called between episodes, otherwise the first actions of a new
        episode are the leftovers of the previous one."""
        self._action_chunk = None
        self._chunk_index = 0

    def _predict_chunk(self, observation: dict) -> np.ndarray:
        images = {
            policy_key: preprocess_image(observation[obs_key], self.image_size, self.crop_ratio).to(self.device)
            for obs_key, policy_key in self.camera_key_mapping.items()
        }
        proprio = torch.from_numpy(np.asarray(observation[self.state_key], dtype=np.float32)).unsqueeze(0)
        proprio = self.proprio_normalizer.normalize(proprio.to(self.device))

        start_time = time.time()
        actions = self.policy.sample(images, proprio, n_steps=self.n_sampling_steps)
        logger.info(f"DiT agent inference took {((time.time() - start_time)*1000):.2f} ms")

        actions = self.action_normalizer.unnormalize(actions)
        return actions.squeeze(0).cpu().numpy()

    def get_action(self, observation: dict) -> np.ndarray:
        if self._action_chunk is None or self._chunk_index >= self.action_chunk_size:
            self._action_chunk = self._predict_chunk(observation)
            self._chunk_index = 0
        action = self._action_chunk[self._chunk_index]
        self._chunk_index += 1
        return action
