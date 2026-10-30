import time

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from loguru import logger

from robot_imitation_glue.base import BaseAgent


def make_lerobot_policy(pretrained_path, device="cuda"):
    """load a policy checkpoint (a `pretrained_model` dir saved by lerobot-train) for inference.

    returns (policy, policy_preprocessor, policy_postprocessor). The policy does not normalize its inputs and
    outputs itself: the preprocessor (normalization, batching, device placement) and the postprocessor
    (unnormalization) are stored next to the weights, together with the dataset statistics, so the training
    dataset is not needed.
    """
    policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
    policy = get_policy_class(policy_config.type).from_pretrained(pretrained_path)
    policy.to(device)
    policy.eval()
    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, policy_preprocessor, policy_postprocessor


class LerobotAgent(BaseAgent):
    """
    agent for inference on a policy trained with Lerobot.

    """

    def __init__(self, policy, device, observation_preprocessor, policy_preprocessor, policy_postprocessor):
        """
        observation_preprocessor must take the env obs dict and do
        1) numpy to tensor
        2) batchifying the observation
        3) renaming the keys to the policy expected keys
        4) (optional) do any other preprocessing, such as image resizing/cropping...

        policy_preprocessor / policy_postprocessor are the processor pipelines saved with the checkpoint (see
        `make_lerobot_policy`), they (un)normalize around `policy.select_action`.
        """
        super().__init__()
        self.policy = policy
        self.device = device
        self.observation_preprocessor = observation_preprocessor
        self.policy_preprocessor = policy_preprocessor
        self.policy_postprocessor = policy_postprocessor

    def get_action(self, observation):
        start_time = time.time()
        observation = self.observation_preprocessor(observation)
        end_time = time.time()
        logger.info(f"Lerobot agent observation preprocessor took {((end_time - start_time)*1000):.2f} ms")
        observation = {k: v.to(self.device) for k, v in observation.items()}
        with torch.no_grad():
            time_start = time.time()
            action = self.policy.select_action(self.policy_preprocessor(observation))
            action = self.policy_postprocessor(action)
            time_end = time.time()
            logger.info(f"Lerobot agent inference took {((time_end - time_start)*1000):.2f} ms")
        return action.squeeze(0).cpu().numpy()

    def reset(self):
        self.policy.reset()
        self.policy_preprocessor.reset()
        self.policy_postprocessor.reset()


if __name__ == "__main__":
    import argparse

    # defaults of the command line arguments: edit them here or override them on the command line.
    CHECKPOINT_PATH = "outputs/train/pick-cube_diffusion/checkpoints/030000/pretrained_model"
    DATASET_PATH = "datasets/pick-cube-remapped"

    parser = argparse.ArgumentParser(description="Smoke-test a LeRobot checkpoint on random observations.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="pretrained_model dir of the checkpoint")
    parser.add_argument("--dataset", default=DATASET_PATH, help="dataset to take an example observation from")
    args = parser.parse_args()
    path, dataset_path = args.checkpoint, args.dataset

    policy, policy_preprocessor, policy_postprocessor = make_lerobot_policy(path, "cpu")

    dataset = LeRobotDataset(repo_id="dataset", root=dataset_path)

    # test policy

    batch = dataset[0]
    batch.pop("action")
    # unsqueeze all tensors
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)

    time_before = time.time()
    action = policy_postprocessor(policy.select_action(policy_preprocessor(batch))).squeeze(0)
    time_after = time.time()
    print(f"inferece took {time_after-time_before} s")
    print(f"policy action: {action}")
    print(f"dataset ['action']: {dataset[0]['action']}")

    print("testing inference on dummy observations")

    def observation_preprocessor(observation):
        observation["img1"] = observation["img1"].transpose(2, 0, 1)
        observation["img2"] = observation["img2"].transpose(2, 0, 1)
        observation["img1"] = observation["img1"].astype(np.float32) / 255.0
        observation["img2"] = observation["img2"].astype(np.float32) / 255.0

        observation["observation.images.scene_image"] = torch.from_numpy(observation["img1"]).float()
        observation["observation.images.wrist_image"] = torch.from_numpy(observation["img2"]).float()
        observation["observation.state"] = torch.from_numpy(observation["state"]).float()

        # drop old keys
        observation.pop("img1")
        observation.pop("img2")
        observation.pop("state")

        # resize images to 224x224
        from torchvision import transforms

        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(196)])
        observation["observation.images.scene_image"] = transform(observation["observation.images.scene_image"])
        observation["observation.images.wrist_image"] = transform(observation["observation.images.wrist_image"])

        # add batch dimension
        observation["observation.images.scene_image"] = observation["observation.images.scene_image"].unsqueeze(0)
        observation["observation.images.wrist_image"] = observation["observation.images.wrist_image"].unsqueeze(0)
        observation["observation.state"] = observation["observation.state"].unsqueeze(0)
        return observation

    policy, policy_preprocessor, policy_postprocessor = make_lerobot_policy(path, "cuda")
    agent = LerobotAgent(policy, "cuda", observation_preprocessor, policy_preprocessor, policy_postprocessor)
    for _ in range(20):
        test_obs = {
            "img1": np.random.randint(0, 255, (256, 256, 3)),
            "img2": np.random.randint(0, 255, (256, 256, 3)),
            "state": np.random.randn(7),
        }
        action = agent.get_action(test_obs)
    print(f"agent action: {action}")
