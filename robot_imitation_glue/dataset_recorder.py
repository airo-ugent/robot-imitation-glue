import atexit
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robot_imitation_glue.base import BaseDatasetRecorder


class LeRobotDatasetRecorder(BaseDatasetRecorder):
    DEFAULT_FEATURES = {
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "seed": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "timestamp": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
    }

    def __init__(
        self,
        example_obs_dict: dict,
        example_action: np.array,
        root_dataset_dir: Path,
        dataset_name: str,
        fps: int,
        use_videos=True,
    ):

        self.root_dataset_dir = Path(root_dataset_dir)
        self.dataset_name = dataset_name
        self.fps = fps

        self._n_recorded_episodes = 0
        self.key_mapping_dict = {}

        self.image_keys = []
        self.state_keys = []

        # create features  using the example dict. assume all numpy arrays. 0D is a scalar, 1D with size 1 is a scalar, 1D with size > 1 is a vector, 3D is an image.
        features = self.DEFAULT_FEATURES.copy()
        for key, value in example_obs_dict.items():
            shape = value.shape
            if len(shape) == 0:
                features[key] = {"dtype": "float32", "shape": (1,), "names": None}
            elif len(shape) == 1:
                features[key] = {"dtype": "float32", "shape": shape, "names": None}
            elif len(shape) == 3:
                if not shape[0] == 3:
                    # not hannel first! reorder shape
                    shape = (shape[2], shape[0], shape[1])
                if use_videos:
                    features[key] = {"dtype": "video", "names": ["channel", "height", "width"], "shape": shape}
                else:
                    features[key] = {"dtype": "image", "shape": shape, "names": None}
            else:
                raise ValueError(f"Unsupported shape for feature {key}: {shape}")

            if len(shape) == 3:
                self.image_keys.append(key)
            else:
                self.state_keys.append(key)

        # add action to features
        features["action"] = {"dtype": "float32", "shape": example_action.shape, "names": None}
        print(f"Features: {features}")

        if self.root_dataset_dir.exists():
            print(f"Dataset {dataset_name} already exists. Loading it.")
            self.lerobot_dataset = self._resume_existing()
            self._n_recorded_episodes = self.lerobot_dataset.meta.total_episodes
            print(f"Loaded {self._n_recorded_episodes} episodes.")
        else:
            print(f"Dataset {dataset_name} does not exist. Creating it.")
            self.lerobot_dataset = LeRobotDataset.create(
                repo_id=dataset_name,
                fps=self.fps,
                root=self.root_dataset_dir,
                features=features,
                use_videos=use_videos,
                image_writer_processes=0,
                image_writer_threads=8,
            )
        print("Dataset created:", self.lerobot_dataset)
        # lerobot keeps the data and episode-metadata parquet files open until finalize(): without it the dataset
        # on disk is unreadable. save_episode() finalizes after every episode, this covers an unsaved last one.
        atexit.register(self.finish_recording)

    def _resume_existing(self) -> LeRobotDataset:
        """resume the dataset in root_dataset_dir after checking that it is consistent."""
        dataset = self._resume()
        meta = dataset.meta
        # a dataset without saved episodes (a session that quit before saving one) has no episode metadata yet
        if meta.total_episodes > 0 and (meta.episodes is None or len(meta.episodes) != meta.total_episodes):
            # a session that was killed while saving an episode, appending would hide the damage
            raise RuntimeError(
                f"{self.root_dataset_dir} is inconsistent: info.json counts {meta.total_episodes} episodes, the "
                f"episode metadata has {0 if meta.episodes is None else len(meta.episodes)}. Repair or move it."
            )
        if (self.root_dataset_dir / "action_representation.json").exists():
            print(
                "WARNING: this dataset has relative action stats (action_representation.json). Run "
                "lerobot_dataset/relative_action_stats.py again after recording, the new episodes add absolute "
                "action stats."
            )
        return dataset

    def _resume(self) -> LeRobotDataset:
        return LeRobotDataset.resume(
            repo_id=self.dataset_name, root=self.root_dataset_dir, image_writer_processes=0, image_writer_threads=16
        )

    def start_episode(self):
        pass

    def record_step(self, obs, action):
        frame = {
            "action": torch.from_numpy(action),
            "next.reward": torch.tensor([0.0]),
            "next.success": torch.tensor([False]),
            "seed": torch.tensor([0]),  # TODO: store the seed
            "task": "",
        }
        for key in self.image_keys:
            lerobot_key = self.key_mapping_dict.get(key, key)
            frame[lerobot_key] = obs[key]

        for key in self.state_keys:
            frame[key] = torch.tensor(obs[key])
        self.lerobot_dataset.add_frame(frame)

    def set_episode_success(self, success):
        """Episode-level label: sets next.success of *all* frames in the current episode buffer (see
        BaseDatasetRecorder.set_episode_success)."""
        # lerobot keeps the unsaved episode in the writer's `episode_buffer`
        episode_buffer = self.lerobot_dataset.writer.episode_buffer
        if not self.lerobot_dataset.has_pending_frames():
            return
        # add_frame() converts the torch tensors it is handed to numpy, so write numpy here too:
        # every buffer entry then has the same type whether it was set live or patched in hindsight.
        episode_buffer["next.success"] = [np.array([bool(success)])] * episode_buffer["size"]

    def delete_episode(self):
        """Discard the current episode buffer (and its temporary images); saved episodes are untouched."""
        # clear_episode_buffer() waits for the async image writer before it removes this episode's image dirs
        self.lerobot_dataset.clear_episode_buffer()

    def save_episode(self):
        self.lerobot_dataset.save_episode()
        self._n_recorded_episodes += 1
        # finalize and reopen after every episode so each saved episode is complete on disk (one data file per
        # episode): otherwise a session that is killed (SIGKILL, closed terminal, driver crash) before
        # finish_recording() leaves the whole dataset unreadable, including the episodes of earlier sessions.
        self.lerobot_dataset.finalize()
        self.lerobot_dataset = self._resume()

    def finish_recording(self):
        """Flush and close the parquet writers; no episodes can be recorded afterwards."""
        self.lerobot_dataset.finalize()
        atexit.unregister(self.finish_recording)

    @property
    def n_recorded_episodes(self):
        return self._n_recorded_episodes


if __name__ == "__main__":

    example_obs = {
        "robot_pose": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "image": np.random.rand(800, 400, 3).astype(np.float32),
        "image1": np.random.rand(3, 800, 400).astype(np.float32),
        "image2": np.random.rand(3, 800, 400).astype(np.float32),
        "image3": np.random.rand(3, 800, 400).astype(np.float32),
        "gripper_state": np.array([0.1], dtype=np.float32),
    }

    example_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)

    import os

    # remove entire dataset directory
    os.system("rm -rf datasets")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=example_obs,
        example_action=example_action,
        root_dataset_dir=Path("datasets"),
        dataset_name="test_dataset",
        fps=30,
        use_videos=True,
    )

    for j in range(3):
        dataset_recorder.start_episode()
        for i in range(10 - j):
            dataset_recorder.record_step(example_obs, example_action)
        dataset_recorder.save_episode()

    dataset_recorder.finish_recording()
    print(f"Recorded {dataset_recorder.n_recorded_episodes} episodes.")

    dataset = LeRobotDataset(repo_id="test_dataset", root=Path("datasets"), episodes=[0, 1])
    print(f"Loaded {len(dataset)} steps.")
    print(dataset.meta.episodes["dataset_from_index"], dataset.meta.episodes["dataset_to_index"])
