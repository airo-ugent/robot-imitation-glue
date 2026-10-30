"""Write the action statistics of UMI-style relative actions into a dataset, for `use_relative_actions` policies.

A diffusion policy with `use_relative_actions` (lerobot's se3_relative_action_processor.py) trains on every action
chunk re-expressed in the frame of the latest observed pose, but the dataset stores absolute [xyz, rot6d, ...]
targets, so `meta/stats.json` describes the wrong quantity for its normalizer. This replaces the `action` entry of
`meta/stats.json` with the statistics of exactly the relative chunks training samples, and records that in
`action_representation.json`. The per-episode action stats in `meta/episodes/` are replaced as well: lerobot's
dataset tools (e.g. remove_lerobot_dataset_episode.py) rebuild `meta/stats.json` from them, so an edited dataset
keeps relative stats. Recording more episodes into the dataset adds absolute per-episode stats: run this again
afterwards. The data itself is not changed, so running it again gives the same result.

The chunks follow the sampling of lerobot-train: every frame t except the last `drop_n_last_frames` of an episode
starts a sample, its actions are t-(n_obs_steps-1) .. t-(n_obs_steps-1)+horizon-1 (copy-padded with the first/last
frame of the episode) and its reference is the state at t; episodes of at most `drop_n_last_frames` frames are
skipped, as lerobot's sampler does. Use the values of the policy config.

example:
    python -m robot_imitation_glue.lerobot_dataset.relative_action_stats --root datasets/pick-cube-remapped
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lerobot.processor import absolute_to_relative_pose_actions

# the values of ur5station/lerobot_train/pick-block-diffusion-config.json
HORIZON = 16
N_OBS_STEPS = 2
DROP_N_LAST_FRAMES = 7
QUANTILES = (1, 10, 50, 90, 99)  # the quantiles lerobot stores (q01 ... q99)


def episode_relative_action_chunks(
    actions: np.ndarray,
    states: np.ndarray,
    episode_index: np.ndarray,
    horizon: int,
    n_obs_steps: int,
    drop_n_last_frames: int,
) -> dict[int, tuple[np.ndarray, bool]]:
    """per episode: ((N_chunks * horizon, action_dim) relative actions, whether lerobot-train samples the episode).

    An episode that is too short to be sampled still gets the chunk of its first frame, only for its per-episode
    stats (lerobot needs stats for every episode); it does not contribute to the dataset stats."""
    offsets = np.arange(horizon) - (n_obs_steps - 1)
    chunks = {}
    for episode in np.unique(episode_index):
        index = np.flatnonzero(episode_index == episode)
        first, last = index[0], index[-1]
        sampled = len(index) > drop_n_last_frames
        starts = index[: len(index) - drop_n_last_frames] if sampled else index[:1]
        chunk_index = np.clip(starts[:, None] + offsets[None], first, last)
        chunk = torch.from_numpy(actions[chunk_index])  # (n_starts, horizon, action_dim)
        reference = torch.from_numpy(states[starts])  # (n_starts, state_dim)
        relative = absolute_to_relative_pose_actions(chunk, reference).reshape(-1, actions.shape[-1]).numpy()
        chunks[int(episode)] = (relative, sampled)
    return chunks


def relative_action_chunks(
    actions: np.ndarray,
    states: np.ndarray,
    episode_index: np.ndarray,
    horizon: int,
    n_obs_steps: int,
    drop_n_last_frames: int,
) -> np.ndarray:
    """(N_chunks * horizon, action_dim) relative actions of all training samples of the dataset."""
    episodes = episode_relative_action_chunks(actions, states, episode_index, horizon, n_obs_steps, drop_n_last_frames)
    return np.concatenate([relative for relative, sampled in episodes.values() if sampled])


def _stats(values: np.ndarray) -> dict[str, list]:
    """the statistics lerobot stores per feature (stats.json and the stats/<feature>/* episode columns)."""
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [int(values.shape[0])],
        **{f"q{q:02d}": np.quantile(values, q / 100, axis=0).tolist() for q in QUANTILES},
    }


def _write_episode_action_stats(root: Path, episode_stats: dict[int, dict[str, list]]) -> None:
    """replace the stats/action/* columns of meta/episodes/*.parquet, keeping the schema."""
    for path in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_pylist()
        for key in episode_stats[episodes[0]]:
            column = f"stats/action/{key}"
            if column not in table.column_names:
                continue
            field_index = table.schema.get_field_index(column)
            values = pa.array([episode_stats[ep][key] for ep in episodes], type=table.schema.field(column).type)
            table = table.set_column(field_index, table.schema.field(column), values)
        pq.write_table(table, path)


def write_relative_action_stats(
    root: Path,
    horizon: int = HORIZON,
    n_obs_steps: int = N_OBS_STEPS,
    drop_n_last_frames: int = DROP_N_LAST_FRAMES,
    state_key: str = "observation.state",
) -> None:
    root = Path(root)
    files = sorted((root / "data").glob("*/*.parquet"))
    table = pd.concat(pd.read_parquet(f, columns=["index", "episode_index", "action", state_key]) for f in files)
    table = table.sort_values("index")
    actions = np.stack(table["action"].to_numpy()).astype(np.float64)
    states = np.stack(table[state_key].to_numpy()).astype(np.float64)
    episodes = episode_relative_action_chunks(
        actions, states, table["episode_index"].to_numpy(), horizon, n_obs_steps, drop_n_last_frames
    )
    relative = np.concatenate([chunks for chunks, sampled in episodes.values() if sampled])

    _write_episode_action_stats(root, {episode: _stats(chunks) for episode, (chunks, _) in episodes.items()})
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"] = _stats(relative)
    stats_path.write_text(json.dumps(stats, indent=4))
    (root / "action_representation.json").write_text(
        json.dumps(
            {
                "action": "absolute target pose [x, y, z, rot6d(6), ...] in the base frame",
                "policy_input": "relative: SE3RelativeActionsProcessorStep, frame of the latest observation.state pose",
                "stats": "meta/stats.json action = RELATIVE action statistics",
                "use_relative_actions": True,
                "horizon": horizon,
                "n_obs_steps": n_obs_steps,
                "drop_n_last_frames": drop_n_last_frames,
            },
            indent=2,
        )
    )
    print(
        f"{root.name}: relative action stats over {relative.shape[0] // horizon} chunks, position range "
        f"{np.round(relative[:, :3].min(0), 3)} .. {np.round(relative[:, :3].max(0), 3)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Root directory of the local dataset")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="policy.horizon of the training config")
    parser.add_argument("--n-obs-steps", type=int, default=N_OBS_STEPS, help="policy.n_obs_steps")
    parser.add_argument("--drop-n-last-frames", type=int, default=DROP_N_LAST_FRAMES, help="policy.drop_n_last_frames")
    parser.add_argument("--state-key", default="observation.state", help="state feature, starts with [xyz, euler xyz]")
    args = parser.parse_args()
    write_relative_action_stats(args.root, args.horizon, args.n_obs_steps, args.drop_n_last_frames, args.state_key)
