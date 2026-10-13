# AGENTS.md

Guidance for coding agents working in this repository. Human-facing docs: `README.md` and `docs/`.

## What this is

Glue code for imitation learning on a real single-arm + gripper robot: teleoperated data collection into
LeRobot datasets, dataset editing/replay, and real-robot evaluation of policies (LeRobot ACT/DP, OpenVLA,
pi0, and the in-repo BC/DiT baselines), visualised with Rerun. Teleop devices and policies are both
`BaseAgent`s; robot setups are `BaseEnv`s.

## Setup

Python 3.12 only (`requires-python = ">=3.12,<3.13"`), managed with uv.

```bash
git submodule update --init --recursive   # airo-mono, airo-ipc, lerobot (tlpss fork, pinned), airo-teleop
uv sync                                   # core deps only; needs no submodule checkout (what CI uses)
uv sync --extra lerobot --extra baseline  # or --all-extras; extras are listed below
```

Extras (`pyproject.toml`): `airo` (airo-ipc + airo-mono packages), `lerobot` (vendored fork),
`teleop` (airo-teleop), `baseline` (torch/torchvision), `hardware` (RealSense, Schunk, Dynamixel,
SpaceMouse, `ur-analytic-ik`), `remote-policy` (OpenVLA HTTP client). All submodule packages are editable
path sources under `[tool.uv.sources]`. `openpi-client` is deliberately not installable here (it pins
numpy<2); pi0 eval needs a separate env. `pyzmq` is overridden to `>=26.2.1` to reconcile airo-robots and
lerobot; do not remove that override.

## Tests and CI

```bash
uv sync --frozen && uv run --frozen pytest test/   # exactly what .github/workflows/test.yaml runs
```

- With a bare `uv sync`, `test_baseline_policy.py`, `test_dit_policy.py` (need torch) and
  `test_dataset_recorder_smoke.py` (needs the pinned lerobot) are skipped via `pytest.importorskip`. To run
  them: `uv sync --extra baseline --extra lerobot` (requires the submodules).
- Tests must stay CPU-only and hardware-free. Use `robot_imitation_glue/mock.py` (`MockEnv`, which
  implements every `BaseEnv` method incl. `act_joints`) or duck-typed fakes (see
  `test/test_airo_teleop_agent.py`). New optional-dependency tests must `importorskip`, not fail.
- CI uses `--frozen`, not `--locked`, because the lock check would need the submodule metadata. If you
  change dependencies, run `uv lock` with submodules checked out and commit `uv.lock`.
- Formatting (`.github/workflows/pre-commit.yaml`): `pre-commit run --all-files` = black and isort at line
  length 119, autoflake, flake8 (config in `setup.cfg`).

## Architecture

```
robot_imitation_glue/
  base.py              BaseEnv, BaseAgent, BaseDatasetRecorder, ActionType, ActionSpec
  collect_data.py      collect_data(): teleop loop + keyboard state machine + recording
  eval_agent.py        eval(): teleop to set the scene, then policy rollouts, recorded as a dataset
  dataset_recorder.py  LeRobotDatasetRecorder (feature spec inferred from an example obs dict)
  mock.py              MockEnv / MockAgent / mock_agent_to_pose_converter (no hardware)
  agents/              gello/, spacemouse_agent, airo_teleop_agent (adapter), lerobot_agent,
                       openvla_agent, openpi_agent
  baseline/            plain-pytorch policies: BC (policy/train/agent) and DiT flow matching
                       (dit_policy/train_dit/dit_agent); read datasets through lerobot only
  lerobot_dataset/     dataset tools: transform_dataset, remap_lerobot_keys, remove/add-description,
                       replay_episode, upload_dataset
  hardware/            ipc_camera (airo-ipc shared-memory cameras), grippers/schunk_process
  ur5station/          UR5e + Schunk + Gello station: env, data_collection, prepare_datasets,
                       replay, eval_{act,diffusion}_lerobot, eval_pi0, lerobot_train/*.json configs
  uR3station/          UR3e + Robotiq + SpaceMouse station: env, collect_data, eval_openvla
```

Data flow: station `data_collection.py` -> `collect_data()` -> `LeRobotDatasetRecorder` writes
`datasets/<name>` -> optional `transform_dataset` step (e.g. `ur5station/prepare_datasets.py` renames keys
to `observation.*` and converts rot6d EEF actions to joint actions via IK) -> train with the lerobot fork
(configs in `ur5station/lerobot_train/`) or `python -m robot_imitation_glue.baseline.train[_dit]` ->
station `eval_*.py` wraps the policy in an agent and calls `eval()`, which records rollouts as a new dataset.

Both loops integrate targets: the converter receives the previous *target* pose/gripper, not the measured
pose. Converters have the signatures `teleop_to_pose_converter(target_pose, target_gripper, action) ->
(4x4 pose, gripper)` and `abs_pose_to_policy_action(cur_pose, cur_gripper, new_pose, new_gripper) -> action`.

## Conventions that are easy to get wrong

- **Action formats are only defined by their converter pair.** Nothing validates them:
  `collect_data` asserts `env.ACTION_SPEC == teleop_agent.ACTION_SPEC`, but both are `None` on the real
  stations. Formats in use: UR5 raw collection = 10-dim absolute `[xyz, rot x-axis, rot y-axis, gripper]`
  in the robot base frame (`ur5station/data_collection.py`); UR5 prepared = 7-dim absolute
  `[6 joints, gripper]` (`abs_joint_policy_action_to_se3`); Gello = 7-dim joints + normalised gripper
  (`convert_abs_gello_actions_to_se3`, maps to width as `(1 - g) * 0.08`); airo-teleop poses = 9-dim
  absolute `[xyz, x-axis, y-axis]` (+ optional gripper width) via `SE3_ACTION_DIM` in
  `agents/airo_teleop_agent.py`; UR3 = 7-dim relative `[dxyz, d-euler xyz, d-gripper]` expressed in the
  current EEF frame (`abs_se3_to_relative_policy_action_converter` in `uR3station/collect_data.py`), while its
  SpaceMouse teleop deltas are base-aligned (`delta_action_to_abs_se3_converter`). A wrong
  converter does not raise, it moves the robot elsewhere; confirm with a replay.
- **`ActionType` / `act_joints`.** `base.py` defines `ActionType` (`ABS_EEF`, `REL_EEF`, `ABS_JOINT`,
  `REL_JOINT`) and `BaseEnv.act_joints()`, implemented by `UR5eStation` and `MockEnv`. `collect_data()` and
  `eval()` currently always call `env.act()` with an SE3 pose, so joint-space policies are executed through
  forward kinematics in the converter. Keep `act()` and `act_joints()` non-blocking; the caller owns timing.
- **Observation keys.** Envs return flat dicts of numpy arrays; images are HWC uint8 RGB. `collect_data()`
  hardcodes `scene_image` and `wrist_image`; `eval()` takes `env_observation_image_key`. UR5 also returns
  `*_image_original` (full-res, usually dropped later), `state` (euler pose + gripper), `robot_pose`,
  `gripper_state`, `joints`. The recorder keeps keys as-is and declares image features channel-first; policies expect
  `observation.images.<name>` / `observation.state`, produced by `remap_lerobot_keys.py` or a
  `transform_dataset` step.
- **Preprocessing must match across collection, training and inference** (resize, crop, `/255`, CHW, batch
  dim). The baseline shares `to_chw_float_image` / `resize_image` between `train.py` and `agent.py`; reuse
  them. For lerobot agents the preprocessor in each `eval_*.py` is the contract; return a fresh dict, never
  mutate the observation in place (it is recorded after `get_action`). Never draw on an observation without
  `.copy()`.
- **Reset agents between episodes.** `eval()` calls `policy_agent.reset()` before each rollout. Chunking
  agents (`LerobotAgent`, `DiTAgent`, `OpenPIAgent`) keep action queues; any custom loop must reset too.
- **Pinned lerobot API.** Imports are `lerobot.common.datasets.lerobot_dataset` (the pinned fork), and the
  task is an argument: `dataset.add_frame(frame, task=...)`, not a frame key. Do not "upgrade" code to the
  newer upstream lerobot layout (`lerobot.datasets`) without bumping the submodule and the lock together.
  `make_lerobot_policy` needs the training dataset locally for normalisation stats.
- **Recording into an existing dir appends** to that dataset. Use a fresh directory for a fresh dataset.
- **Station scripts are configured by editing paths in `__main__`** (checkpoints, dataset roots, ports),
  not by CLI flags. Do not commit machine-specific paths you add.
- Do not commit `datasets/`, `outputs/`, checkpoints or large media (see `.gitignore`).

## Safety rules for agents

- Never run anything under `ur5station/`, `uR3station/`, `agents/gello/`, `agents/spacemouse_agent.py`,
  `hardware/`, `lerobot_dataset/replay_episode.py`, or the `__main__` of `collect_data.py` / `eval_agent.py`.
  These move a robot, open cameras or grippers, or grab the keyboard. Only a human operator at the station
  runs them; prepare the command and hand it over. `uR3station/collect_data.py` connects to the robot at
  module level, so do not even import it (`ur5station/prepare_datasets.py` likewise rewrites a dataset on
  import).
- Do not loosen the safety checks in `UR5eStation.act()` (z < 0, unreachable pose, gripper pointing up,
  `MAX_TRANSLATION` clipping) or add motion without a confirmation step.
- Keep tests and verification on CPU with mocks; do not launch GPU training or inference unasked.
- No `git push`, PRs, dataset uploads (`upload_dataset.py`, `push_to_hub`) or other remote writes unless the
  user explicitly asks. Do not modify submodule contents; change the pin deliberately or not at all.

## Further reading

- `docs/debugging.md`: replaying episodes, offline inference on recorded data, mock env, dataset tools,
  common pitfalls. Parts of sections 3 and 6 predate the latest merges: `MockEnv` is now instantiable,
  `environment.yaml` is gone (use uv, above), and the UR5 hardware imports are fixed.
- `robot_imitation_glue/agents/airo_teleop_agent.py` module docstring: airo-teleop action formats.
- `robot_imitation_glue/baseline/train.py`, `train_dit.py` docstrings: training CLI examples.
- `robot_imitation_glue/agents/gello/readme.md`: Gello calibration.
