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
git submodule update --init --recursive   # airo-mono, airo-ipc, lerobot (airo-ugent fork, pinned), airo-teleop
uv sync                                   # core deps only; needs no submodule checkout (what CI uses)
uv sync --extra lerobot --extra baseline  # or --all-extras; extras are listed below
```

Extras (`pyproject.toml`): `airo` (airo-ipc + airo-mono packages), `lerobot` (vendored, with its dataset,
training and diffusion extras),
`teleop` (airo-teleop), `baseline` (torch/torchvision), `hardware` (RealSense, Schunk, Dynamixel,
SpaceMouse, `ur-analytic-ik`), `remote-policy` (OpenVLA HTTP client). All submodule packages are editable
path sources under `[tool.uv.sources]`. `openpi-client` is deliberately not installable here (it pins
numpy<2); pi0 eval needs a separate env. airo-robots' hardware drivers are extras of airo-robots itself
(`airo-robots[ur]` here; add e.g. `[realman]` for a RealMan station). lerobot's own pins (e.g. `numpy<2.3`,
`datasets>=4.8`, `torch<2.12`) constrain the whole environment.

## Tests and CI

```bash
uv sync --frozen && uv run --frozen pytest test/   # exactly what .github/workflows/test.yaml runs
```

- With a bare `uv sync`, the tests that need torch, the pinned lerobot, the airo packages or `ur-analytic-ik`
  (e.g. `test_baseline_policy.py`, `test_dataset_recorder_smoke.py`, `test_transform_dataset_smoke.py`,
  `test_station_env.py`, `test_prepare_datasets.py`) are skipped via `pytest.importorskip`, so CI does not run
  them. To run everything: `uv sync --extra airo --extra teleop --extra hardware --extra lerobot --extra baseline`
  (requires the submodules).
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
  collect_data.py      collect_data(): teleop loop + keyboard/teleop-button state machine + recording
  eval_agent.py        eval(): teleop to set the scene, then policy rollouts, recorded as a dataset
  dataset_recorder.py  LeRobotDatasetRecorder (feature spec inferred from an example obs dict)
  preprocessing.py     action encodings, state/key/image conversions shared by collection, dataset prep, eval
  action_spaces.py     EEF vs joint action space of an env/agent pair, joint-target safety check
  station_env.py       ManipulatorStation: the shared env of the robot stations (airo-robots arm + gripper,
                       wrist/scene camera over ipc); stations override preprocess_images/is_safe_pose/servo_gripper
  mock.py              MockEnv / MockAgent / mock_agent_to_pose_converter (no hardware)
  agents/              airo_teleop_agent (adapter + make_gello/spacemouse/phone_agent(env, ...) for any station),
                       lerobot_agent, openvla_agent, openpi_agent
  baseline/            plain-pytorch policies: BC (policy/train/agent) and DiT flow matching
                       (dit_policy/train_dit/dit_agent); read datasets through lerobot only
  lerobot_dataset/     dataset tools: transform_dataset, remap_lerobot_keys, remove/add-description,
                       replay_episode, upload_dataset
  hardware/            ipc_camera (airo-ipc shared-memory cameras), grippers/schunk_process
  ur5station/          UR5e + Schunk + Gello station: ur5_robot_env, data_collection, prepare_datasets,
                       replay, eval_lerobot, eval_pi0, lerobot_train/*.json configs
  ur3station/          UR3e + Robotiq + SpaceMouse station: ur3_robot_env, data_collection, eval_openvla
```

Data flow: station `data_collection.py` -> `collect_data()` -> `LeRobotDatasetRecorder` writes
`datasets/<name>` -> optional `transform_dataset` step (e.g. `ur5station/prepare_datasets.py` renames keys
to `observation.*` and converts rot6d EEF actions to joint actions via IK) -> train with `lerobot-train`
(configs in `ur5station/lerobot_train/`) or `python -m robot_imitation_glue.baseline.train[_dit]` ->
station `eval_*.py` wraps the policy in an agent and calls `eval()`, which records rollouts as a new dataset.

For EEF actions both loops integrate targets: the converter receives the previous *target* pose/gripper, not
the measured pose. `env.act()` returns `False` when the env refuses a pose (UR5 safety checks); both loops then
continue from the measured pose, and `collect_data` does not record that step. Converters have the signatures
`teleop_to_pose_converter(target_pose, target_gripper, action) -> (4x4 pose, gripper)` and
`abs_pose_to_policy_action(cur_pose, cur_gripper, new_pose, new_gripper) -> action`.

## Conventions that are easy to get wrong

- **Action formats are only defined by their converter pair.** Nothing validates them:
  `action_spaces.resolve_action_type` compares the `ACTION_SPEC`s, but they are `None` on the real stations.
  Formats in use: UR5 raw collection = 10-dim absolute `[xyz, rot x-axis, rot y-axis, gripper]` in the robot base
  frame (`ur5station/data_collection.py`); UR5 prepared = 7-dim absolute `[6 joints, gripper]`
  (`abs_joint_policy_action_to_se3`); Gello = the same 7-dim joints + gripper width
  (`convert_abs_gello_actions_to_se3`, the UR5's airo-teleop `Gello4UR_ParallelGripper`); airo-teleop poses = 9-dim
  absolute `[xyz, x-axis, y-axis]` (+ optional gripper width) via `SE3_ACTION_DIM` in
  `agents/airo_teleop_agent.py`; UR3 = 7-dim relative `[dxyz, d-euler xyz, d-gripper]` expressed in the current EEF
  frame (`abs_se3_to_relative_policy_action_converter`, used in `ur3station/data_collection.py`), while its
  SpaceMouse teleop deltas (airo-teleop, `action_mode="delta"`) are base-aligned
  (`delta_action_to_abs_se3_converter`). The UR3 gripper is recorded as a delta but
  `relative_policy_action_to_abs_se3_converter` (OpenVLA) decodes 0/1, so binarize first
  (`ur3station/zero_action_remover.ipynb`). A wrong converter does not raise, it moves the robot elsewhere; confirm
  with a replay.
- **`ActionType` / `act_joints`.** `collect_data()` and `eval()` pick the action space with
  `action_spaces.resolve_action_type(env, agent)`: EEF actions go through the pose converters to `env.act()`,
  joint actions through the joint converters (which get the *measured* joint configuration) to
  `env.act_joints()`, after a slow approach move and with `max_joint_delta` rejection of large jumps. An env
  without an `ACTION_SPEC` (both real stations today) means absolute EEF poses, so the UR5 joint-space
  policies are still executed through forward kinematics in their converter. Keep `act()` and `act_joints()`
  non-blocking; the caller owns timing.
- **Observation keys.** Envs return flat dicts of numpy arrays; images are HWC uint8 RGB. `collect_data()`
  hardcodes `scene_image` and `wrist_image`; `eval()` takes `env_observation_image_key`. The stations
  (`ManipulatorStation`) also return `*_image_original` (full-res, usually dropped later), `state` (euler pose +
  gripper), `robot_pose`, `gripper_state`, `joints`. The recorder keeps keys as-is and declares image features
  channel-first; policies expect `observation.images.<name>` / `observation.state`, produced by
  `remap_lerobot_keys.py` or a `transform_dataset` step.
- **Preprocessing must match across collection, training and inference** (resize, crop, `/255`, CHW, batch dim).
  `preprocessing.py` is the single source of truth for the action encodings and key/image conversions; import from
  it instead of re-implementing a converter in a script. The baseline shares `to_chw_float_image` / `resize_image`
  between `train.py` and `agent.py`; reuse them. For lerobot agents the preprocessor in each `eval_*.py` is the
  contract; return a fresh dict, never mutate the observation in place (it is recorded after `get_action`). Never
  draw on an observation without `.copy()`.
- **Reset agents between episodes.** `eval()` calls `policy_agent.reset()` before each rollout. Chunking
  agents (`LerobotAgent`, `DiTAgent`, `OpenPIAgent`) keep action queues; any custom loop must reset too.
- **Pinned lerobot API (v0.6.1, dataset format v3.0).** The submodule is `airo-ugent/lerobot` (a fork of
  huggingface/lerobot; lab changes go to its `airo/main` branch), pinned at the v0.6.1 release. Imports are
  `lerobot.datasets.lerobot_dataset`, the task is a frame key (`add_frame({..., "task": ""})`), a dataset being
  recorded must be `finalize()`d (the recorder's `finish_recording()`) or it is unreadable, appending to an
  existing dataset goes through `LeRobotDataset.resume`, and episode boundaries are in
  `dataset.meta.episodes[i]["dataset_from_index" / "dataset_to_index"]`. Policies do not normalize themselves:
  `make_lerobot_policy` returns the policy plus the pre/post-processors saved with the checkpoint, and
  `LerobotAgent` needs both. Bump the submodule and `uv.lock` together. The `ur5station/lerobot_train/*.json`
  configs are full `TrainPipelineConfig` dumps for this version (`lerobot-train --config_path=...`).
- **Recording into an existing dir appends** to that dataset. Use a fresh directory for a fresh dataset.
- **Station scripts take CLI arguments whose defaults are constants at the top of the script**
  (checkpoints, dataset roots, ports), so they can be run as-is after editing those constants. Keep that
  pattern for new scripts, and do not commit machine-specific absolute paths as defaults.
- Do not commit `datasets/`, `outputs/`, checkpoints or large media (see `.gitignore`).

## Safety rules for agents

- Never run anything under `ur5station/`, `ur3station/`, `hardware/`, `lerobot_dataset/replay_episode.py`, or the
  `__main__` of `collect_data.py` / `eval_agent.py`.
  These move a robot, open cameras or grippers, or grab the keyboard. Only a human operator at the station
  runs them; prepare the command and hand it over. Importing these modules is fine (no side effects at import).
- Do not loosen the safety checks in `UR5eStation.is_safe_pose()` (z < 0, unreachable pose, gripper pointing
  up, `MAX_TRANSLATION` clipping) or the joint-target check (`action_spaces.joint_target_is_safe`), or add
  motion without a confirmation step (like the one before `UR5eStation` moves to `HOME_JOINTS`).
- Keep tests and verification on CPU with mocks; do not launch GPU training or inference unasked.
- No `git push`, PRs, dataset uploads (`upload_dataset.py`, `push_to_hub`) or other remote writes unless the
  user explicitly asks. Do not modify submodule contents; change the pin deliberately or not at all.

## Further reading

- `docs/debugging.md`: replaying episodes, offline inference on recorded data, mock env, dataset tools,
  common pitfalls.
- `robot_imitation_glue/agents/airo_teleop_agent.py` module docstring: airo-teleop action formats.
- `robot_imitation_glue/baseline/train.py`, `train_dit.py` docstrings: training CLI examples.
- Gello calibration: the UR5 uses airo-teleop's `GelloTeleopDevice.GELLO1_DEFAULT_CONFIG` (`GELLO_CONFIG` in
  `ur5station/ur5_robot_env.py`); recalibrate with airo-teleop's `airo-teleop-devices/calibration/gello_calibration.py`.
