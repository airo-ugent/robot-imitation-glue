# Debugging best practices

Imitation learning pipelines fail silently: a wrong image scaling or a mismatched action format does not
raise an exception, it just produces a policy that does not work. This page collects the debugging entry
points that exist in this codebase, and the pitfalls that have actually bitten us.

The golden rule: **isolate one stage at a time**, from the cheapest to the most expensive.

| Question | Tool |
|---|---|
| Is my dataset any good? | [replay it on the robot](#1-replay-recorded-episodes-on-the-robot) |
| Does the policy produce sane actions at all? | [run inference on recorded data](#2-run-inference-on-recorded-data-instead-of-live-hardware) |
| Does the control loop / recorder plumbing work? | [mock env](#3-hardware-free-testing-with-the-mock-env) |
| What is actually in my dataset? | [dataset utilities](#4-dataset-inspection-and-manipulation) |
| Everything looks right but the robot misbehaves | [gotchas](#5-common-gotchas) |
| `ImportError` / the wrong `airo-*` version | [environment gotchas](#6-environment-and-import-gotchas) |

Everything in the live loops (`collect_data`, `eval`) is logged to [Rerun](https://rerun.io/), which is
started with `rr.init("robot_imitation_glue", spawn=True)`. Loguru is used for the numeric logging
(target poses, timings, actions) — run with a debug log level to see the per-step target poses.

## 1. Replay recorded episodes on the robot

Replaying the recorded *actions* (not the observations) is the cheapest way to check that your dataset
actually encodes the demonstration you think it does, and that your action converters are each other's
inverse.

`replay_episode()` in [`robot_imitation_glue/lerobot_dataset/replay_episode.py`](../robot_imitation_glue/lerobot_dataset/replay_episode.py):

```python
replay_episode(env, dataset, action_to_env_converter, image_key, dataset_image_key, episode_idx=0, fps=10)
```

It moves the robot to the first action of the episode (after an `input()` confirmation), waits for a
second confirmation, and then feeds every recorded action through `action_to_env_converter` into
`env.act()` at `fps`. Target pose, current pose and the current `state` observation are logged at debug
level each step, so you can diff the commanded trajectory against the executed one.

A ready-made entry point for the UR5 station is
[`robot_imitation_glue/ur5station/replay.py`](../robot_imitation_glue/ur5station/replay.py). The dataset and
episode are arguments, with defaults at the top of the file:

```bash
python -m robot_imitation_glue.ur5station.replay --dataset datasets/<name> --episode 0
```

Notes:

- The converter must match the action format **of that dataset**. `replay.py` passes
  `abs_joint_policy_action_to_se3`, so it only works on a joint-action dataset (e.g. one produced by
  [`ur5station/prepare_datasets.py`](../robot_imitation_glue/ur5station/prepare_datasets.py)), not on the raw
  10-dim rot6d dataset written by `ur5station/data_collection.py`. Using the wrong converter does not
  error, it just moves the robot somewhere else.
- `image_key` is unused: `replay_episode` loads the dataset's initial image (`dataset_image_key`) but does not
  show it. To reproduce the recorded scene, set the scene up by hand while comparing against the dataset
  image — the `eval` loop does show this blending (see below).
- Replay is open loop w.r.t. the scene: if the object is not where it was during recording, a successful
  replay only tells you the *robot* trajectory is faithfully stored.

## 2. Run inference on recorded data instead of live hardware

Before booking the robot, check that the policy + preprocessor combination produces reasonable actions
offline. Two `__main__` blocks in the repo do exactly this.

**LeRobot policies** — [`robot_imitation_glue/agents/lerobot_agent.py`](../robot_imitation_glue/agents/lerobot_agent.py):

```bash
python -m robot_imitation_glue.agents.lerobot_agent --checkpoint <.../pretrained_model> --dataset <training dataset>
```

It loads the checkpoint with `make_lerobot_policy(pretrained_path, device)`, takes a frame straight
from the training dataset, and prints the predicted action next to `dataset[0]["action"]`. If those two do
not roughly agree on a *training* frame, the problem is the checkpoint or the dataset stats, not your robot.
The second half of that block then runs a `LerobotAgent` with a `observation_preprocessor` on randomly
generated observations, which is how you check that your preprocessor produces the exact keys, dtypes,
scaling and batch dimension that the policy expects, and how long it takes (both the preprocessor and the
inference time are logged in ms by `LerobotAgent.get_action`).

`make_lerobot_policy` returns the policy together with its `policy_preprocessor` / `policy_postprocessor`:
the policy itself does not (un)normalize, the processor pipelines saved next to the weights
(`policy_preprocessor.json` + `.safetensors` with the dataset stats) do. The training dataset is therefore
not needed for inference, but a `LerobotAgent` without those two processors would feed unnormalized inputs to
the policy and return normalized actions.

**OpenPI / remote policy servers** — [`robot_imitation_glue/agents/openpi_agent.py`](../robot_imitation_glue/agents/openpi_agent.py):

```bash
python -m robot_imitation_glue.agents.openpi_agent
```

Sends zero-filled observations of the right shape to the websocket policy server and times the round trip.
Use this to verify server reachability (and your SSH port forward) and inference latency against your
control period, before adding the robot to the equation.

### Teacher-forced replay: the most informative single experiment

There is no upstream script for it, but the following combination of the building blocks above is worth
writing once per project, because it separates "the policy is wrong" from "the scene is wrong" in a single
run. Drive the robot through an episode using the **recorded** actions (as in section 1), but at every step
*also* query the policy on the **live** observation and only log its prediction instead of executing it:

```python
dataset = LeRobotDataset(repo_id="", root=train_dataset_path)
policy, policy_preprocessor, policy_postprocessor = make_lerobot_policy(checkpoint_path, "cuda")
agent = LerobotAgent(policy, "cuda", preprocessor, policy_preprocessor, policy_postprocessor)
episode = dataset.meta.episodes[episode_idx]
start, stop = episode["dataset_from_index"], episode["dataset_to_index"]

agent.reset()                                    # clear the action chunk queue
for i in range(start, stop):
    observations = env.get_observations()
    predicted = agent.get_action(observations)   # logged, NOT executed
    ground_truth = dataset[i]["action"].cpu().numpy()
    rr.log(...)                                  # predicted vs ground_truth, live vs dataset image
    robot_pose, gripper = action_to_env_converter(env.get_robot_pose_se3(), env.get_gripper_opening(), ground_truth)
    env.act(robot_pose, gripper, time.time() + 1 / fps)
    precise_wait(...)
```

Because the robot follows the recorded trajectory exactly, the live observations stay on the training
distribution, so any divergence between `predicted` and `ground_truth` is the policy's (or the
preprocessor's) fault and not compounding drift. Log both action vectors per step (Rerun, or a CSV you can
plot afterwards) and you immediately see *which action dimensions* are wrong and *when* in the episode.
Keep a hard safety check in the loop (e.g. abort if the predicted or commanded TCP z drops below the table)
and wrap the loop in `try/finally` so your log is written even when you abort.

### Dump the raw input, the annotated overlay and the numbers, side by side

Whenever you debug anything perception-shaped (which image the policy actually saw, where a detector thinks
something is), write three artefacts per sample to a debug directory: the **raw** image exactly as it
entered the policy, an **overlay** image with your annotation drawn on a copy of it, and a small **JSON**
with the structured prediction/state for that sample. Same stem, different suffixes
(`sample_0000_raw.jpg`, `sample_0000_overlay.jpg`, `sample_0000.json`), so they sort together and you can
scrub through them later without re-running the robot.

The upstream building blocks are just `cv2.imwrite` / `json.dump` plus Rerun for the live view; the
important part is the discipline. Two things this catches that a live Rerun view does not: colour-channel
mistakes (write with `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` — the envs produce RGB, OpenCV writes BGR), and
crop/resize mistakes, because you are dumping the *post-preprocessor* tensor converted back to an image,
not the camera frame. Dumping the same frame as the dataset stores it next to the live one is the fastest
way to spot the preprocessing mismatches described in section 5.

### Shrink the problem: isolate one action dimension

When a policy "almost" works, restrict what it is allowed to do rather than staring at 7 or 10 numbers at
once. `eval()` takes the `policy_to_pose_converter` as an argument, so you can pass a wrapper that keeps
the dimension under test and freezes the rest at the current pose, e.g. only let the policy control z:

```python
def only_delta_z(current_pose, current_gripper, action):
    target_pose, gripper = real_converter(current_pose, current_gripper, action)
    frozen = current_pose.copy()
    frozen[2, 3] = target_pose[2, 3]
    return frozen, current_gripper
```

Same trick for the gripper alone, or for translation-without-rotation. This is much safer on hardware than
a full rollout and pinpoints which part of the action space the policy got wrong.

## 3. Hardware-free testing with the mock env

[`robot_imitation_glue/mock.py`](../robot_imitation_glue/mock.py) contains `MockEnv` (a 2D point on a
224x224 grid, rendered as a red "robot" blob and a green "target" blob), `MockAgent` (keyboard teleop on
`i`/`j`/`k`/`l`) and `mock_agent_to_pose_converter`. `MockEnv` implements the full `BaseEnv` interface the
real stations do — `get_observations()`, `act()`, `act_joints()`, `get_joint_configuration()`,
`get_robot_pose_se3()`, `get_gripper_opening()`, `move_robot_to_tcp_pose()`, `move_gripper()` — so it lets you
exercise the control loop, the keyboard state machine, the Rerun logging and the dataset recorder without
any hardware. Its "joint configuration" is simply the 2D grid position, and it declares an `ABS_EEF`
`ACTION_SPEC` (as does `MockAgent`). `test/test_mock_env.py` shows how to drive it from a test.

The `__main__` blocks of [`collect_data.py`](../robot_imitation_glue/collect_data.py) and
[`eval_agent.py`](../robot_imitation_glue/eval_agent.py) are the reference examples: they wire `MockEnv`,
`MockAgent`s and a `LeRobotDatasetRecorder` into `collect_data()` / `eval()`. `MockEnv.get_observations()`
returns the same image keys as the stations (`scene_image`, `wrist_image`), plus `state`, `joints` and
`gripper_state`. `test/test_joint_space_collection.py` drives `collect_data()` headless with scripted
keyboard events.

For a hardware-free check of the **recorder** specifically, run

```bash
uv run python -m robot_imitation_glue.dataset_recorder   # needs the lerobot extra
```

which writes three synthetic episodes with images and state to `datasets/` and reads them back with
`LeRobotDataset`. Handy to validate the feature-spec inference in `LeRobotDatasetRecorder.__init__`
(0D/1D → state features, 3D → image/video features) for a new observation dict.

## 4. Dataset inspection and manipulation

All of these live in [`robot_imitation_glue/lerobot_dataset/`](../robot_imitation_glue/lerobot_dataset/) and
operate on LeRobot datasets. They all leave the original dataset untouched or offer a `--backup` flag —
use it.

| Script | Purpose |
|---|---|
| `remove_lerobot_dataset_episode.py` | Delete bad episodes and renumber the remaining ones: `python -m robot_imitation_glue.lerobot_dataset.remove_lerobot_dataset_episode --repo-id <id> --root <dir> -e 1-5,7 -b` |
| `add_description_lerobot_dataset.py` | Add/overwrite the language instruction (`task`) of episodes: `... --repo-id <id> --root <dir> -e 0-20 --description "pick the cube" -b` |
| `remap_lerobot_keys.py` | Rename features to the `observation.*` keys the policies expect: `python -m robot_imitation_glue.lerobot_dataset.remap_lerobot_keys --root-dir <dir> --new-root-dir <dir> --feature-mapping '{"scene_image": "observation.images.scene_image"}'` |
| `transform_dataset.py` | The generic building block: `transform_dataset(root_dir, new_root_dir, transform_fn, transform_features_fn, features_to_drop, episodes_to_drop)` — writes a new dataset frame by frame. Use it for anything the CLIs do not cover (see `ur5station/prepare_datasets.py` for a real example that converts rot6d EEF actions to joint actions with IK and drops a broken episode). |
| `upload_dataset.py` | Push a local dataset to the HF hub. |
| `ur3station/zero_action_remover.ipynb` | Notebook that drops near-zero actions and binarises the gripper action (used for OpenVLA). |

Useful when writing a `transform_fn`: `transform_dataset` already strips the auto-generated
`index`/`timestamp`/`frame_index`/`episode_index`/`task_index` fields, unsqueezes 0D tensors to 1D and
permutes images back to channel-last before calling your function — so your frames look like what
`add_frame` expects, not like what `dataset[i]` returns. Keep the `task` key in the returned frame: lerobot
requires it in every frame.

For plain inspection, `LeRobotDataset(repo_id="", root=<dir>)` plus `dataset.meta`,
`dataset.features`, `dataset.num_episodes` and `dataset.meta.episodes[i]` (with the
`"dataset_from_index"`/`"dataset_to_index"` frame indices and the `"length"` of episode `i`) is usually all
you need. LeRobot's own `visualize_dataset` scripts work on
these datasets too.

## 5. Common gotchas

**Never draw on an observation you still need.** The visualisation code in `eval()` and `collect_data()`
annotates the scene image with `cv2.putText`, which writes *in place*. Because the same `observations` dict
is later handed to `policy_agent.get_action()` and `recorder.record_step()`, forgetting the `.copy()`
poisons both the policy input and the recorded dataset with the overlay text. This was a real bug, fixed in
`fddf17f` (`vis_image = observations[env_observation_image_key].copy()`). The same reasoning applies to
observation preprocessors: in `eval()`, `record_step(observations, action)` is called *after*
`policy_agent.get_action(observations)`, so a preprocessor that mutates the dict in place (`observation.pop(...)`,
in-place `transpose`/scaling — as the illustrative preprocessor in `lerobot_agent.py`'s `__main__` does)
will corrupt what is written to the eval dataset. Build and return a fresh dict, like the preprocessors in
`ur5station/eval_lerobot.py` do.

**Preprocessing must match between collection, training and inference.** The env already does part of the work
(`UR5eStation.preprocess_images()` resizes to 320x240, the UR3 crops and resizes to 224x224, and both keep the
full-resolution images under `*_image_original`), the dataset stores images channel-first, and the policy
expects channel-first floats in `[0, 1]`. The eval preprocessors therefore do `/255.0` + `permute(2, 0, 1)` +
`unsqueeze(0)`. Get any of these wrong and you get a policy that runs happily and outputs garbage. Watch out
in particular for cropping/resizing applied twice: commit `ce10010` ("centercrop is already handled in DP
policy") removed a center crop that the LeRobot diffusion policy was already applying internally, and
`OpenPIAgent` resizes every HWC uint8 image with `resize_with_pad` to 224x224 *on top of* whatever your
preprocessor did.

**Action format mismatches are silent.** The action stored in the dataset is whatever
`abs_pose_to_policy_action` produced during collection, and it is only meaningful together with the inverse
converter. On the UR5 station there are at least three formats in circulation: 10-dim
position+rot6d+gripper (`abs_se3_to_policy_action_converter` /`policy_action_to_abs_se3_converter` in
`ur5station/data_collection.py`), 7-dim absolute joints+gripper (`abs_joint_policy_action_to_se3`, produced
by `prepare_datasets.py`), and the Gello teleop format (`convert_abs_gello_actions_to_se3`, the same joints +
gripper width). `action_spaces.resolve_action_type` compares the `ACTION_SPEC`s of env and agents, but they are
`None` on the current stations, so it checks nothing there — check the converters by hand, and confirm with a
replay (section 1). The UR3 datasets record the gripper as a delta (±0.01 m); the OpenVLA decoder
(`relative_policy_action_to_abs_se3_converter`) expects 0/1, so binarize first (`zero_action_remover.ipynb`).

**Chunking policies must be reset between rollouts.** `OpenPIAgent` keeps an `action_queue`, LeRobot
policies keep an internal queue. `eval()` calls `policy_agent.reset()` at the start of every rollout; if you
write your own loop, do the same, otherwise the first actions of a rollout are leftovers from the previous
one.

**Actions can be rejected or clipped.** `UR5eStation.act()` (its `is_safe_pose`) refuses to execute when the
target z < 0, when the pose is unreachable, or when the gripper points upwards, and returns `False`. A target
more than `MAX_TRANSLATION = 0.15` m away is clipped to half of that, 7.5 cm, towards the target. Both only log
a warning. A policy that appears to be "stuck" or to move at half speed is very often being clipped here, so
read the warnings. `act()` normalises and clips `robot_pose_se3` in place, and both loops integrate targets
(`target_pose = new_robot_target_pose`) rather than reading back the measured pose, so a clip propagates
into the next step's target. After a refused pose both loops continue from the measured pose instead;
`collect_data()` does not record the refused step, `eval()` still records what the policy predicted.

**Recording into an existing directory continues that dataset.** `LeRobotDatasetRecorder` resumes the dataset
(`LeRobotDataset.resume`) if `root_dataset_dir` exists and continues from `n_recorded_episodes`. In `eval()` that counter also
selects which episode of the `eval_dataset` is used as initial-scene reference, so a continued eval picks up
the scenario list where it left off. If you want a fresh dataset, use a fresh directory.

**A recording session must end with `finish_recording()`.** lerobot keeps the data and episode-metadata
parquet files of a dataset open while recording and only writes their footers in `finalize()`, which
`finish_recording()` calls (`collect_data()` and `eval()` do so on quit; the recorder also registers it with
`atexit`). A process that is killed hard (SIGKILL, power loss) before that leaves a dataset that fails to load
with `Parquet magic bytes not found in footer`; the episodes saved in that session are then lost.

**Deleting an episode.** Pressing `d` in `collect_data()` cancels the episode that is being recorded. Saved
episodes cannot be deleted during collection: finish the session and use `remove_lerobot_dataset_episode.py`.

**Watch the loop timing.** `collect_data()` prints `cycle time exceeded control period` when a step takes
longer than `1/frequency`; `get_observations()` and `LerobotAgent.get_action()` log their duration (at debug
level). If you overrun the control period, the recorded dataset's fps no longer matches reality and the policy
will be trained on a different effective control rate than it is evaluated at.

**Set the scene up the same way you recorded it.** `eval()` accepts an `eval_dataset` and blends the initial
scene image of the corresponding episode over the live camera image in Rerun
(`rr.log("initial_scene_image", ...)`), which is the intended way to reproduce evaluation scenarios. It also
reads the language `instruction` from that dataset frame — but note it is only logged, so a VLA agent's
prompt still comes from wherever the agent takes it (`OpenPIAgent(default_prompt=...)`, or hardcoded in
`OpenVLAAgent.get_action`).

## 6. Environment and import gotchas

**The wrong `airo-*` / `lerobot` version is being imported.** The environment is managed with uv (see the
README). The vendored submodules — `airo-ipc/`, the `airo-mono/` packages, `lerobot/` (the `airo-ugent/lerobot`
fork of huggingface/lerobot, pinned in `.gitmodules`) and `airo-teleop/` — are installed as *editable* path dependencies
(`[tool.uv.sources]` in `pyproject.toml`) behind optional extras, so edits under a submodule take effect
immediately. Things that still go wrong:

- The submodules are not checked out (or not at the pinned commit): run
  `git submodule update --init --recursive`, then `uv sync` with the extras you need
  (e.g. `uv sync --extra lerobot --extra airo`). A bare `uv sync` installs none of them.
- You run code with another interpreter than the uv environment (a conda env, a system Python, an IDE
  pointing elsewhere), or another project's `airo-mono` / `lerobot` is on `PYTHONPATH`. Use `uv run ...`.

When behaviour does not match the code you are reading, check what actually got imported before you debug
anything else:

```python
import airo_robots, airo_camera_toolkit, lerobot
print(airo_robots.__file__, airo_camera_toolkit.__file__, lerobot.__file__)
```

The paths should point into this repository's submodules. This matters most for `lerobot`: the dataset
format and API differ noticeably between versions. The code here targets the pinned lerobot v0.6.1
(`lerobot.datasets...`, `dataset.add_frame({..., "task": ...})`, dataset format v3.0); an older lerobot
(`lerobot.common.datasets...`, format v2.1) fails on import, and cannot read v3.0 datasets. Datasets
recorded with the old format can be converted with lerobot's `convert_dataset_v21_to_v30.py` script.

**Import errors for modules that clearly exist.** `ipc_camera.py` and `grippers/schunk_process.py` live
under `robot_imitation_glue/hardware/` (moved in `5cb582e`; both stations import that location now). If an
import error mentions a module that clearly exists, check whether it moved, and whether the extra that
provides it (`hardware`, `airo`, ...) is installed. airo-robots' drivers are extras of airo-robots itself:
`URrtde` raises an `ImportError` without `airo-robots[ur]` (which the `airo` extra installs).
