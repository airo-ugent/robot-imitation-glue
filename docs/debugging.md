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
[`robot_imitation_glue/ur5station/replay.py`](../robot_imitation_glue/ur5station/replay.py). It is a script,
not a CLI: edit the dataset `root` and the episode index in the file, then

```bash
python robot_imitation_glue/ur5station/replay.py
```

Notes:

- The converter must match the action format **of that dataset**. `replay.py` passes
  `abs_joint_policy_action_to_se3`, so it only works on a joint-action dataset (e.g. one produced by
  [`ur5station/prepare_datasets.py`](../robot_imitation_glue/ur5station/prepare_datasets.py)), not on the raw
  10-dim rot6d dataset written by `ur5station/data_collection.py`. Using the wrong converter does not
  error, it just moves the robot somewhere else.
- The `image_key` / `dataset_image_key` arguments only feed the (currently commented out) overlay of the
  dataset's initial image on the live camera image, so `image_key` is effectively unused today. To
  reproduce the recorded scene, set the scene up by hand while comparing against the dataset image — the
  `eval` loop does have a working version of this blending (see below).
- Replay is open loop w.r.t. the scene: if the object is not where it was during recording, a successful
  replay only tells you the *robot* trajectory is faithfully stored.

## 2. Run inference on recorded data instead of live hardware

Before booking the robot, check that the policy + preprocessor combination produces reasonable actions
offline. Two `__main__` blocks in the repo do exactly this.

**LeRobot policies** — [`robot_imitation_glue/agents/lerobot_agent.py`](../robot_imitation_glue/agents/lerobot_agent.py):

```bash
python -m robot_imitation_glue.agents.lerobot_agent   # edit `path` and `dataset_path` at the top of __main__
```

It loads the checkpoint with `make_lerobot_policy(pretrained_path, dataset_path)`, takes a frame straight
from the training dataset, and prints the predicted action next to `dataset[0]["action"]`. If those two do
not roughly agree on a *training* frame, the problem is the checkpoint or the dataset stats, not your robot.
The second half of that block then runs a `LerobotAgent` with a `observation_preprocessor` on randomly
generated observations, which is how you check that your preprocessor produces the exact keys, dtypes,
scaling and batch dimension that the policy expects, and how long it takes (both the preprocessor and the
inference time are logged in ms by `LerobotAgent.get_action`).

Note that `make_lerobot_policy` needs the *training* dataset to be present locally, because the policy is
built from `dataset.meta` (normalisation stats). A common failure mode is pointing it at a differently
normalised dataset.

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
agent = LerobotAgent(make_lerobot_policy(checkpoint_path, train_dataset_path), "cuda", preprocessor)
start, stop = (dataset.episode_data_index[k][episode_idx].item() for k in ("from", "to"))

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

The `__main__` block of [`eval_agent.py`](../robot_imitation_glue/eval_agent.py) is the reference example:
it wires `MockEnv` + two `MockAgent`s + a `LeRobotDatasetRecorder` into `eval()`.

One thing to be aware of before you rely on it (a current limitation of `mock.py`, not of your setup):
`MockEnv.get_observations()` returns the keys `agent_pos` and `scene`, whereas `collect_data()` hardcodes
`observation["scene_image"]` and `observation["wrist_image"]` for its visualisation. The mock therefore
works with `eval()` (whose image key is configurable via `env_observation_image_key`) but not
out-of-the-box with `collect_data()`; override `get_observations()` in a subclass to add those keys.

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
| `uR3station/zero_action_remover.ipynb` | Notebook that drops near-zero actions and binarises the gripper action (used for OpenVLA). |

Useful when writing a `transform_fn`: `transform_dataset` already strips the auto-generated
`index`/`timestamp`/`frame_index`/`episode_index`/`task_index` fields, unsqueezes 0D tensors to 1D and
permutes images back to channel-last before calling your function — so your frames look like what
`add_frame` expects, not like what `dataset[i]` returns.

For plain inspection, `LeRobotDataset(repo_id="", root=<dir>)` plus `dataset.meta`,
`dataset.features`, `dataset.num_episodes` and `dataset.episode_data_index` (a dict with `"from"`/`"to"`
frame indices per episode) is usually all you need. LeRobot's own `visualize_dataset` scripts work on
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
`ur5station/eval_act_lerobot.py` and `ur5station/eval_diffusion_lerobot.py` do.

**Preprocessing must match between collection, training and inference.** The env already does part of the
work (`UR5eStation.get_observations()` resizes to 320x240 and keeps the full-resolution images under
`*_image_original`), the dataset stores images channel-first, and the policy expects channel-first floats
in `[0, 1]`. The eval preprocessors therefore do `/255.0` + `permute(2, 0, 1)` + `unsqueeze(0)`. Get any of
these wrong and you get a policy that runs happily and outputs garbage. Watch out in particular for
cropping/resizing applied twice: commit `ce10010` ("centercrop is already handled in DP policy") removed a
center crop that the LeRobot diffusion policy was already applying internally, and `OpenPIAgent` resizes
every HWC uint8 image with `resize_with_pad` to 224x224 *on top of* whatever your preprocessor did.

**Action format mismatches are silent.** The action stored in the dataset is whatever
`abs_pose_to_policy_action` produced during collection, and it is only meaningful together with the inverse
converter. On the UR5 station there are at least three formats in circulation: 10-dim
position+rot6d+gripper (`abs_se3_to_policy_action_converter` /`policy_action_to_abs_se3_converter` in
`ur5station/data_collection.py`), 7-dim absolute joints+gripper (`abs_joint_policy_action_to_se3`, produced
by `prepare_datasets.py`), and the Gello teleop format (`convert_abs_gello_actions_to_se3`). `collect_data`
asserts `env.ACTION_SPEC == teleop_agent.ACTION_SPEC`, but both are `None` on the current stations, so that
assert protects nothing — check the converters by hand, and confirm with a replay (section 1).

**Chunking policies must be reset between rollouts.** `OpenPIAgent` keeps an `action_queue`, LeRobot
policies keep an internal queue. `eval()` calls `policy_agent.reset()` at the start of every rollout; if you
write your own loop, do the same, otherwise the first actions of a rollout are leftovers from the previous
one.

**Actions can be silently rejected or clipped.** `UR5eStation.act()` refuses to execute when the target
z < 0, when the pose is unreachable, or when the gripper points upwards, and it *clips* translations larger
than `MAX_TRANSLATION = 0.15` m. It only logs a warning. A policy that appears to be "stuck" or to move at
half speed is very often being clipped here, so read the warnings. Note also that `act()` normalises and
clips `robot_pose_se3` in place, and both loops integrate targets (`target_pose = new_robot_target_pose`)
rather than reading back the measured pose, so a clip propagates into the next step's target.

**Recording into an existing directory continues that dataset.** `LeRobotDatasetRecorder` loads the dataset
if `root_dataset_dir` exists and continues from `n_recorded_episodes` (`192543f` fixed the episode count
being read from `episode_data_index` instead of `episode_data_index["from"]`). In `eval()` that counter also
selects which episode of the `eval_dataset` is used as initial-scene reference, so a continued eval picks up
the scenario list where it left off. If you want a fresh dataset, use a fresh directory.

**Deleting the last episode during collection is not implemented.** Pressing `d` in `collect_data()` raises
`NotImplementedError`. Finish the session and use `remove_lerobot_dataset_episode.py` instead.

**Watch the loop timing.** `collect_data()` prints `cycle time exceeded control period` when a step takes
longer than `1/frequency`; `get_observations()` and `LerobotAgent.get_action()` log their duration. If you
overrun the control period, the recorded dataset's fps no longer matches reality and the policy will be
trained on a different effective control rate than it is evaluated at.

**Set the scene up the same way you recorded it.** `eval()` accepts an `eval_dataset` and blends the initial
scene image of the corresponding episode over the live camera image in Rerun
(`rr.log("initial_scene_image", ...)`), which is the intended way to reproduce evaluation scenarios. It also
reads the language `instruction` from that dataset frame — but note it is only logged, so a VLA agent's
prompt still comes from wherever the agent takes it (`OpenPIAgent(default_prompt=...)`, or hardcoded in
`OpenVLAAgent.get_action`).

## 6. Environment and import gotchas

**The wrong `airo-*` / `lerobot` version is being imported.** The environment is managed with uv (see the
README). The vendored submodules — `airo-ipc/`, the `airo-mono/` packages, `lerobot/` (the `tlpss/lerobot`
fork, pinned in `.gitmodules`) and `airo-teleop/` — are installed as *editable* path dependencies
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
format and API differ noticeably between versions. The code here targets the pinned fork
(`lerobot.common.datasets...`, `dataset.add_frame(frame, task=...)`); a newer upstream lerobot
(`lerobot.datasets...`) fails on import or on `add_frame`.

**Import errors for modules that clearly exist.** `ipc_camera.py` and `grippers/schunk_process.py` live
under `robot_imitation_glue/hardware/` (moved in `5cb582e`; both stations import that location now). If an
import error mentions a module that clearly exists, check whether it moved, and whether the extra that
provides it (`hardware`, `airo`, ...) is installed.
