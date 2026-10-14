import time

# create type for callable that takes obs and returns action
from typing import Callable

import cv2
import loguru
import numpy as np
import rerun as rr

from robot_imitation_glue.action_spaces import (
    DEFAULT_MAX_JOINT_DELTA,
    approach_teleop_joint_configuration,
    confirm_joint_approach,
    is_joint_action_type,
    joint_target_is_safe,
    resolve_action_type,
)
from robot_imitation_glue.base import BaseAgent, BaseDatasetRecorder, BaseEnv
from robot_imitation_glue.utils import precise_wait

converter_callable = Callable[dict[str, np.ndarray], np.ndarray]

logger = loguru.logger


class State:
    is_recording = False
    is_stopped = False
    is_paused = False


class Event:
    start_recording = False
    stop_recording = False
    delete_last = False
    pause = False
    resume = False
    quit = False

    def clear(self):
        for attr in self.__dict__:
            setattr(self, attr, False)


def init_keyboard_listener(event: Event, state: State):
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            # "space bar"
            if key == keyboard.Key.enter and not state.is_recording:
                event.start_recording = True

            elif key == keyboard.Key.enter and state.is_recording:
                event.stop_recording = True

            elif hasattr(key, "char") and key.char == "p" and not state.is_recording and not state.is_paused:
                # pause the episode
                event.pause = True

            elif hasattr(key, "char") and key.char == "p" and state.is_paused:
                # resume the episode
                event.resume = True

            elif hasattr(key, "char") and key.char == "q":
                event.quit = True

            elif hasattr(key, "char") and key.char == "d" and not state.is_recording:
                # delete the last episode
                event.delete_last = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener


def collect_data(  # noqa: C901
    env: BaseEnv,
    teleop_agent: BaseAgent,
    dataset_recorder: BaseDatasetRecorder,
    frequency=10,
    teleop_to_pose_converter: converter_callable = None,
    abs_pose_to_policy_action: converter_callable = None,
    teleop_to_joints_converter: converter_callable = None,
    abs_joints_to_policy_action: converter_callable = None,
    max_joint_delta: float = DEFAULT_MAX_JOINT_DELTA,
    confirm_joint_approach=confirm_joint_approach,
):
    """collect teleop demonstrations with the given env/agent pair.

    The action space of the env/agent pair (env.ACTION_SPEC.action_type) determines which converters
    are used and how the actions are sent to the env:

    - EEF actions: `teleop_to_pose_converter` and `abs_pose_to_policy_action` are used and the target
      pose is *integrated* over time (see the TODO at the end of the loop), actions are sent to `env.act`.
    - joint actions: `teleop_to_joints_converter` and `abs_joints_to_policy_action` are used and the
      converters receive the *measured* joint configuration of the robot (no integration, teleop
      devices such as a Gello arm already provide absolute joint targets), actions are sent to
      `env.act_joints`.
      Before the loop starts, the robot is moved slowly (`env.move_robot_to_joint_configuration`, after
      `confirm_joint_approach(current, target)`, pass None to skip the confirmation) to the configuration of the
      teleop device. Joint targets that are more than `max_joint_delta` rad away from the measured configuration
      are not sent (and not recorded), see `action_spaces.joint_target_is_safe`.

    See `action_spaces.resolve_action_type` for how the ACTION_SPECs of env and agent are checked.
    """
    action_type = resolve_action_type(env, teleop_agent)
    is_joint_action = is_joint_action_type(action_type)

    if is_joint_action:
        assert (
            teleop_to_joints_converter is not None and abs_joints_to_policy_action is not None
        ), f"env/agent use {action_type.value} actions, so the joint converters have to be provided."
    else:
        assert (
            teleop_to_pose_converter is not None and abs_pose_to_policy_action is not None
        ), f"env/agent use {action_type.value} actions, so the pose converters have to be provided."

    rr.init("robot_imitation_glue", spawn=True)

    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)

    control_period = 1 / frequency

    if is_joint_action:
        # absolute joint teleop devices (gello) are not integrated from the robot state, so the first command
        # would make the robot jump to the device's configuration: approach it slowly first.
        approach_teleop_joint_configuration(env, teleop_agent, teleop_to_joints_converter, confirm_joint_approach)

    target_pose = env.get_robot_pose_se3()
    target_gripper_state = env.get_gripper_opening()

    while not state.is_stopped:
        cycle_end_time = time.time() + control_period

        before_observation_time = time.time()
        observation = env.get_observations()
        after_observation_time = time.time()
        after_observation_time - before_observation_time
        # print("observation time: ", observation_time)

        # update & handle state machine events
        if not state.is_recording and event.start_recording:
            state.is_recording = True
            print("start recording")
            dataset_recorder.start_episode()

        elif state.is_recording and event.stop_recording:
            state.is_recording = False
            print("stop recording")
            # save episode
            dataset_recorder.save_episode()
            # TODO: allow for textual description of the episode?

        elif event.delete_last and not state.is_recording:
            print("delete last episode")
            # delete last episode
            raise NotImplementedError("delete last episode not implemented")

        elif event.pause and not state.is_recording:
            state.is_paused = True
            print("pause teleop")

        elif event.resume and state.is_paused:
            state.is_paused = False
            print("resume teleop")

        elif event.quit:
            print("quit")
            state.is_stopped = True
            listener.stop()
            dataset_recorder.finish_recording()
            return

        # clear all events
        event.clear()

        # update GUI.
        vis_img = observation["scene_image"].copy()

        # visualize state is_recording, is_paused
        if state.is_recording:
            cv2.putText(vis_img, "RECORDING", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if state.is_paused:
            cv2.putText(vis_img, "PAUSED", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(
            vis_img,
            f" # episodes: {dataset_recorder.n_recorded_episodes}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        rr.log("image", rr.Image(vis_img, rr.ColorModel.RGB))
        rr.log("wrist_image", rr.Image(observation["wrist_image"], rr.ColorModel.RGB))
        rr.log("scene_image", rr.Image(observation["scene_image"], rr.ColorModel.RGB))

        # if paused, do not collect teleop or execute action
        if state.is_paused:
            time.sleep(0.1)
            continue

        action = teleop_agent.get_action(observation)
        logger.info(f"Action: {action}")

        if is_joint_action:
            # joint targets are absolute joint configurations, there is no need to integrate a target
            # state: the converter simply gets the measured joint configuration of the robot.
            current_joints = env.get_joint_configuration()
            current_gripper_state = env.get_gripper_opening()

            new_joint_target, new_gripper_target_width = teleop_to_joints_converter(
                current_joints, current_gripper_state, action
            )

            policy_formatted_action = abs_joints_to_policy_action(
                current_joints, current_gripper_state, new_joint_target, new_gripper_target_width
            )

            if joint_target_is_safe(current_joints, new_joint_target, max_joint_delta):
                env.act_joints(
                    robot_joints=new_joint_target,
                    gripper_pose=new_gripper_target_width,
                    timestamp=time.time() + control_period,
                )
            else:
                # not executed, so do not record it as a demonstration step either
                policy_formatted_action = None

        else:
            new_robot_target_se3_pose, new_gripper_target_width = teleop_to_pose_converter(
                target_pose, target_gripper_state, action
            )

            # store the actions in absolute format, to facilitate any action conversion later on.
            # observation["target_abs_robot_se3e_pose"] = new_robot_target_se3_pose
            # observation["target_abs_gripper_pose"] = new_gripper_target_width

            policy_formatted_action = abs_pose_to_policy_action(
                target_pose, target_gripper_state, new_robot_target_se3_pose, new_gripper_target_width
            )

            env.act(
                robot_pose_se3=new_robot_target_se3_pose,
                gripper_pose=new_gripper_target_width,
                timestamp=time.time() + control_period,
            )

        if state.is_recording and policy_formatted_action is not None:
            dataset_recorder.record_step(observation, policy_formatted_action)

        # wait for end of the control period
        if cycle_end_time > time.time():
            precise_wait(cycle_end_time)
        else:
            print("cycle time exceeded control period")

        if not is_joint_action:
            # update the target pose and target gripper state for the next iteration
            target_pose = new_robot_target_se3_pose
            target_gripper_state = new_gripper_target_width

            # TODO: we now use 'integration' to get the next target pose instead of using the current pose.
            # this is to avoid 'shaking' of the robot, as is done in diffusion policy teleop for example.
            # but need to verify that this does not cause mismatch between teleop and policy.
            # and should also check if the distance between the target and the actual robot does not diverge too much.


if __name__ == "__main__":
    # create dummy env, agent and recorder to test the flow without hardware.
    import shutil
    from pathlib import Path

    from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
    from robot_imitation_glue.mock import MockAgent, MockEnv, mock_agent_to_pose_converter

    def mock_abs_pose_to_policy_action(robot_pose, gripper_pose, abs_se3_action, gripper_action):
        """absolute poses, encoded as [pos(3), rot-x(3), rot-y(3), gripper]"""
        del robot_pose, gripper_pose
        policy_action = np.zeros(10, dtype=np.float32)
        policy_action[:3] = abs_se3_action[:3, 3]
        policy_action[3:6] = abs_se3_action[:3, 0]
        policy_action[6:9] = abs_se3_action[:3, 1]
        policy_action[9] = gripper_action
        return policy_action

    env = MockEnv()
    env.reset()
    agent = MockAgent()

    dataset_name = "test_dataset"
    root_dataset_dir = Path(f"datasets/{dataset_name}")
    if root_dataset_dir.exists():
        shutil.rmtree(root_dataset_dir)

    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((10,), dtype=np.float32),
        root_dataset_dir=root_dataset_dir,
        dataset_name=dataset_name,
        fps=10,
    )

    print("use ijkl to move the agent, enter to start/stop recording an episode and q to quit.")

    # MockEnv/MockAgent use ABS_EEF actions, so the pose converters are used here. For a joint-space
    # env/agent pair (a UR5 with a Gello arm in joint mode, for example), pass the joint converters
    # (see robot_imitation_glue.action_spaces) instead.
    collect_data(
        env,
        agent,
        dataset_recorder,
        frequency=10,
        teleop_to_pose_converter=mock_agent_to_pose_converter,
        abs_pose_to_policy_action=mock_abs_pose_to_policy_action,
    )
