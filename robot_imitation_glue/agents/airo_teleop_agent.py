"""Adapter to use teleop agents from `airo-teleop <https://github.com/airo-ugent/airo-teleop>`_ in this repo.

Teleop device drivers and the agents that bind them to `airo-robots` manipulators live in the separate
airo-teleop repo, so that they can be shared between projects. This module contains a thin adapter that wraps
an ``airo_teleop_agents.teleop_agent.TeleopAgent`` and exposes it as a robot-imitation-glue
:class:`~robot_imitation_glue.base.BaseAgent`, i.e. it makes the airo-teleop agent usable with
:func:`robot_imitation_glue.collect_data.collect_data` and :func:`robot_imitation_glue.eval_agent.eval`.

Installation
------------
airo-teleop is not on PyPI. It is vendored as a git submodule at ``airo-teleop/``, and its two packages
(``airo-teleop-devices`` and ``airo-teleop-agents``) are wired up as editable path dependencies, behind an
optional extra (see ``pyproject.toml`` for its name)::

    git clone --recurse-submodules <this repo>   # or: git submodule update --init --recursive
    uv sync --extra <the airo-teleop extra>

Without the submodule, installing a clone of airo-teleop manually into the same environment also works::

    pip install -e <clone>/airo-teleop-devices -r <clone>/airo-teleop-devices/requirements.txt
    pip install -e <clone>/airo-teleop-agents -r <clone>/airo-teleop-agents/requirements.txt

The import of airo-teleop is lazy (see ``_import_teleop_agent_class``), so robot_imitation_glue keeps working
without it installed; you only need it when you actually instantiate an airo-teleop agent.

Action formats
--------------
An airo-teleop ``TeleopAgent.get_action()`` returns *absolute* targets, in one of three shapes:

1. a 1D array of joint angles, optionally with a gripper width appended (e.g. ``Gello4UR``);
2. a 4x4 homogeneous TCP pose matrix (e.g. ``SpaceMouse4PositionManipulator``, ``Phone4PositionManipulator``);
3. a ``(4x4 homogeneous TCP pose, gripper width)`` tuple (e.g. ``SpaceMouse4PositionManipulator_ParallelGripper``).

robot-imitation-glue expects agents to return a flat numpy array, which is turned into an
(SE3 pose, gripper width) target by the ``teleop_to_pose_converter`` that is passed to the data collection loop.
This adapter therefore flattens shapes 2 and 3 into the absolute 'position + 6D rotation (+ gripper)' encoding
that is already used elsewhere in this repo, and passes shape 1 (joint targets) through unchanged.

For the pose-based agents, :func:`airo_teleop_se3_action_to_pose` is the matching ``teleop_to_pose_converter``.
For joint-space agents you need a converter that runs forward kinematics for your robot, e.g.
``robot_imitation_glue.ur5station.ur5_robot_env.convert_abs_gello_actions_to_se3``. Note that airo-teleop's
Gello agents already return the gripper target as an absolute width in meters, whereas the vendored Gello agent
in this repo returns a normalized [0,1] value, so such a converter should not rescale the gripper value again.
"""

import numpy as np

from robot_imitation_glue.base import BaseAgent

# number of elements of the flattened absolute pose action: position (3) + x-axis (3) + y-axis (3)
SE3_ACTION_DIM = 9


def _import_teleop_agent_class():
    """Lazily import the airo-teleop TeleopAgent class, returns None if airo-teleop is not installed."""
    try:
        from airo_teleop_agents.teleop_agent import TeleopAgent
    except ImportError:
        return None
    return TeleopAgent


def se3_pose_to_action(pose: np.ndarray) -> np.ndarray:
    """Encode a 4x4 homogeneous pose as [position, rotation x-axis, rotation y-axis]."""
    action = np.zeros(SE3_ACTION_DIM, dtype=np.float64)
    action[:3] = pose[:3, 3]
    action[3:6] = pose[:3, 0]
    action[6:9] = pose[:3, 1]
    return action


def action_to_se3_pose(action: np.ndarray) -> np.ndarray:
    """Decode [position, rotation x-axis, rotation y-axis] into a 4x4 homogeneous pose."""
    x = action[3:6] / np.linalg.norm(action[3:6])
    y = action[6:9] - np.dot(action[6:9], x) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)

    pose = np.eye(4)
    pose[:3, 3] = action[:3]
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    return pose


def airo_teleop_se3_action_to_pose(current_pose, current_gripper_state, action: np.ndarray):
    """`teleop_to_pose_converter` for pose-based airo-teleop agents wrapped in an AiroTeleopAgent.

    The actions are absolute targets, so the current pose/gripper state are ignored. If the wrapped agent
    does not command the gripper, the current gripper state is kept.
    """
    del current_pose

    action = np.asarray(action, dtype=np.float64)
    pose = action_to_se3_pose(action)
    gripper = action[SE3_ACTION_DIM] if action.shape[0] > SE3_ACTION_DIM else current_gripper_state
    return pose, gripper


class AiroTeleopAgent(BaseAgent):
    """Adapter that exposes an airo-teleop TeleopAgent as a robot-imitation-glue BaseAgent.

    Example::

        from airo_robots.manipulators.hardware.ur_rtde import URrtde
        from airo_teleop_agents.spacemouse_teleop_agents import SpaceMouse4PositionManipulator

        from robot_imitation_glue.agents.airo_teleop_agent import AiroTeleopAgent, airo_teleop_se3_action_to_pose

        agent = AiroTeleopAgent(SpaceMouse4PositionManipulator(position_manipulator=URrtde(ip_address=...)))
        collect_data(env, agent, recorder, teleop_to_pose_converter=airo_teleop_se3_action_to_pose, ...)

    The main thing this adapter absorbs is the interface mismatch: airo-teleop agents are polled as
    ``get_action()`` without arguments, while this repo's :class:`BaseAgent` is polled as
    ``get_action(observation)``. Wrapping the agent avoids having to fork the data collection loop for it.

    The wrapped agent stays reachable as ``.teleop_agent`` (and its device as ``.teleop_device``), so that
    raw device inputs (e.g. ``agent.teleop_device.last_raw_inputs`` for the phone) remain available during
    data collection.

    Note that airo-teleop agents talk to the robot themselves to compute absolute targets (they hold a reference
    to an airo-robots manipulator), but they never *execute* actions; executing them remains the responsibility
    of the robot-imitation-glue environment.
    """

    ACTION_SPEC = None

    def __init__(self, teleop_agent, action_spec=None):
        """
        Args:
            teleop_agent: an airo-teleop `TeleopAgent` instance (or anything with a `get_action()` method).
            action_spec: optional ActionSpec, has to match the ActionSpec of the environment.
        """
        # duck-typed on purpose, so that this adapter (and its tests) do not require airo-teleop to be installed.
        if not hasattr(teleop_agent, "get_action"):
            hint = ""
            if _import_teleop_agent_class() is None:
                hint = " airo-teleop does not seem to be installed, see the module docstring for instructions."
            raise TypeError(f"{teleop_agent} has no get_action() method, so it is not a TeleopAgent.{hint}")

        self.teleop_agent = teleop_agent
        self.ACTION_SPEC = action_spec

    @property
    def teleop_device(self):
        """the raw teleop device of the wrapped agent, e.g. to read `last_raw_inputs`."""
        return getattr(self.teleop_agent, "teleop_device", None)

    def get_action(self, observation=None) -> np.ndarray:
        """Get the absolute action from the wrapped airo-teleop agent as a flat numpy array.

        The observation is ignored: airo-teleop agents read their teleop device (and the robot) themselves.
        """
        del observation

        action = self.teleop_agent.get_action()

        if isinstance(action, tuple):
            # (4x4 pose, gripper width)
            pose, gripper = action
            return np.concatenate([se3_pose_to_action(np.asarray(pose, dtype=np.float64)), [float(gripper)]])

        action = np.asarray(action, dtype=np.float64)

        if action.shape == (4, 4):
            # 4x4 pose, without gripper
            return se3_pose_to_action(action)

        if action.ndim == 1:
            # joint targets, optionally with a gripper width appended.
            return action

        raise ValueError(f"Cannot convert action of shape {action.shape} from {self.teleop_agent} to a glue action.")

    def reset(self):
        if hasattr(self.teleop_agent, "reset"):
            self.teleop_agent.reset()

    def close(self):
        if hasattr(self.teleop_agent, "close"):
            self.teleop_agent.close()
