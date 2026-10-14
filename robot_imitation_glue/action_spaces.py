"""Helpers to deal with the action space (EEF vs joint) of an env/agent pair.

Joint-space actions are stored for the policy/dataset as a flat vector

    [joints(n), gripper]

i.e. the absolute target joint configuration followed by the absolute target gripper width. This
mirrors the `[pos(3), rot-x(3), rot-y(3), gripper]` encoding that is used for absolute EEF actions
(see `ur5station/data_collection.py`).
"""

import loguru
import numpy as np

from robot_imitation_glue.base import ActionType, BaseEnv

logger = loguru.logger

JOINT_ACTION_TYPES = (ActionType.ABS_JOINT, ActionType.REL_JOINT)

# default bound on the joint motion that the loops send in a single control step, in rad (~11.5 degrees),
# measured w.r.t. the *measured* joint configuration. At 10 Hz this allows ~2 rad/s, which is well above
# normal teleop/policy motion but catches e.g. a teleop arm that does not match the robot or a policy that
# jumps to a different IK branch.
DEFAULT_MAX_JOINT_DELTA = 0.2


def is_joint_action_type(action_type: ActionType) -> bool:
    """whether actions of this type should be sent to `env.act_joints` instead of `env.act`"""
    return action_type in JOINT_ACTION_TYPES


def _spec_signature(action_spec):
    return tuple(
        getattr(action_spec, attribute, None) for attribute in ("action_type", "representation", "reference_frame")
    )


def resolve_action_type(env, *agents) -> ActionType:
    """check that the env and the agent(s) agree on the action space and return that action space.

    - Declared specs (ACTION_SPEC is not None) must be equal: same action_type, representation and
      reference_frame. Otherwise a ValueError is raised.
    - Nothing declared (still the case for e.g. the UR5/UR3 stations and the gello/spacemouse agents): this is
      the legacy setup, the actions are ABS_EEF poses as before joint actions existed.
    - Some components declare a spec and others do not: the undeclared ones cannot be checked. A warning is
      logged and the declared action type is used. This allows e.g. to use the (undeclared) UR5 station, which
      implements both `act` and `act_joints`, with a declared joint-space teleop agent. For a joint action type
      the env must implement `act_joints` though, otherwise a ValueError is raised.
    """
    components = (env, *agents)
    declared = {type(c).__name__: c.ACTION_SPEC for c in components if c.ACTION_SPEC is not None}
    undeclared = [type(c).__name__ for c in components if c.ACTION_SPEC is None]

    if len({_spec_signature(spec) for spec in declared.values()}) > 1:
        raise ValueError(
            "action space mismatch: "
            + ", ".join(
                f"{name} uses {spec.action_type.value if spec.action_type else None}"
                f" (representation={spec.representation}, reference_frame={spec.reference_frame})"
                for name, spec in declared.items()
            )
        )

    if not declared:
        return ActionType.ABS_EEF

    action_type = next(iter(declared.values())).action_type or ActionType.ABS_EEF
    if undeclared:
        logger.warning(
            f"{', '.join(undeclared)} do(es) not declare an ACTION_SPEC, assuming it uses the {action_type.value}"
            f" actions declared by {', '.join(declared)}. Declare ACTION_SPEC to have this checked."
        )
    if is_joint_action_type(action_type) and type(env).act_joints is BaseEnv.act_joints:
        raise ValueError(
            f"{type(env).__name__} does not implement act_joints, it cannot execute {action_type.value} actions"
        )
    return action_type


def joint_target_is_safe(current_joints, target_joints, max_joint_delta=DEFAULT_MAX_JOINT_DELTA) -> bool:
    """whether the target is within `max_joint_delta` (rad, per joint) of the current joint configuration.

    The loops refuse to send joint targets that are not, as servoing to them would make the robot jump.
    `max_joint_delta=None` disables the check."""
    if max_joint_delta is None:
        return True
    delta = np.max(np.abs(np.asarray(target_joints, dtype=np.float64) - np.asarray(current_joints, dtype=np.float64)))
    if delta > max_joint_delta:
        logger.error(
            f"refusing joint target {target_joints}: it is {delta:.3f} rad away from the current configuration"
            f" {current_joints}, the maximum per control step is {max_joint_delta} rad."
        )
        return False
    return True


def confirm_joint_approach(current_joints, target_joints):
    """ask the operator to confirm a (slow) joint move, raises if not confirmed"""
    delta = np.max(np.abs(np.asarray(target_joints) - np.asarray(current_joints)))
    answer = input(
        f"about to move the robot slowly (moveJ) from {np.round(current_joints, 3)} to the teleop configuration"
        f" {np.round(target_joints, 3)} (max joint delta {delta:.3f} rad). Press enter to move, anything else aborts: "
    )
    if answer.strip() != "":
        raise RuntimeError("joint approach aborted by the operator")


def approach_teleop_joint_configuration(env, teleop_agent, teleop_to_joints_converter, confirm=confirm_joint_approach):
    """move the robot slowly (env.move_robot_to_joint_configuration, i.e. moveJ) to the joint configuration of
    the joint-space teleop device, so that the servo loop starts without a jump. `confirm(current, target)` is
    called before moving (pass None to skip the confirmation)."""
    current_joints = env.get_joint_configuration()
    action = teleop_agent.get_action(env.get_observations())
    target_joints, _ = teleop_to_joints_converter(current_joints, env.get_gripper_opening(), action)
    if confirm is not None:
        confirm(current_joints, target_joints)
    logger.debug(f"moveJ robot to teleop joint configuration {target_joints}")
    env.move_robot_to_joint_configuration(target_joints)
    logger.debug("robot moved to teleop joint configuration")


def abs_joints_to_policy_action(robot_joints, gripper_width, abs_joints_action, gripper_action) -> np.ndarray:
    """absolute joint targets, encoded as [joints(n), gripper]"""
    del robot_joints, gripper_width  # the encoding is absolute, the current state is not needed
    policy_action = np.concatenate([np.asarray(abs_joints_action).ravel(), np.asarray(gripper_action).ravel()])
    return policy_action.astype(np.float32)


def policy_action_to_abs_joints(robot_joints, gripper_width, policy_action):
    """decode a [joints(n), gripper] action into an absolute joint configuration and gripper width.

    This is the inverse of `abs_joints_to_policy_action`. It can also be used as teleop converter for
    teleop devices that already provide absolute joint configurations, such as a Gello arm.
    """
    del robot_joints, gripper_width
    policy_action = np.asarray(policy_action, dtype=np.float64)
    return policy_action[:-1], policy_action[-1]
