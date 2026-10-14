import abc
import enum


class ActionType(enum.Enum):
    """the space in which actions are expressed."""

    ABS_EEF = "abs_eef"
    REL_EEF = "rel_eef"
    ABS_JOINT = "abs_joint"
    REL_JOINT = "rel_joint"


class ActionSpec:
    action_type = None
    representation = None
    reference_frame = None


class ProprioObservationSpec:
    representation = None
    reference_frame = None


class BaseEnv(abc.ABC):
    ACTION_SPEC = None
    PROPRIO_OBS_SPEC = None

    @abc.abstractmethod
    def get_observations(self):
        """observations as a dict of (str, np.ndarray)"""

    @abc.abstractmethod
    def act(self, robot_pose_se3, gripper_pose, timestamp):
        raise NotImplementedError

    def act_joints(self, robot_joints, gripper_pose, timestamp):
        """servo to an absolute target joint configuration (n,) and absolute gripper width, non-blocking.

        Only envs with ACTION_SPEC.action_type in (ABS_JOINT, REL_JOINT) need to implement this."""
        raise NotImplementedError

    def reset(self):
        pass

    @abc.abstractmethod
    def get_joint_configuration(self):
        """joint configuration as a (n,) numpy array"""

    @abc.abstractmethod
    def get_robot_pose_se3(self):
        """robot pose in base frame as a 4x4 numpy array"""

    @abc.abstractmethod
    def get_gripper_opening(self):
        """absolute gripper opening in meters as a (1,) numpy array"""

    @abc.abstractmethod
    def move_robot_to_tcp_pose(self, pose):
        """move robot to a given SE3 tcp pose"""

    @abc.abstractmethod
    def move_gripper(self, width):
        """move gripper to a given width"""

    @property
    def action_spec(self):
        return self.ACTION_SPEC

    @property
    def proprio_obs_spec(self):
        return self.PROPRIO_OBS_SPEC


class BaseAgent(abc.ABC):
    ACTION_SPEC = None

    @abc.abstractmethod
    def get_action(self, observation):
        pass

    def reset(self):
        pass

    @property
    def action_spec(self):
        return self.ACTION_SPEC


# define different PROPRIO_OBS_TYPES
# how are orientations represented?
# what are they relative to?


class BaseDatasetRecorder:
    def __init__(self):
        pass

    def start_episode(self):
        raise NotImplementedError

    def record_step(self, obs, action):
        raise NotImplementedError

    def save_episode(self):
        raise NotImplementedError

    def set_episode_success(self, success):
        """Label the current (unsaved) episode with its outcome. This is an *episode-level* label: every
        frame recorded so far in the episode gets `next.success = success` (overwriting the per-step
        default of False written by record_step()). Call it once the outcome is known, right before
        save_episode(); outcomes are often only knowable in hindsight (e.g. a check-then-recover loop),
        so a per-step flag cannot label the frames where the task actually completes.

        Note that this differs from the usual (gym/lerobot) per-step meaning of `next.success`, where only
        the terminal frame(s) are True. Consumers should treat the column as constant per episode, e.g.
        keep episodes whose frames are all True. Frames recorded after this call get the default False
        again, so call it last."""
        raise NotImplementedError

    def delete_episode(self):
        """Discard the current (unsaved) episode. Already saved episodes are not affected; the next
        recorded episode reuses the discarded episode's index."""
        raise NotImplementedError

    def finish_recording(self):
        pass

    @property
    def n_recorded_episodes(self):
        pass
