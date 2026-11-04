"""tests the shared station env with fake hardware, without the ipc camera stack."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

# station_env needs the `airo` extra (airo-spatial-algebra)
pytest.importorskip("airo_spatial_algebra")

from robot_imitation_glue.station_env import ManipulatorStation  # noqa: E402


class FakeGripper:
    gripper_specs = SimpleNamespace(min_width=0.0, max_width=0.08)

    def __init__(self):
        self.width = 0.03
        self.commands = []

    def get_current_width(self):
        return self.width

    def move(self, width):
        self.commands.append(width)


class FakeRobot:
    def __init__(self):
        self.gripper = FakeGripper()
        self.joints = np.linspace(-1, 1, 6)
        self.servo_poses = []
        self.servo_joints = []

    def get_tcp_pose(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        return pose

    def get_joint_configuration(self):
        return self.joints

    def servo_to_tcp_pose(self, pose, duration):
        self.servo_poses.append((pose, duration))

    def servo_to_joint_configuration(self, joints, duration):
        self.servo_joints.append((joints, duration))


class FakeCamera:
    def grab_images(self):
        pass

    def retrieve_rgb_image_as_int(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)


class FakeStation(ManipulatorStation):
    def _setup_cameras(self, create_wrist_camera, create_scene_camera):
        self._wrist_camera_subscriber = create_wrist_camera()
        self._scene_camera_subscriber = create_scene_camera() if create_scene_camera is not None else None
        self._scene_camera_publisher = None

    def preprocess_images(self, wrist_image, scene_image):
        return wrist_image[::2, ::2], None if scene_image is None else scene_image[::4, ::4]


def make_station(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    return FakeStation(FakeRobot(), FakeCamera, FakeCamera)


def test_observations(monkeypatch):
    # the keys, dtypes and shapes are what the recorded datasets contain, so they must not change
    env = make_station(monkeypatch)
    obs = env.get_observations()

    assert set(obs.keys()) == {
        "wrist_image_original",
        "scene_image_original",
        "wrist_image",
        "scene_image",
        "state",
        "robot_pose",
        "gripper_state",
        "joints",
    }
    for key in ("wrist_image_original", "scene_image_original", "wrist_image", "scene_image"):
        assert obs[key].dtype == np.uint8
    assert obs["wrist_image_original"].shape == (480, 640, 3)
    assert obs["wrist_image"].shape == (240, 320, 3)
    assert obs["scene_image"].shape == (120, 160, 3)

    for key in ("state", "robot_pose", "gripper_state", "joints"):
        assert obs[key].dtype == np.float32
    assert obs["robot_pose"].shape == (6,)
    assert obs["gripper_state"].shape == (1,)
    assert obs["state"].shape == (7,)
    assert obs["joints"].shape == (6,)

    np.testing.assert_allclose(obs["robot_pose"][:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(obs["gripper_state"], [0.03])
    np.testing.assert_allclose(obs["state"], np.concatenate((obs["robot_pose"], obs["gripper_state"])))


def test_act_servos_robot_and_clips_gripper(monkeypatch):
    env = make_station(monkeypatch)
    pose = np.eye(4)
    pose[:3, 3] = [0.2, 0.2, 0.2]

    env.act(robot_pose_se3=pose, gripper_pose=np.array([0.5]), timestamp=time.time() + 0.1)

    assert len(env.robot.servo_poses) == 1
    np.testing.assert_allclose(env.robot.servo_poses[0][0], pose)
    assert 0 < env.robot.servo_poses[0][1] <= 0.1
    assert env.gripper.commands == [0.08]


def test_unsafe_pose_is_not_executed(monkeypatch):
    env = make_station(monkeypatch)
    env.is_safe_pose = lambda pose: False

    executed = env.act(robot_pose_se3=np.eye(4), gripper_pose=0.02, timestamp=time.time())

    assert executed is False
    assert env.robot.servo_poses == []
    assert env.gripper.commands == [0.02]


def test_act_joints(monkeypatch):
    env = make_station(monkeypatch)
    joints = np.ones(6)

    env.act_joints(robot_joints=joints, gripper_pose=-1.0, timestamp=0)

    assert len(env.robot.servo_joints) == 1
    np.testing.assert_allclose(env.robot.servo_joints[0][0], joints)
    assert env.robot.servo_joints[0][1] == 0
    assert env.gripper.commands == [0.0]


def test_observations_without_scene_camera(monkeypatch):
    # a station with only a wrist camera has no scene keys, instead of placeholder values
    monkeypatch.setattr(time, "sleep", lambda _: None)
    env = FakeStation(FakeRobot(), FakeCamera, None)
    obs = env.get_observations()

    assert "scene_image" not in obs and "scene_image_original" not in obs
    assert obs["wrist_image"].shape == (240, 320, 3)
    assert obs["state"].shape == (7,)
