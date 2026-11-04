"""
Shared environment for a robot station: a position-controlled arm with a parallel gripper (airo-robots
interfaces), a wrist camera and a scene camera, each published over ipc.

Actions as absolute target pose (rotation vector) in robot base frame and absolute gripper width
Proprioception as robot pose (euler angles) in robot base frame and gripper width

To add a station (e.g. a RealMan arm): subclass ManipulatorStation, create the robot (a PositionManipulator with
its gripper attached) and pass it together with the camera creation functions to super().__init__. Override
`preprocess_images` for the station's image resizing, `is_safe_pose` for extra safety checks on servo targets
and `servo_gripper` if the gripper has a faster call than `move`. See ur3station/ur3_robot_env.py and
ur5station/ur5_robot_env.py for examples.

The scene camera is optional: pass `create_scene_camera=None` for a station with only a wrist camera. The
observations then have no `scene_image` / `scene_image_original` keys and `preprocess_images` gets None for it.
"""

import time

import loguru
import numpy as np
from airo_spatial_algebra.se3 import SE3Container, normalize_so3_matrix

from robot_imitation_glue.base import BaseEnv

WRIST_CAM_RGB_TOPIC = "wrist_rgb"
WRIST_CAM_RESOLUTION_TOPIC = "wrist_resolution"
SCENE_CAM_RGB_TOPIC = "scene_rgb"
SCENE_CAM_RESOLUTION_TOPIC = "scene_resolution"

logger = loguru.logger


class ManipulatorStation(BaseEnv):
    ACTION_SPEC = None
    PROPRIO_OBS_SPEC = None

    def __init__(self, robot, create_wrist_camera, create_scene_camera=None):
        self.robot = robot
        self.gripper = robot.gripper
        self._setup_cameras(create_wrist_camera, create_scene_camera)
        # wait for first images
        time.sleep(2)

    def _setup_cameras(self, create_wrist_camera, create_scene_camera):
        # imported here so that the env can be constructed without the ipc stack (e.g. in tests)
        from robot_imitation_glue.hardware.ipc_camera import RGBCameraPublisher, RGBCameraSubscriber

        logger.info("Creating wrist camera publisher.")
        self._wrist_camera_publisher = RGBCameraPublisher(
            create_wrist_camera, WRIST_CAM_RGB_TOPIC, WRIST_CAM_RESOLUTION_TOPIC, 100
        )
        self._wrist_camera_publisher.start()
        logger.info("Creating wrist camera subscriber.")
        self._wrist_camera_subscriber = RGBCameraSubscriber(WRIST_CAM_RESOLUTION_TOPIC, WRIST_CAM_RGB_TOPIC)

        self._scene_camera_publisher = None
        self._scene_camera_subscriber = None
        if create_scene_camera is None:
            logger.info("No scene camera.")
            return
        logger.info("Creating scene camera publisher.")
        self._scene_camera_publisher = RGBCameraPublisher(
            create_scene_camera, SCENE_CAM_RGB_TOPIC, SCENE_CAM_RESOLUTION_TOPIC, 100
        )
        self._scene_camera_publisher.start()
        logger.info("Creating scene camera subscriber.")
        self._scene_camera_subscriber = RGBCameraSubscriber(SCENE_CAM_RESOLUTION_TOPIC, SCENE_CAM_RGB_TOPIC)

    # hooks for station-specific behaviour

    def preprocess_images(self, wrist_image, scene_image):
        """resized images for the policy, the originals are kept in the observation as well.

        `scene_image` is None for a station without a scene camera, return None for it then.
        """
        return wrist_image, scene_image

    def is_safe_pose(self, robot_pose_se3):
        """whether a servo target pose may be executed, may also adjust the pose in place"""
        return True

    def servo_gripper(self, width):
        """non-blocking gripper command, called at the control frequency"""
        self.gripper.move(width)

    # observations

    def get_joint_configuration(self):
        return self.robot.get_joint_configuration()

    def get_robot_pose_euler(self):
        """
        pose as [x,y,z,rx,ry,rz] in robot base frame using Euler angles
        """
        hom_pose = self.robot.get_tcp_pose()
        rotation_vector = SE3Container.from_homogeneous_matrix(hom_pose).orientation_as_euler_angles
        position = hom_pose[:3, 3]
        return np.concatenate((position, rotation_vector), axis=0)

    def get_robot_pose_se3(self):
        return self.robot.get_tcp_pose()

    def get_gripper_opening(self):
        return np.array([self.gripper.get_current_width()])

    def get_observations(self):
        start_time = time.time()
        self._wrist_camera_subscriber.grab_images()
        wrist_image = self._wrist_camera_subscriber.retrieve_rgb_image_as_int()
        scene_image = None
        if self._scene_camera_subscriber is not None:
            self._scene_camera_subscriber.grab_images()
            scene_image = self._scene_camera_subscriber.retrieve_rgb_image_as_int()
        robot_state = self.get_robot_pose_euler().astype(np.float32)
        gripper_state = self.get_gripper_opening().astype(np.float32)
        joints = self.robot.get_joint_configuration().astype(np.float32)

        state = np.concatenate((robot_state, gripper_state), axis=0)
        wrist_image_resized, scene_image_resized = self.preprocess_images(wrist_image, scene_image)

        obs_dict = {
            "wrist_image_original": wrist_image,
            "wrist_image": wrist_image_resized,
            "state": state,
            "robot_pose": robot_state,
            "gripper_state": gripper_state,
            "joints": joints,
        }
        if scene_image is not None:
            obs_dict["scene_image_original"] = scene_image
            obs_dict["scene_image"] = scene_image_resized
        logger.debug(f"get_observations time: {time.time() - start_time}")
        return obs_dict

    # actions

    def _duration_until(self, timestamp):
        duration = timestamp - time.time()
        if duration < 0:
            logger.warning("Action duration is negative, setting it to 0")
            duration = 0
        return duration

    def _clip_gripper_width(self, gripper_pose):
        if isinstance(gripper_pose, np.ndarray):
            gripper_pose = gripper_pose[0].item()
        return np.clip(gripper_pose, self.gripper.gripper_specs.min_width, self.gripper.gripper_specs.max_width)

    def act(self, robot_pose_se3, gripper_pose, timestamp):
        duration = self._duration_until(timestamp)
        logger.debug(f"Moving robot to pose \n {robot_pose_se3} with duration {duration}")

        robot_pose_se3[:3, :3] = normalize_so3_matrix(robot_pose_se3[:3, :3])
        executed = self.is_safe_pose(robot_pose_se3)
        if executed:
            self.robot.servo_to_tcp_pose(robot_pose_se3, duration)

        gripper_width = self._clip_gripper_width(gripper_pose)
        logger.debug(f"Setting gripper width to {gripper_width}")
        self.servo_gripper(gripper_width)

        # do not wait, handling timings is the responsibility of the caller
        return executed

    def act_joints(self, robot_joints, gripper_pose, timestamp):
        duration = self._duration_until(timestamp)
        logger.debug(f"Moving robot to joint configuration {robot_joints} with duration {duration}")

        self.robot.servo_to_joint_configuration(robot_joints, duration)

        gripper_width = self._clip_gripper_width(gripper_pose)
        logger.debug(f"Setting gripper width to {gripper_width}")
        self.servo_gripper(gripper_width)

        # do not wait, handling timings is the responsibility of the caller
        return

    def move_robot_to_tcp_pose(self, pose):
        self.robot.move_to_tcp_pose(pose).wait()

    def move_robot_to_joint_configuration(self, joints):
        self.robot.move_to_joint_configuration(joints).wait()

    def move_gripper(self, width):
        self.gripper.move(width).wait()

    def close(self):
        self._wrist_camera_publisher.stop()
        if self._scene_camera_publisher is not None:
            self._scene_camera_publisher.stop()
