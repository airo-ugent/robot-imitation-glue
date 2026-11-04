"""
Environement for Robot Station with 2 realsense cameras and a UR3e robot with a Robotiq Gripper
"""

import time

import cv2
import loguru
import numpy as np
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85
from airo_robots.manipulators.hardware.ur_rtde import URrtde

from robot_imitation_glue.station_env import ManipulatorStation

WRIST_REALSENSE_SERIAL = "944122073290"
# None for a station without a scene camera (it used to be "231122072220")
SCENE_REALSENSE_SERIAL = None

ROBOT_IP = "10.42.0.162"

logger = loguru.logger


class CameraFactory:
    def create_wrist_camera():
        return Realsense(resolution=Realsense.RESOLUTION_480, fps=30, serial_number=WRIST_REALSENSE_SERIAL)

    def create_scene_camera():
        return Realsense(resolution=Realsense.RESOLUTION_480, fps=30, serial_number=SCENE_REALSENSE_SERIAL)


class UR3eStation(ManipulatorStation):
    def __init__(self):
        logger.info("connecting to gripper.")
        gripper = Robotiq2F85(ROBOT_IP)
        logger.info("connecting to robot.")
        robot = URrtde(ROBOT_IP, URrtde.UR3E_CONFIG, gripper=gripper)
        create_scene_camera = CameraFactory.create_scene_camera if SCENE_REALSENSE_SERIAL is not None else None
        super().__init__(robot, CameraFactory.create_wrist_camera, create_scene_camera)

    def preprocess_images(self, wrist_image, scene_image):
        # resize wrist image to 224x224, cut the first 200 x pixels of the scene image and resize to 224x224
        wrist_image_resized = cv2.resize(wrist_image, (224, 224))
        scene_image_resized = None if scene_image is None else cv2.resize(scene_image[:, 200:], (224, 224))
        return wrist_image_resized, scene_image_resized

    def servo_gripper(self, width):
        # `move` sleeps 10 ms per call and returns an awaitable, too slow for every control step
        self.gripper._set_target_width(width)


if __name__ == "__main__":
    # moves the robot up and down between z = 0.1 and 0.2 m while showing the cameras, press q to quit
    from robot_imitation_glue.preprocessing import delta_action_to_abs_se3_converter

    env = UR3eStation()
    cv2.namedWindow("wrist", cv2.WINDOW_NORMAL)
    cv2.namedWindow("scene", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("wrist", 640, 480)
    cv2.resizeWindow("scene", 640, 480)

    direction = 1
    while True:
        obs = env.get_observations()
        cv2.imshow("wrist", obs["wrist_image"])
        if "scene_image" in obs:
            cv2.imshow("scene", obs["scene_image"])

        current_z = obs["robot_pose"][2]
        if current_z > 0.2:
            direction = -1
        elif current_z < 0.1:
            direction = 1
        # [dx,dy,dz, drx,dry,drz, dgripper]: move along z and rotate around z, gripper unchanged
        action = np.array([0, 0, 0.02 * direction, 0, 0, 0.02 * direction, 0])
        robot_se3, gripper = delta_action_to_abs_se3_converter(env.get_robot_pose_se3(), obs["gripper_state"], action)
        env.act(robot_pose_se3=robot_se3, gripper_pose=gripper, timestamp=time.time() + 0.1)
        if cv2.waitKey(100) == ord("q"):
            break

    env.close()
