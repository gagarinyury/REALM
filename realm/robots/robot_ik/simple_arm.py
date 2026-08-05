"""Minimal Franka arm MJCF wrapper, without a dependency on dm_robotics.moma
(dm-robotics-moma/dm-robotics-controllers have no Windows wheel — see
robot_ik_solver.py for the full rationale). Exposes exactly the subset of the
original dm_robotics.moma.models.robots.robot_arms.robot_arm.RobotArm
interface that RobotIKSolver actually uses.
"""

import os

from dm_control import mjcf


class SimpleFrankaArm:
    def __init__(self):
        self._name = "franka"
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_file = os.path.join(dir_path, "franka", "panda.xml")
        self._mjcf_root = mjcf.from_path(model_file)
        self._joints = self._mjcf_root.find_all("joint")
        self._actuators = self._mjcf_root.find_all("actuator")
        self._wrist_site = self._mjcf_root.find("site", "wrist_site")
        self._base_site = self._mjcf_root.find("site", "base_site")

    @property
    def name(self):
        return self._name

    @property
    def joints(self):
        return self._joints

    @property
    def actuators(self):
        return self._actuators

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def wrist_site(self):
        return self._wrist_site

    @property
    def base_site(self):
        return self._base_site

    def update_state(self, physics, qpos, qvel):
        physics.bind(self._joints).qpos[:] = qpos
        physics.bind(self._joints).qvel[:] = qvel

    def set_joint_angles(self, physics, qpos):
        physics.bind(self._joints).qpos[:] = qpos
