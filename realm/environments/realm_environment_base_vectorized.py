import numpy as np
import yaml
import torch

from realm.environments.task_progressions import TASK_PROGRESSIONS
from realm.helpers import compute_rot_diff_magnitude, apply_blur_and_contrast
from realm.robots.droid_joint_controller import IndividualJointPDController
from realm.robots.droid_gripper_controller import MultiFingerGripperController
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.controllers import REGISTERED_CONTROLLERS
from omnigibson.object_states.open_state import _get_relevant_joints
from omnigibson.utils.object_utils import compute_base_aligned_bboxes, compute_bbox_offset
from omnigibson.utils.usd_utils import create_joint


REGISTERED_CONTROLLERS["CustomJointController"] = IndividualJointPDController
REGISTERED_CONTROLLERS["CustomGripperController"] = MultiFingerGripperController


class RealmEnvironmentBase:
    def __init__(
        self,
        og_vec_env,
        main_objects,
        target_objects,
        task,
        task_type,
        robots,
        use_droid_with_base,
        mo_cfgs
    ):
        self.env = og_vec_env
        print("DEBUG:", self.env)
        self.use_droid_with_base = use_droid_with_base

        self.main_objects = main_objects
        self.target_objects = target_objects

        self.mo_pos_orig = np.array([c["position"] for c in mo_cfgs])
        self.mo_rot_orig = np.array([c["orientation"] if "orientation" in c else [0, 0, 0, 1] for c in mo_cfgs])
        self.mo_bbox_orig = np.array([c["bounding_box"] for c in mo_cfgs])

        self.task = task
        self.task_type = task_type
        self.robots = robots
        self.robot_finger_links = [{robot._links[link] for link in robot.finger_link_names[robot.default_arm]} for robot in robots]

        self.was_lifted = False

        if task_type in TASK_PROGRESSIONS:
            self.task_progression = [TASK_PROGRESSIONS[task_type].copy() for _ in range(len(self.robots))]
        elif task in TASK_PROGRESSIONS:
            self.task_progression = [TASK_PROGRESSIONS[task].copy() for _ in range(len(self.robots))]
        else:
            self.task_progression = None

        if task_type in ["open_close_drawer"]:
            self.mo_joint = []
            for mo in main_objects:
                cabinet = mo[0]
                assert "cabinet" in cabinet.name
                relevant_joints = _get_relevant_joints(cabinet)[1]
                self.mo_joint.append(relevant_joints[0])
            self.joint_range = [j.upper_limit - j.lower_limit for j in self.mo_joint]
            self.init_openness_fraction = np.array(
                [(j.get_state()[0][0] - j.lower_limit) / r for j, r in zip(self.mo_joint, self.joint_range)])
        else:
            self.mo_joint = None

        self.success_conditions = {
            "REACH": self.check_reach_condition,
            "GRASP": self.check_grasp_condition,
            "TOUCH": self.check_touch_condition,
            "LIFT_SLIGHT": self.check_lift_slight_condition,
            "LIFT_LARGE": self.check_lift_large_condition,
            "ROTATED": self.check_rotated,
            "PUSH": self.check_push,
            "MOVE_CLOSE": self.check_move_close_condition,
            "PLACE_INTO": self.check_place_condition,
            "PLACE_ONTO": self.check_place_onto_condition,
            "TOUCH_AND_MOVE_JOINT": self.check_touching_and_moved_mo_joint,
            "OPEN_JOINT_SMALL": self.check_opened_mo_joint_small,
            "OPEN_JOINT_LARGE": self.check_opened_mo_joint_large,
            "OPEN_JOINT_FULL": self.check_opened_mo_joint_full,
            "CLOSE_JOINT_SMALL": self.check_closed_mo_joint_small,
            "CLOSE_JOINT_LARGE": self.check_closed_mo_joint_large,
            "CLOSE_JOINT_FULL": self.check_closed_mo_joint_full,
            "MOVE_JOINT_SMALL": self.check_moved_mo_joint_small,
            "MOVE_JOINT_LARGE": self.check_moved_mo_joint_large,
            "MOVE_JOINT_FULL": self.check_moved_mo_joint_full,
            "TOGGLED_ON": self.check_toggled_on_condition,
            "POURED": self.check_pour
        }

    def apply_scene_fixes_from_cfg(self, config_path, scene_model, scene_part):
        special_prims = yaml.load(open(f"{config_path}/scenes/scenes.yaml", "r"), Loader=yaml.FullLoader)
        if scene_model in special_prims and scene_part in special_prims[scene_model]:
            for env in self.env.envs:
                for obj in env.scene.objects:
                    if "to_fix" in special_prims[scene_model][scene_part] and obj.name in special_prims[scene_model][scene_part]["to_fix"]:
                        obj.fixed_base = True
                        create_joint(
                            prim_path=f"{obj.prim_path}/rootJoint",
                            joint_type="FixedJoint",
                            body1=f"{obj.prim_path}/{obj._root_link_name}",
                        )
                    if "to_remove" in special_prims[scene_model][scene_part] and obj.name in special_prims[scene_model][scene_part]["to_remove"]:
                        tmp = obj.name
                        env.scene.remove_object(obj)
                        assert env.scene.object_registry("name", tmp) is None

    def update_robot_physics(self, friction, armature):
        for env in self.env.envs:
            robot = env.robots[0]
            joint_names = robot.arm_joint_names
            for idx in range(7):
                prim_path = f"{robot.prim_path}/panda_link{idx}/{joint_names['0'][idx]}"
                joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)
                assert joint_prim.IsValid()
                joint_prim.GetAttribute("physxJoint:jointFriction").Set(friction[idx])
                joint_prim.GetAttribute("physxJoint:armature").Set(armature[idx])

    def reset(self, active_perturbations, supported_perturbations, config_path, scene_model, scene_part, reset_qpos):
        reset_output = self.env.reset()
        if reset_output is None:
            num_envs = self.env.num_envs
            #no_op_action = np.tile(np.append(reset_qpos[:7], -1), (num_envs, 1))
            no_op_action = torch.as_tensor(np.append(reset_qpos[:7], -1)).repeat(num_envs, 1).float()
            obs, _, _, _, _ = self.env.step(no_op_action)
            _ = {}
        else:
            obs, _ = reset_output

        self.apply_scene_fixes_from_cfg(config_path, scene_model, scene_part)
        #self.disable_visual_toggles() # TODO: turn this back on

        self.was_lifted = False
        if self.task_progression is not None:
             self.task_progression = [TASK_PROGRESSIONS[self.task_type].copy() for _ in range(len(self.robots))]

        if supported_perturbations is not None:
            for p in active_perturbations:
                supported_perturbations[p]()
        if "V-AUG" in active_perturbations:
            self.v_aug_sigma = np.random.uniform(0.0, 3.0)
            self.v_aug_alpha = np.random.uniform(0.5, 2.0)
            obs = apply_blur_and_contrast(obs, self.v_aug_sigma, self.v_aug_alpha)

        return obs, _

    def step(self, batched_action, active_perturbations):
        obs, rew, terminated, truncated, info = self.env.step(batched_action)

        task_progression = self.recompute_task_progression_vectorized(obs)

        if "V-AUG" in active_perturbations:
            obs = apply_blur_and_contrast(obs, self.v_aug_sigma, self.v_aug_alpha)

        return obs, task_progression, terminated, truncated, info

    def warmup(self, obs, active_perturbations):
        print("Starting warmup...")
        for _ in range(15):
            og.sim.render()

        is_gripper_closed = True
        for t in range(19):
            batched_joint_states = np.array([obs[idx]['franka']['proprio'][:7].cpu().numpy() for idx in range(len(obs))])
            batched_gripper_states = np.repeat(np.expand_dims(np.atleast_1d(np.array([-1])), axis=0), len(obs), axis=0)
            assert batched_joint_states.shape == (len(obs), 7)
            assert batched_gripper_states.shape == (len(obs), 1)
            new_batched_action = np.concatenate((
                batched_joint_states,
                batched_gripper_states
            ), axis=-1)
            if t != 0 and t % 10 == 0:
                is_gripper_closed = not is_gripper_closed
            new_batched_action[:, -1] = 1 if is_gripper_closed else -1

            obs, rew, terminated, truncated, info = self.step(torch.as_tensor(new_batched_action).float(), active_perturbations)

        self.mo_pos_orig = np.array([mo[0].get_position_orientation()[0] for mo in self.main_objects])
        self.mo_rot_orig = np.array([mo[0].get_position_orientation()[1] for mo in self.main_objects])
        print("Warmup finished.")
        return obs, rew, terminated, truncated, info

    # ============================== [SUCCESS METRICS] ==============================
    def recompute_task_progression_vectorized(self, obs):
        rewards = np.zeros(len(obs))
        if self.task_progression is not None and len(self.task_progression) > 0:
            stages = list(self.task_progression[0].keys())

            # Check conditions for all stages
            for stage in stages:
                 checker_function = self.success_conditions.get(stage)
                 if checker_function:
                    results = checker_function(obs) # Expected to return boolean array of shape (num_envs,)

                    for i in range(len(obs)):
                         if self.task_progression[i][stage] or results[i]:
                             if not self.task_progression[i][stage]:
                                 self.task_progression[i][stage] = True

            # Calculate rewards
            for i in range(len(obs)):
                reward = 0.0
                for stage, is_completed_flag in self.task_progression[i].items():
                    if is_completed_flag:
                        reward += 1 / len(self.task_progression[i].keys())
                    else:
                        break # Sequential task progression
                rewards[i] = reward

        return rewards

    def check_reach_condition(self, obs):
        is_touching = self.check_touch_condition(obs)
        if self.task_type in ["open_close_drawer"]:
            return is_touching

        res = np.array([
            np.linalg.norm(mo[0].get_position_orientation()[0] - list(finger_links)[0].get_position_orientation()[0]) < 0.1 or
            np.linalg.norm(mo[0].get_position_orientation()[0] - list(finger_links)[1].get_position_orientation()[0]) < 0.1
            for mo, finger_links in zip(self.main_objects, self.robot_finger_links)
        ])
        return res | is_touching

    def check_grasp_condition(self, obs):
        is_robot_touching_obj = self.check_touch_condition(obs)
        res = np.array([
            len(mo[0].states[ContactBodies].get_value().intersection(robot_finger_links)) == 2 and
            (0.45 - obs[i]['franka']['proprio'][7:9].cpu().numpy()[0] > 1e-3 or 0.45 - obs[i]['franka']['proprio'][7:9].cpu().numpy()[1] > 1e-3)
            for i, (mo, robot_finger_links) in enumerate(zip(self.main_objects, self.robot_finger_links))
        ])
        return res & is_robot_touching_obj

    def check_touch_condition(self, obs):
        res = np.array([
            robot.states[og.object_states.Touching].get_value(mo[0])
            for robot, mo in zip(self.robots, self.main_objects)
        ])
        return res

    def get_mo_joint_delta(self, obs):
        assert self.mo_joint is not None
        openness_fraction = self.get_mo_joint_openness_fraction(obs)
        delta_openness_fraction = self.init_openness_fraction - openness_fraction
        return delta_openness_fraction

    def get_mo_joint_openness_fraction(self, obs):
        assert self.mo_joint is not None
        return np.array([(j.get_state()[0][0] - j.lower_limit) / r for j, r in zip(self.mo_joint, self.joint_range)])

    def check_touching_and_moved_mo_joint(self, obs, threshold=0.025):
        delta_openness_fraction = self.get_mo_joint_delta(obs)
        is_touching = self.check_touch_condition(obs)
        if self.task == "open_drawer":
            return is_touching & (delta_openness_fraction > threshold)
        elif self.task == "close_drawer":
            return is_touching & (delta_openness_fraction < -threshold)
        else:
            raise NotImplementedError()

    def check_opened_mo_joint_small(self, obs):
        return self.get_mo_joint_openness_fraction(obs) > 0.125

    def check_opened_mo_joint_large(self, obs):
        return self.get_mo_joint_openness_fraction(obs) > 0.65

    def check_opened_mo_joint_full(self, obs):
        return self.get_mo_joint_openness_fraction(obs) > 0.95

    def check_closed_mo_joint_small(self, obs):
        return self.get_mo_joint_openness_fraction(obs) < 0.875

    def check_closed_mo_joint_large(self, obs):
        return self.get_mo_joint_openness_fraction(obs) < 0.35

    def check_closed_mo_joint_full(self, obs):
        return self.get_mo_joint_openness_fraction(obs) < 0.05

    def check_moved_mo_joint_small(self, obs):
        return self.check_closed_mo_joint_small(obs) | self.check_opened_mo_joint_small(obs)

    def check_moved_mo_joint_large(self, obs):
        return self.check_closed_mo_joint_large(obs) | self.check_opened_mo_joint_large(obs)

    def check_moved_mo_joint_full(self, obs):
        return self.check_closed_mo_joint_full(obs) | self.check_opened_mo_joint_full(obs)

    def check_rotated(self, obs, rot_threshold=1.1):
        mo_rot_curr = np.array([mo[0].get_position_orientation()[1] for mo in self.main_objects])
        mo_rot_orig = self.mo_rot_orig
        rot_diff = np.array([compute_rot_diff_magnitude(orig, curr) for orig, curr in zip(mo_rot_orig, mo_rot_curr)])
        return np.abs(rot_diff) > rot_threshold

    def check_lift_and_distance_condition(self, distance_threshold=0.05, lift_threshold=0.01):
        mo_pos_curr = np.array([mo[0].get_position_orientation()[0] for mo in self.main_objects])
        distance = np.linalg.norm(mo_pos_curr - self.mo_pos_orig, axis=1)
        return (mo_pos_curr[:, 2] - self.mo_pos_orig[:, 2] > lift_threshold) & (distance > distance_threshold)

    def check_lift_slight_condition(self, obs):
        return self.check_lift_and_distance_condition()

    def check_lift_large_condition(self, obs):
        return self.check_lift_and_distance_condition(distance_threshold=0.1, lift_threshold=0.075)

    def check_push(self, obs):
        push_cond = self.check_lift_and_distance_condition(distance_threshold=0.1, lift_threshold=-0.05)
        is_lifted = self.check_lift_and_distance_condition(distance_threshold=-0.05, lift_threshold=0.05)
        self.was_lifted |= is_lifted
        is_robot_touching_obj = self.check_touch_condition(obs)
        return push_cond & is_robot_touching_obj & ~self.was_lifted

    def check_move_close_condition(self, obs):
        assert len(self.main_objects) == len(self.target_objects)
        pos1 = np.array([mo[0].get_position_orientation()[0] for mo in self.main_objects])
        pos2 = np.array([to[0].get_position_orientation()[0] for to in self.target_objects])
        distance = np.linalg.norm(pos1 - pos2, axis=1)
        return distance < 0.125

    def check_place_condition(self, obs):
        inside_or_on_top = np.array([
            mo[0].states[og.object_states.OnTop].get_value(to[0]) or mo[0].states[og.object_states.Inside].get_value(
                to[0])
            for mo, to in zip(self.main_objects, self.target_objects)
        ])
        return inside_or_on_top & ~self.check_grasp_condition(obs)

    def check_place_onto_condition(self, obs):
        on_top = np.array([
            mo[0].states[og.object_states.OnTop].get_value(to[0])
            for mo, to in zip(self.main_objects, self.target_objects)
        ])
        return on_top & ~self.check_grasp_condition(obs)

    def check_toggled_on_condition(self, obs):
        return np.array([mo[0].states[og.object_states.ToggledOn].get_value() for mo in self.main_objects])

    def check_pour(self):
        return False
