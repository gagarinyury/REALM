import math
import time

import isaacsim  # noqa: F401
import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy  # noqa: F401
import omnigibson.utils.transform_utils as T
import torch as th
import datetime
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from omnigibson.macros import gm
from tqdm.auto import tqdm

USE_DROID_WITH_BASE = True
if USE_DROID_WITH_BASE:
    from realm.robots.franka_robotiq_mounted import FrankaPandaRobotiq  # noqa: F401
else:
    from realm.robots.franka_robotiq import FrankaPandaRobotiq  # noqa: F401


from omnigibson.controllers import REGISTERED_CONTROLLERS
from realm.robots.droid_joint_controller import IndividualJointPDController
from realm.robots.droid_gripper_controller import MultiFingerGripperController
REGISTERED_CONTROLLERS["CustomJointController"] = IndividualJointPDController
REGISTERED_CONTROLLERS["CustomGripperController"] = MultiFingerGripperController


freq = 15
gm.DEFAULT_SIM_STEP_FREQ = freq
gm.DEFAULT_RENDERING_FREQ = freq
gm.DEFAULT_PHYSICS_FREQ = 120
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_HQ_RENDERING = True
gm.ENABLE_FLATCACHE = True
gm.USE_NUMPY_CONTROLLER_BACKEND = False

cfg = dict()

cfg["scene"] = {
    "type": "InteractiveTraversableScene",
    "scene_model": "Rs_int",
    "load_object_categories": ["floors", "walls"],
}

cfg["robots"] = [
    {
        "type": "FrankaPandaRobotiq",
        "name": "franka",
        "obs_modalities": ["proprio"],  # "rgb",
        "proprio_obs": ["joint_qpos"],
        "position": [0, 0, 0],  # 0.87],
        "orientation": T.euler2quat(
            th.tensor([0, 0, -math.pi], dtype=th.float32)
        ).tolist(),
        "control_freq": freq,
        "action_normalize": False,
        "controller_name": "CustomJointController",  # "JointController",
        "controller_config": {
            "arm_0": {
                "name": "CustomJointController",  # "JointController"
                "motor_type": "position",
                "control_freq": 15,
                "use_delta_commands": False,
                "use_impedances": True,
                "use_gravity_compensation": False,
                "use_cc_compensation": False,
                "Kq": [40, 30, 50, 35, 35, 25, 10],
                "Kqd": [4, 6, 5, 5, 3, 2, 1],
                "Kx": [400, 400, 400, 15, 15, 15],
                "Kxd": [37, 37, 37, 2, 2, 2],
                "command_output_limits": None,
                "command_input_limits": None,
            },
            "gripper_0": {
                "name": "CustomGripperController",
                "mode": "binary",
            },
        },
    }
]

cfg["task"] = {
    "type": "DummyTask",
    "termination_config": dict(),
    "reward_config": dict(),
}

cfg["env"] = {
    "external_sensors": [
        {
            "sensor_type": "VisionSensor",
            "name": "external_sensor0",
            "relative_prim_path": "/external_sensor0",
            "modalities": ["rgb"],
            "sensor_kwargs": {
                "image_height": 720,
                "image_width": 1280,
            },
            "position": th.tensor([-1.15716, 0, 0.73043], dtype=th.float32),
            "orientation": th.tensor([0.5, -0.5, -0.5, 0.5], dtype=th.float32),
            "pose_frame": "parent",
        },
    ],
}

cfg["objects"] = [
    {
        "type": "DatasetObject",
        "name": "table",
        "category": "breakfast_table",
        "model": "lcsizg",
        "position": [-0.6, -0.0, 0.55],
    },
    {
        "type": "PrimitiveObject",
        "name": "obj0",
        "primitive_type": "Cube",
        "fixed_base": False,
        "scale": [0.03, 0.03, 0.03],
        "position": [-0.49, 0.0, 0.715],
        "orientation": [0, 0, 0, 1],
    }
]
vec_env = og.VectorEnvironment(num_envs=5, config=cfg)

reset_output = vec_env.reset()
if reset_output is None:
    num_envs = vec_env.num_envs
    no_op_action = th.as_tensor(np.append(np.zeros(7), -1)).repeat(num_envs, 1).float()
    obs, _, _, _, _ = vec_env.step(no_op_action)
    _ = {}
else:
    obs, _ = reset_output

for e in vec_env.envs:
    scene = e.scene
    print("OG objects:", [o.name for o in scene.object_registry.objects])
    scene.wake_scene_objects()  # just in case, OG exposes this

traj_id = "traj_0"
grasp_state = np.array(
    [
        0.0,  # 0.25,
        1.0,  # 0.95
        0.0,
        -2,
        0.0,
        2.75,  # 2.825,
        0.0,
    ]
)

reach_state = grasp_state.copy()
reach_state[1] = 0
reach_state[6] += 0.3

video = []

close = False

start = time.time()
for t in tqdm(range(30)):
    sample = vec_env.envs[0].action_space.sample()
    base_im = np.concatenate([obs[i]["external"]["external_sensor0"]["rgb"].cpu().numpy()[..., :3] for i in range(vec_env.num_envs)])
    video.append(base_im)

    if t < 15:
        intended_state = reach_state
    else:
        intended_state = grasp_state
    sample["franka"][:-1] = intended_state  # robot_state
    sample["franka"][-1] = -1

    if t > 30:
        sample["franka"][0] -= 0.001 * (t - 100)
    if t > 45:
        sample["franka"][-1] = 1
    if t > 250:
        sample["franka"][1] -= 0.15

    print(t, sample)
    batched_action = np.expand_dims(sample["franka"], axis=0)
    observations, rewards, terminates, truncates, infos = vec_env.step(th.from_numpy(batched_action))

end = time.time()

total_time = end - start
print(total_time)

video = np.stack(video)
global_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
save_filename = f"./logs/{global_timestamp}_debug_vec_envs"
ImageSequenceClip(list(video), fps=15).write_videofile(
    save_filename + ".mp4", codec="libx264"
)

og.shutdown()
print("Done!")
