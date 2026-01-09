import numpy as np
import torch
from queue import Queue
import datetime
import os
import random
import omnigibson as og
from omnigibson.macros import gm
#from realm.environments.realm_environment_dynamic import RealmEnvironmentDynamic
from realm.environments.realm_environment_dynamic_vectorized import RealmEnvironmentDynamic
from realm.inference import InferenceClient, extract_from_obs
from realm.logging import VideoRecorder, save_results_to_csv


SUPPORTED_TASKS = [
    "put_green_block_in_bowl", #0
    "put_banana_into_box", #1
    "rotate_marker", #2
    "rotate_mug", #3
    "pick_spoon", #4
    "pick_water_bottle", #5
    "stack_cubes", #6
    "push_switch", #7
    "open_drawer", #8
    "close_drawer", #9
]

SUPPORTED_PERTURBATIONS = [
    'Default', #0
    'V-AUG', # 1
    'V-VIEW', # 2
    'V-SC', # 3
    'V-LIGHT', # 4
    'S-PROP', # 5
    'S-LANG', # 6
    'S-MO', # 7
    'S-AFF', # 8
    'S-INT', # 9
    'B-HOBJ', # 10
    'SB-NOUN', # 11
    'SB-VRB', # 12
    'VB-POSE', # 13
    'VB-MOBJ', # 14
    'VSB-NOBJ' # 15
]


def set_sim_config():
    gm.DEFAULT_SIM_STEP_FREQ = 15
    gm.DEFAULT_RENDERING_FREQ = 15
    gm.DEFAULT_PHYSICS_FREQ = 120
    gm.ENABLE_TRANSITION_RULES = False # this needs to be off to avoid bug with sludge state during collision: https://github.com/StanfordVL/BEHAVIOR-1K/issues/1201
    gm.ENABLE_OBJECT_STATES = True # this needs to be on because push_switch task usees the ToggledOn state

    # gm.USE_GPU_DYNAMICS = False
    # gm.ENABLE_HQ_RENDERING = True  # True
    # gm.ENABLE_FLATCACHE = True
    # gm.HEADLESS = headless
    # gm.USE_NUMPY_CONTROLLER_BACKEND = False
    # if appdata_path is not None:
    #     gm.APPDATA_PATH = appdata_path

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(
        task_id=0,
        perturbation_id=0,
        repeats=1,
        max_steps=500,
        horizon=8,
        model_type="pi0_FAST",
        port=8000,
        num_envs=1,
        rmgb_path=None,
        log_dir="/app/logs"
):
    set_sim_config()
    if rmgb_path is None:
        rmgb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    task = SUPPORTED_TASKS[task_id]
    perturbations = [SUPPORTED_PERTURBATIONS[perturbation_id]]

    os.makedirs(log_dir, exist_ok=True)

    client = InferenceClient(model_type, port)

    env = RealmEnvironmentDynamic(
        config_path=f"{rmgb_path}/realm/config",
        task=task,
        perturbations=perturbations,
        num_envs=num_envs
    )

    global_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    results = []

    video_recorders = [VideoRecorder(log_dir, global_timestamp, repeats) for _ in range(repeats)]
    task_progression = [0.0 for _ in range(repeats)]
    task_progression_timestamps = [[] for _ in range(repeats)]

    for batch_env_idx in range(repeats // num_envs):
        observations, _ = env.reset(
            env.omnigibson_vector_env,
            env.active_perturbations,
            None,  # self.realm_env.supported_pertrubations,
            env.config_path,
            env.scene_model,
            env.scene_part
        )
        observations, rewards, terminates, truncates, infos = env.warmup(observations, env.omnigibson_vector_env, env.active_perturbations)

        action_buffer = [Queue() for _ in range(num_envs)]
        instruction = env.instruction

        for t in range(max_steps):
            batched_actions = []
            for env_idx in range(num_envs):
                run_id = num_envs * batch_env_idx + env_idx
                base_im, base_im_second, wrist_im, robot_state, gripper_state = extract_from_obs(observations[env_idx])

                if action_buffer[env_idx].empty():
                    pred_action_chunk = client.infer(
                        instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
                        use_base_im_second=(env.task_type == "open_close_drawer" if hasattr(env, "task_type") else False)
                    )

                    if len(pred_action_chunk.shape) == 2:
                        assert pred_action_chunk.shape[-1] == 8
                        for action in pred_action_chunk[:horizon]:
                            action = np.squeeze(action)
                            action_buffer[env_idx].put(action)
                    else:
                        action_buffer[env_idx].put(pred_action_chunk)

                video_recorders[run_id].add_frame(base_im, wrist_im)

                action = action_buffer[env_idx].get()

                new_joint_action = action.copy()[:7]
                new_gripper_state = 1 if action[7] > 0.5 else -1  # Prediction: (1,0) -> Target: (1,-1)
                new_gripper_state = np.atleast_1d(np.array(new_gripper_state))
                new_action = np.concatenate((new_joint_action, new_gripper_state))
                batched_actions.append(new_action)

            observations, rewards, terminates, truncates, infos = env.step(
                np.array(batched_actions), env.omnigibson_vector_env, env.active_perturbations
            )

            for idx, reward in enumerate(rewards):
                run_idx = num_envs * batch_env_idx + idx
                if reward > task_progression[run_idx]:
                    task_progression[run_idx] = reward
                    task_progression_timestamps[run_idx].append(t)

            print(batch_env_idx, t,
                  task_progression[num_envs * batch_env_idx: num_envs * batch_env_idx + num_envs])
            if all(prog == 1.0 for prog in
                   task_progression[num_envs * batch_env_idx: num_envs * batch_env_idx + num_envs]):
                print(f"All environments finished at step {t}")
                break

    for run_id in range(repeats):
        results.append({
            "task": task,
            "perturbation": perturbations,
            "model": model_type,
            "real2sim": "Simulated",
            "task_progression": task_progression[run_id],
            "task_progression_timestamps": task_progression_timestamps[run_id],
            "binary_SR": 1.0 if task_progression[run_id] == 1.0 else 0.0,
        })

        save_filename = os.path.join(log_dir, f"{global_timestamp}_{model_type}_rollout_{task}_{perturbations}_{run_id}")
        video_recorders[run_id].save_video(save_filename)
        video_recorders[run_id].cleanup()

    # ------------------------------------------------------------------------------
    save_results_to_csv(results, log_dir, global_timestamp, model_type, task, perturbations[0])
    print("Done!")