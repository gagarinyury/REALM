from queue import Queue
import datetime
import os
import random
import csv
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

import omnigibson as og
from omnigibson.macros import gm

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.inference import InferenceClient, extract_from_obs
from realm.realm_logging import VideoRecorder, save_results, append_trajectory, append_video
from realm.helpers import apply_blur_and_contrast



SUPPORTED_TASKS = [
    "put_green_block_into_bowl", #0
    "put_banana_into_box", #1
    "rotate_marker", #2
    "rotate_mug", #3
    "pick_spoon", #4
    "pick_water_bottle", #5
    "stack_cubes", #6
    "push_switch", #7
    "open_drawer", #8
    "close_drawer", #9
    "pour_from_bottle" #10
]

SUPPORTED_PERTURBATIONS = [
    'Default', #0
    'V-AUG', # 1
    'V-VIEW',  # 2
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
    'VB-POSE',  # 13
    'VB-MOBJ',  # 14
    'VSB-NOBJ' # 15
]


def _set_gm(name, value):
    """gm.* macros are write-once-per-process: trying to overwrite a macro
    after it's been read raises. Helper that no-ops when the value already
    matches, and bypasses the lock via dict access otherwise."""
    if gm.get(name) == value:
        return
    gm[name] = value


def set_sim_config(rendering_mode=None, robot="DROID", og_lite=False):
    if robot == "WidowX": # TODO: just read this from the yamls...
        _set_gm("DEFAULT_SIM_STEP_FREQ", 5)
        _set_gm("DEFAULT_RENDERING_FREQ", 5)
    elif "UR5" in robot:
        _set_gm("DEFAULT_SIM_STEP_FREQ", 30)
        _set_gm("DEFAULT_RENDERING_FREQ", 30)
    else:
        _set_gm("DEFAULT_SIM_STEP_FREQ", 15)
        _set_gm("DEFAULT_RENDERING_FREQ", 15)

    _set_gm("DEFAULT_PHYSICS_FREQ", 120)
    _set_gm("ENABLE_TRANSITION_RULES", False)  # off to avoid sludge-state bug in BEHAVIOR-1K #1201
    _set_gm("ENABLE_OBJECT_STATES", True)  # push_switch needs ToggledOn
    _set_gm("RENDER_VIEWER_CAMERA", False)
    _set_gm("ENABLE_HQ_RENDERING", False if rendering_mode == "r" else True)

    if og_lite:
        # Physics-only stepping; render only on explicit render_obs() calls
        _set_gm("ENABLE_VISUAL_UPDATES", False)
        _set_gm("OBJECT_STATE_UPDATE_WHITELIST", ["ToggledOn"])
        _set_gm("RENDER_ON_STEP", False)

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
        host="127.0.0.1",
        log_dir="/app/logs",
        resume=False,
        multi_view=False,
        no_record=False,
        no_render=False,
        rendering_mode=None,
        spp=8,
        task_cfg_path=None,
        robot="DROID",
        og_lite=False,
        n_pre_obs_renders=3,
        max_render_interval=8,
        randomize_scene=False,
):
    if rendering_mode is None:
        rendering_mode = "rt"
    set_sim_config(rendering_mode=rendering_mode, robot=robot, og_lite=og_lite)

    # -------------------- Create the environment + client --------------------
    if task_cfg_path is None:
        task = SUPPORTED_TASKS[task_id]
        task_cfg_path = f"REALM_DROID10/{task}/default.yaml"
    else:
        task = task_cfg_path.split("/")[-2]
        config_name = task_cfg_path.split("/")[-1].replace(".yaml", "").replace(".cfg", "")
        if config_name != "default":
            task = f"{task}_{config_name}"

    perturbations = [SUPPORTED_PERTURBATIONS[perturbation_id]]

    os.makedirs(log_dir, exist_ok=True)

    model_type = model_type # TODO: infer type from model name, rn this will just default to a pi model inference inside the client
    client = InferenceClient(model_type, host=host, port=port)

    # Read the task YAML once to know which (scene_model, scene_part) pairs we
    # can sample from if --randomize_scene is enabled.
    import yaml as _yaml
    with open(f"/app/realm/config/tasks/{task_cfg_path}", "r") as _f:
        _task_yaml = _yaml.load(_f, Loader=_yaml.FullLoader)
    supported_scenes = _task_yaml.get("supported_scenes", {})
    scene_pairs = [(sm, sp) for sm, parts in supported_scenes.items() for sp in parts]
    if randomize_scene and not scene_pairs:
        og.log.warning("--randomize_scene was set but the task YAML has no supported_scenes; ignoring.")
        randomize_scene = False

    def _build_env(scene_model=None, scene_part=None):
        return RealmEnvironmentDynamic(
            config_path="/app/realm/config",
            task_cfg_path=task_cfg_path,
            perturbations=perturbations,
            multi_view=multi_view,
            no_rendering=no_render,
            rendering_mode=rendering_mode,
            spp=spp,
            robot=robot,
            scene_model=scene_model,
            scene_part=scene_part,
        )

    env = _build_env()

    results = []
    start_repeat = 0
    results_filename = None

    if resume:
        potential_csv = os.path.join(log_dir, "reports", f"{task}_{perturbations[0]}.csv")
        if os.path.exists(potential_csv):
            results_filename = potential_csv
            with open(results_filename, 'r') as f:
                reader = csv.DictReader(f)
                existing_results = list(reader)
            results = existing_results
            start_repeat = len(results)
            og.log.info(f"Resuming run from repeat {start_repeat}. Using file: {results_filename}")
        else:
            og.log.info(f"Resume requested but no report found. Starting fresh.")

    for run_id in range(repeats):
        # ------------------------ pre-configure each run --------------------------------
        # seed = 1234 + run_id
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        if run_id < start_repeat:
            continue

        # Optionally tear down and rebuild the env on a new (scene_model,
        # scene_part) before this repeat. og.clear() purges the current
        # scene and relaunches the simulator instance so the new env
        # starts from a clean slate.
        if randomize_scene and scene_pairs:
            scene_model, scene_part = scene_pairs[np.random.randint(len(scene_pairs))]
            og.log.info(f"[--randomize_scene] repeat {run_id}: rebuilding env on ({scene_model}, {scene_part})")
            # og.clear() purges the scene and relaunches sim with the same
            # macro settings it was first configured with — do NOT call
            # set_sim_config here; gm.* macros are write-once and re-setting
            # them throws "Cannot set attribute … it has already been used".
            og.clear()
            env = _build_env(scene_model=scene_model, scene_part=scene_part)

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video_recorder = VideoRecorder(log_dir, timestamp, run_id, task, perturbations[0])

        qpos = []
        actions = []
        action_buffer = Queue()

        # -------------------- Rollout loop --------------------
        obs, _ = env.reset()

        # Warmup must render every step so the renderer is in sync with physics
        # after the robot settles.  In og_lite mode RENDER_ON_STEP is False, so
        # we temporarily re-enable it for the duration of warmup.
        if og_lite:
            og.sim._render_on_step = True
        obs, rew, terminated, truncated, info = env.warmup(obs)
        if og_lite:
            og.sim._render_on_step = False

        t = 0
        task_progression = 0.0
        task_progression_timestamps = []
        binary_success = False
        terminal_steps = 15

        ee_poses = []
        collisions_self = 0
        collisions_env = 0
        is_self_col_active = False
        is_env_col_active = False
        drops = 0
        was_grasping = False
        steps_since_render = 0

        while t < max_steps and terminal_steps > 0:
            need_new_chunk = action_buffer.empty()

            # On-demand rendering (og_lite): refresh obs at chunk boundaries, or
            # as a stability fallback when max_render_interval steps have elapsed
            # without a render (prevents the renderer from drifting too far from
            # physics state, which can trigger segfaults).
            need_render = need_new_chunk or (og_lite and steps_since_render >= max_render_interval)
            if og_lite and need_render:
                # Flush IsaacSim's rendering pipeline before capturing the
                # observation.  After N blind physics steps the scene state has
                # changed; extra render() calls propagate those changes through
                # the renderer before we read the sensors.
                for _ in range(n_pre_obs_renders):
                    og.sim.render()
                obs, _ = env.omnigibson_env.render_obs()
                steps_since_render = 0
                if "V-AUG" in env.active_perturbations:
                    obs = apply_blur_and_contrast(obs, env.v_aug_sigma, env.v_aug_alpha)

            # Image + robot-state extraction: only needed at chunk boundaries
            # (where inference will run). Fallback renders refresh obs but images
            # are not needed when inference is not running.
            if not og_lite or need_new_chunk:
                base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state = extract_from_obs(obs, robot_name=env.robot.name)

            # Physics-based metrics: run every step regardless of rendering mode.
            # check_collisions() and check_grasp_condition() query physics contacts,
            # not camera data, so stale obs is fine in og_lite mode.
            is_self_col, is_env_col = env.check_collisions()
            if is_self_col and not is_self_col_active:
                collisions_self += 1
            is_self_col_active = is_self_col

            if is_env_col and not is_env_col_active:
                collisions_env += 1
            is_env_col_active = is_env_col

            is_grasping = env.check_grasp_condition(obs)
            if was_grasping and not is_grasping:
                is_placed = False
                if hasattr(env, "task_type") and env.task_type in ["put", "stack"] and len(env.target_objects) > 0:
                    mo = env.main_objects[0]
                    target = env.target_objects[0]
                    inside = mo.states[og.object_states.Inside].get_value(target)
                    on_top = mo.states[og.object_states.OnTop].get_value(target)
                    if inside or on_top:
                        is_placed = True
                elif hasattr(env, "task_type") and env.task_type == "pour_liquid":
                    # Releasing the source after a successful pour is goal-aligned,
                    # not a drop. Use the pour success check as the "placed" signal.
                    if env.check_pour(obs):
                        is_placed = True
                elif hasattr(env, "task_type") and env.task_type == "pour_proxy":
                    # Same intent as pour_liquid: a goal-aligned release shouldn't
                    # count as a drop. Use the foam-ball-in-target check.
                    if env.check_pour_proxy(obs):
                        is_placed = True

                if not is_placed:
                    drops += 1
            was_grasping = is_grasping

            # Cartesian metrics: ee_pose is cheap (reads physics, no render needed)
            ee_pos, ee_rot = env.get_ee_pose()
            ee_poses.append(ee_pos)

            if need_new_chunk:
                # Compute robot-relative cartesian position for models that need it (e.g. DreamZero)
                _ee_pos = ee_pos.cpu().numpy() if hasattr(ee_pos, 'cpu') else np.array(ee_pos)
                _ee_rot = ee_rot.cpu().numpy() if hasattr(ee_rot, 'cpu') else np.array(ee_rot)
                _ee_euler = Rot.from_quat(_ee_rot).as_euler('xyz')
                _ee_pose_world = np.concatenate([_ee_pos, _ee_euler])
                cartesian_position = env._world2robot(_ee_pose_world).astype(np.float32)

                pred_action_chunk = client.infer(
                    env.instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
                    use_base_im_second=(env.task_type == "open_close_drawer" if hasattr(env, "task_type") else False),
                    ee_control=env.ee_control,
                    cartesian_position=cartesian_position
                )

                if len(pred_action_chunk.shape) == 2:
                    for action in pred_action_chunk[:horizon]:
                        action = np.squeeze(action)
                        action_buffer.put(action)
                elif len(pred_action_chunk.shape) < 2:
                    action_buffer.put(pred_action_chunk)
                else:
                    assert len(pred_action_chunk.shape) <= 2, f"Unsupported number of dimensions in action chunk with shape: {pred_action_chunk.shape}. The chunk is expected to be 2D."

            # Video: in og_lite mode record one frame per chunk (at render point)
            if not no_record and (not og_lite or need_new_chunk):
                video_recorder.add_frame(base_im, wrist_im, base_im_second)

            # qpos: in og_lite mode read fresh joint positions directly from physics
            if og_lite:
                fresh_proprio, _ = env.robot.get_proprioception()
                fresh_proprio_np = fresh_proprio.cpu().numpy()
                qpos.append(np.concatenate((fresh_proprio_np[:7], np.atleast_1d(np.array(fresh_proprio_np[7] / 0.05)))))
            else:
                qpos.append(np.concatenate((robot_state, np.atleast_1d(np.array(gripper_state)))))

            action = action_buffer.get()
            actions.append(action)

            new_action = action.copy()
            if model_type in ["debug", "openpi", "GR00T", "GR00T_N16", "dreamzero"]: # TODO: use a model config
                new_action[-1] = 1 if action[-1] > 0.5 else -1  # Prediction: (1,0) -> Target: (1,-1)
            elif model_type == "molmoact":
                new_action[-1] = 1 if action[-1] < 0.5 else -1  # Prediction: (0,1) -> Target: (1,-1)
            else:
                raise NotImplementedError()


            # new_gripper_state = 1 if action[-1] > 0.5 else -1  # Prediction: (1,0) -> Target: (1,-1)
            # new_gripper_state = np.atleast_1d(np.array(new_gripper_state))
            # new_action = np.concatenate((new_action, new_gripper_state))

            if og_lite:
                env.omnigibson_env.step_blind(new_action)
                # ToggledOn (push_switch) requires CAN_TOGGLE_STEPS=5 consecutive per-step updates
                # where the finger overlaps the button. step_blind skips _non_physics_step(), so
                # the counter would never accumulate across the 8-step blind chunk without this call.
                # ENABLE_VISUAL_UPDATES=False and OBJECT_STATE_UPDATE_WHITELIST=["ToggledOn"]
                # ensure this is cheap.
                og.sim._non_physics_step()
                # Task progression is computed from the rendered obs at the start of each chunk;
                # carry the last known value for blind intermediate steps.
                curr_task_progression = env.recompute_task_progression(obs) if need_new_chunk else task_progression
            else:
                obs, curr_task_progression, terminated, truncated, info = env.step(new_action)

            if curr_task_progression > task_progression:
                task_progression = curr_task_progression
                task_progression_timestamps.append(t)
            if env.check_final_success(obs):
                binary_success = True
            if task_progression >= 1.0 or binary_success:
                terminal_steps -= 1
            steps_since_render += 1
            t += 1

        # Metrics calculation
        dt = 1.0 / 15.0  # Control freq is 15Hz by default

        qpos_arr = np.stack(qpos)  # (N, 8)
        qpos_joints = qpos_arr[:, :7]

        # Joint space metrics
        if len(qpos_joints) > 4:
            joint_vel = np.diff(qpos_joints, axis=0) / dt
            joint_acc = np.diff(joint_vel, axis=0) / dt
            joint_jerk = np.diff(joint_acc, axis=0) / dt

            joint_vel_var = np.mean(np.var(joint_vel, axis=0) * len(joint_vel))
            joint_acc_var = np.mean(np.var(joint_acc, axis=0) * len(joint_acc))
            joint_jerk_metric = np.mean(np.linalg.norm(joint_jerk, axis=1))
            joint_path_length = np.sum(np.linalg.norm(np.diff(qpos_joints, axis=0), axis=1))
        else:
            joint_vel_var = 0.0
            joint_acc_var = 0.0
            joint_jerk_metric = 0.0
            joint_path_length = 0.0

        # Cartesian space metrics
        ee_pos_arr = np.stack(ee_poses)
        if len(ee_pos_arr) > 4:
            cart_vel = np.diff(ee_pos_arr, axis=0) / dt
            cart_acc = np.diff(cart_vel, axis=0) / dt
            cart_jerk = np.diff(cart_acc, axis=0) / dt

            cart_jerk_metric = np.mean(np.linalg.norm(cart_jerk, axis=1))
            cart_path_length = np.sum(np.linalg.norm(np.diff(ee_pos_arr, axis=0), axis=1))
        else:
            cart_path_length = 0.0
            cart_jerk_metric = 0.0

        stage_to_log = "SUCCESS"
        if env.task_progression is not None:
            for stage, is_completed in env.task_progression.items():
                if not is_completed:
                    stage_to_log = stage
                    break
            if binary_success:
                stage_to_log = "SUCCESS"
        else:
            stage_to_log = "N/A"

        if binary_success and hasattr(env, "task_type") and env.task_type in ["put", "stack"]:
            drops = max(0, drops - 1)

        result_entry = {
            "run_id": run_id,
            "task": task,
            "perturbation": perturbations[0],
            "instruction": env.instruction,
            "model": model_type,
            "real2sim": "Simulated",
            "env": "REALM",
            "task_progression": task_progression,
            "task_progression_timestamps": task_progression_timestamps,
            "stage": stage_to_log,
            "binary_SR": 1.0 if binary_success else 0.0,
            "joint_vel_var": joint_vel_var,
            "joint_acc_var": joint_acc_var,
            "joint_jerk": joint_jerk_metric,
            "joint_path_length": joint_path_length,
            "cart_path_length": cart_path_length,
            "cart_jerk": cart_jerk_metric,
            "collisions_self": collisions_self,
            "collisions_env": collisions_env,
            "object_drops": drops
        }

        result_entry["qpos"] = np.stack(qpos).tolist()
        result_entry["actions"] = np.stack(actions).tolist()
        if not no_record:
            video_bytes = video_recorder.get_video_bytes()
            result_entry["video"] = video_bytes
        
        results.append(result_entry)

        if not no_record:
            append_video(log_dir, task, perturbations[0], run_id, video_bytes)

        append_trajectory(log_dir, task, perturbations[0], run_id, np.stack(qpos), np.stack(actions))

        if not no_record:
            video_recorder.cleanup()

        client.reset()

        results_filename = save_results(results, log_dir + "/reports", task, perturbations[0], filename=results_filename)

    save_results(results, log_dir+"/reports", task, perturbations[0])
    og.log.info("Done!")

