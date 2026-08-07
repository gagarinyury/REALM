import numpy as np


def extract_from_obs(obs: dict, robot_name='DROID', enable_depth=False,
                     gripper_qpos_idx=7, gripper_qpos_range=(0.0, 0.05)):
    """
    NOTE: patched -- `gripper_qpos_idx` / `gripper_qpos_range` were added so the gripper
    state can be normalised for whichever robot is actually loaded. Upstream hardcoded
    `proprio[7] / 0.05`, i.e. the finger travel of REALM's own DROID gripper in metres.
    On any other gripper that silently produces a wrong `observation/gripper_position`:
    with the stock franka_robotiq, proprio[7] is a knuckle *angle* in radians, so the
    policy would be told the gripper is nearly closed while it is wide open.
    The defaults preserve the original behaviour for REALM's own DROID.
    """
    # Fallback to zeros if external sensors are missing (e.g. during no_render)
    if 'external' in obs and 'external_sensor0' in obs['external']:
        base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
        base_depth = obs['external']['external_sensor0']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        # Dummy 128x128 image
        base_im = np.zeros((128, 128, 3), dtype=np.uint8)
        base_depth = np.zeros((128, 128), dtype=np.float32) if enable_depth else None

    if 'external' in obs and 'external_sensor1' in obs['external']:
        base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
        base_depth_second = obs['external']['external_sensor1']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        base_im_second = None
        base_depth_second = None

    # Handle wrist camera. NOTE: patched -- upstream hardcoded
    # 'DROID:gripper_link_camera:Camera:0' and, when that key was absent, silently fell back
    # to a black image. The fallback is right for no_render (the robot carries no cameras at
    # all), but it also masks a misconfigured robot: the policy is then fed 128x128 of black
    # for every step of every rollout while the benchmark reports a clean 0.0 success rate
    # with nothing in the logs. We instead resolve the wrist camera from whatever the robot
    # actually exposes, which also keeps this working across robot models -- the link is
    # `gripper_link_camera` on REALM's own DROID asset but `camera_link` on the stock
    # franka_robotiq model.
    robot_cam_keys = [k for k in obs.get(robot_name, {}) if ':Camera:' in k]
    preferred = [
        f'{robot_name}:gripper_link_camera:Camera:0',
        f'{robot_name}:camera_link:Camera:0',
    ]
    wrist_cam_key = next((k for k in preferred if k in robot_cam_keys), None)
    if wrist_cam_key is None and robot_cam_keys:
        wrist_cam_key = robot_cam_keys[0]
        if len(robot_cam_keys) > 1:
            print(f"[REALM] WARNING: no known wrist-camera key on robot '{robot_name}'; "
                  f"using '{wrist_cam_key}' out of {robot_cam_keys}.")

    if wrist_cam_key is not None:
        wrist_im = obs[robot_name][wrist_cam_key]['rgb'].cpu().numpy()[..., :3]
    else:
        # No cameras on the robot at all -- the legitimate no_render case.
        wrist_im = np.zeros((128, 128, 3), dtype=np.uint8)

    # Proprio is always present in DROID and other robots
    proprio = obs[robot_name]['proprio'].cpu().numpy()
    robot_state = proprio[:7]
    # NOTE: patched -- normalised to [0, 1] against the actual joint's range instead of the
    # hardcoded 0.05 m finger travel of REALM's DROID gripper (see docstring). pi0-FAST
    # consumes this as `observation/gripper_position`, where 0 = open and 1 = closed.
    _lo, _hi = gripper_qpos_range
    gripper_state = float(np.clip((proprio[gripper_qpos_idx] - _lo) / (_hi - _lo), 0.0, 1.0))

    return base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state
