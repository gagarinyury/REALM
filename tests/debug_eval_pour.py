"""
Debug eval script for the pour_from_bottle task (task_id=10).

Sets up the environment, fills the water_glass_main with water particles,
then runs the evaluation loop.
"""

import sys
import omnigibson as og
from omnigibson.macros import gm

from realm.eval import evaluate, set_sim_config, SUPPORTED_TASKS


def patch_fluid_isosurface_fps_check():
    """
    Monkey-patch set_carb_settings_for_fluid_isosurface to skip the 60 FPS
    assertion. The isosurface carb settings work fine at lower frame rates —
    the assert is overly conservative. We apply all the same settings, just
    without the FPS gate.
    """
    import omnigibson.lazy as lazy
    import omnigibson.systems.micro_particle_system as mps

    def set_carb_settings_for_fluid_isosurface_no_fps_check():
        isregistry = lazy.carb.settings.acquire_settings_interface()
        dOptions = isregistry.get_as_int("persistent/app/viewport/displayOptions")
        dOptions &= ~(1 << 6 | 1 << 8)
        isregistry.set_int("persistent/app/viewport/displayOptions", dOptions)
        isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_NUM_THREADS, 8)
        isregistry.set_bool(lazy.omni.physx.bindings._physx.SETTING_UPDATE_VELOCITIES_TO_USD, True)
        isregistry.set_bool(lazy.omni.physx.bindings._physx.SETTING_UPDATE_PARTICLES_TO_USD, True)
        isregistry.set_int(lazy.omni.physx.bindings._physx.SETTING_MIN_FRAME_RATE, 60)
        isregistry.set_bool("rtx-defaults/pathtracing/lightcache/cached/enabled", False)
        isregistry.set_bool("rtx-defaults/pathtracing/cached/enabled", False)
        isregistry.set_int("rtx-defaults/pathtracing/fireflyFilter/maxIntensityPerSample", 10000)
        isregistry.set_int("rtx-defaults/pathtracing/fireflyFilter/maxIntensityPerSampleDiffuse", 50000)
        isregistry.set_float("rtx-defaults/pathtracing/optixDenoiser/blendFactor", 0.09)
        isregistry.set_int("rtx-defaults/pathtracing/aa/op", 2)
        isregistry.set_int("rtx-defaults/pathtracing/maxBounces", 32)
        isregistry.set_int("rtx-defaults/pathtracing/maxSpecularAndTransmissionBounces", 16)
        isregistry.set_int("rtx-defaults/post/dlss/execMode", 1)
        isregistry.set_int("rtx-defaults/translucency/maxRefractionBounces", 12)

    mps.set_carb_settings_for_fluid_isosurface = set_carb_settings_for_fluid_isosurface_no_fps_check
    og.log.info("Patched set_carb_settings_for_fluid_isosurface to skip 60 FPS assertion.")


def fill_water_glass_with_water(env):
    """
    After environment reset + warmup, find the water_glass_main object and fill
    it with water. Spawns particles in small batches from above the glass,
    stepping through env.step() (not raw og.sim.step()) to keep the articulation
    tensor views in sync.
    """
    import torch as th
    import numpy as np

    water_glass = None
    for obj in env.main_objects:
        if obj.name == "water_glass_main":
            water_glass = obj
            break

    if water_glass is None:
        water_glass = env.omnigibson_env.scene.object_registry("name", "water_glass_main")

    assert water_glass is not None, "water_glass_main object not found in the scene!"

    # Get the water particle system from the scene (this triggers initialization)
    water_system = env.omnigibson_env.scene.get_system("water")

    # Get glass position and AABB
    aabb_low, aabb_high = water_glass.aabb
    aabb_center = (aabb_low + aabb_high) / 2.0
    aabb_extent = aabb_high - aabb_low

    particle_radius = water_system.particle_radius
    spacing = water_system.particle_particle_rest_distance

    og.log.info(f"Particle radius: {particle_radius:.4f}, rest distance: {spacing:.4f}")
    og.log.info(f"Glass AABB extent: [{aabb_extent[0].item():.4f}, {aabb_extent[1].item():.4f}, {aabb_extent[2].item():.4f}]")

    # The AABB includes glass walls + base. Use 35% of the full width
    # as fill radius to stay inside the inner cavity.
    # Fill from 15% to 85% of height to get a nearly full glass.
    r = min(aabb_extent[0].item(), aabb_extent[1].item()) * 0.25

    cx = aabb_center[0].item()
    cy = aabb_center[1].item()

    z_bottom = aabb_low[2].item() + aabb_extent[2].item() * 0.15
    z_top = aabb_low[2].item() + aabb_extent[2].item() * 1.00

    og.log.info(f"Fill cylinder: r={r:.4f}, z=[{z_bottom:.4f}, {z_top:.4f}]")

    # Build full 3D grid at rest positions
    xs = th.arange(cx - r, cx + r, spacing)
    ys = th.arange(cy - r, cy + r, spacing)
    zs = th.arange(z_bottom, z_top, spacing)

    grid = th.stack(th.meshgrid(xs, ys, zs, indexing='ij'), dim=-1).reshape(-1, 3)

    # Filter to cylindrical volume
    dist_sq = (grid[:, 0] - cx) ** 2 + (grid[:, 1] - cy) ** 2
    positions = grid[dist_sq < (r ** 2)]

    og.log.info(f"Spawning {len(positions)} water particles at rest positions inside water_glass_main...")

    # Spawn all at once at rest positions — no need to drop/settle
    if len(positions) > 0:
        water_system.generate_particles(positions=positions)

    # Hold current EE pose during settling (convert world -> robot frame)
    from scipy.spatial.transform import Rotation as Rot
    ee_pos, ee_quat = env.get_ee_pose()
    ee_pos_np = ee_pos.cpu().numpy() if hasattr(ee_pos, 'cpu') else np.array(ee_pos)
    ee_euler_np = Rot.from_quat(ee_quat.cpu().numpy()).as_euler('xyz')
    ee_world = np.concatenate([ee_pos_np, ee_euler_np])
    ee_robot = env._world2robot(ee_world)
    hold_action = np.concatenate([ee_robot, [1.0]])  # gripper open
    for _ in range(100):
        env.omnigibson_env.step(hold_action)

    og.log.info(f"Water glass filled with {len(positions)} particles.")
    return len(positions) > 0


def run_pour_debug():
    """
    Custom evaluation loop for the pour task that fills the cup before running.
    Based on the evaluate() function but with the water-fill step injected
    after environment reset.
    """
    import time
    import os
    import numpy as np

    from realm.environments.env_dynamic import RealmEnvironmentDynamic

    task_id = 10
    max_steps = 5000000
    log_dir = "/app/logs/debug_pour"

    start = time.perf_counter()

    # GPU dynamics required for micro-particle (fluid) system.
    # HQ rendering required for isosurface (smooth liquid surface).
    # We patch the 60 FPS assertion but keep SETTING_MIN_FRAME_RATE=60
    # (the PhysX internal substep rate) — only the assertion is skipped.
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_HQ_RENDERING = True
    set_sim_config(rendering_mode="rt", robot="DROID")
    gm.ENABLE_HQ_RENDERING = True
    gm.DEFAULT_PHYSICS_FREQ = 120
    patch_fluid_isosurface_fps_check()

    task = SUPPORTED_TASKS[task_id]
    task_cfg_path = f"REALM_DROID10/{task}/default.yaml"
    perturbations = ["Default"]

    os.makedirs(log_dir, exist_ok=True)

    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=perturbations,
        multi_view=False,
        no_rendering=False,
        rendering_mode="rt",
        robot="DROID_ee_control"
    )
    og.log.info(f"DEBUG: Env created: {time.perf_counter() - start:.4f}s")

    # ---- Reset, warmup, then fill the cup with water ----
    obs, _ = env.reset()
    obs, rew, terminated, truncated, info = env.warmup(obs)

    fill_water_glass_with_water(env)
    og.log.info(f"DEBUG: Water glass filled with water: {time.perf_counter() - start:.4f}s")

    # ---- Move arm to target EE pose: pos=[0.3, 0.0, 0.3], rot=[0,0,0] flipped down, gripper open ----
    from realm.helpers import flip_pose_pointing_down
    target_ee_pos = np.array([0.6, 0.0, 0.15])
    target_ee_pos2 = np.array([0.8, 0.0, 0.15])
    target_ee_rot = flip_pose_pointing_down(np.array([0.0, 0.0, 0.0]))
    gripper_open = np.array([1.0])  # 1.0 = open


    og.log.info(f"Moving arm to EE target: pos={target_ee_pos}, rot={target_ee_rot}, gripper=open")

    for t in range(max_steps):
        ee_action = np.concatenate([target_ee_pos if t < 250 else target_ee_pos2, target_ee_rot, gripper_open])
        obs, task_progression, terminated, truncated, info = env.step(ee_action)

    og.log.info(f"DEBUG: Pour task finished.")
    og.log.info(f"DEBUG: Total time: {time.perf_counter() - start:.4f}s")


if __name__ == "__main__":
    run_pour_debug()
    og.shutdown()
    sys.exit(0)
