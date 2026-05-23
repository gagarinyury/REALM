"""
Debug eval script for the pour_from_bottle task (task_id=10).

The pour task is wired through the standard env_dynamic plumbing now: env.warmup()
fills the source object with water, env.check_pour() reads particles in the
target. This script exists to exercise the task with a scripted EE trajectory
and to quickly check that perturbations (VB-POSE, S-LANG, ...) compose with the
fluid setup.

Usage:
    python tests/debug_eval_pour.py [--perturbation Default|VB-POSE|...] [--no-water]
"""

import argparse
import sys
import time
import os
import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from scipy.spatial.transform import Rotation as R

from realm.eval import set_sim_config, SUPPORTED_TASKS
from realm.environments.env_dynamic import RealmEnvironmentDynamic


def run_pour_debug(perturbation: str = "Default", max_steps: int = 1500, log_dir: str = "/app/logs/debug_pour", spawn_water: bool = True):
    start = time.perf_counter()

    # The pour task itself toggles GPU dynamics + HQ rendering and patches the
    # fluid-isosurface FPS check from inside env_dynamic. We just need to set the
    # baseline sim config here.
    set_sim_config(rendering_mode="rt", robot="DROID")

    task_id = SUPPORTED_TASKS.index("pour_from_bottle")
    task_cfg_path = f"REALM_DROID10/{SUPPORTED_TASKS[task_id]}/default.yaml"

    os.makedirs(log_dir, exist_ok=True)

    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=[perturbation],
        multi_view=False,
        no_rendering=False,
        rendering_mode="rt",
        robot="DROID_ee_control",
    )
    print(f"DEBUG: Env created in {time.perf_counter() - start:.2f}s")

    # Suppress water spawning for fast debugging of the arm trajectory. Leaves
    # env.water_system = None so the env still works without fluid particles.
    if not spawn_water:
        env._fill_pour_source = lambda *a, **kw: None
        print("DEBUG: water spawning DISABLED (--no-water)")

    # Make the spawned water nearly massless. particle_density is read at spawn
    # time inside _fill_pour_source -> generate_particles, so it must be
    # overridden BEFORE warmup runs.
    if spawn_water:
        water_system = env.omnigibson_env.scene.get_system("water")
        orig_density = water_system._particle_density
        water_system._particle_density = 1e-3
        print(f"DEBUG: water particle_density {orig_density} -> {water_system._particle_density}")

    obs, _ = env.reset()
    obs, rew, terminated, truncated, info = env.warmup(obs)
    print(f"DEBUG: Warmup + water fill done in {time.perf_counter() - start:.2f}s")

    # Set bottle mass to 330 g (heavier than the env_dynamic default of 100 g
    # for debug-stability — makes the bottle less twitchy under arm contact)
    # and drop CoM near the bottle's bottom (matches env_dynamic for pour_proxy,
    # but applied here too in case this debug script is run with a non-pour_proxy
    # task).
    target_mass_kg = 0.1
    for obj in env.main_objects:
        if "cola_bottle" in obj.name or obj.category == "cola_bottle":
            total_mass = sum(link.mass for link in obj._links.values())
            scale = target_mass_kg / total_mass
            for link in obj._links.values():
                link.mass = link.mass * scale
            bottle_mass = sum(link.mass for link in obj._links.values())
            print(f"DEBUG: Set {obj.name} mass {total_mass:.4f} -> {bottle_mass:.4f} kg")

            # Push CoM toward the bottle's bottom (85% of half-height below
            # the origin, expressed in body-local frame via the bottle's
            # current world orientation so the math works regardless of how
            # the link's xform is authored).
            try:
                _, bottle_quat_w = obj.get_position_orientation()
                if hasattr(bottle_quat_w, "cpu"):
                    bottle_quat_w = bottle_quat_w.cpu().numpy()
                bottle_quat_w = np.asarray(bottle_quat_w, dtype=float)
                aabb_min, aabb_max = obj.aabb
                world_z_extent = float(
                    (aabb_max[2] - aabb_min[2]).item()
                    if hasattr(aabb_max[2], "item")
                    else aabb_max[2] - aabb_min[2]
                )
                com_drop_world = np.array([0.0, 0.0, -0.85 * (world_z_extent / 2.0)])
                com_offset_body = R.from_quat(bottle_quat_w).inv().apply(com_drop_world)
                link_prim = obj.root_link.prim
                mass_api = lazy.pxr.UsdPhysics.MassAPI.Apply(link_prim)
                if not mass_api.GetCenterOfMassAttr():
                    mass_api.CreateCenterOfMassAttr()
                mass_api.GetCenterOfMassAttr().Set(
                    lazy.pxr.Gf.Vec3f(
                        float(com_offset_body[0]),
                        float(com_offset_body[1]),
                        float(com_offset_body[2]),
                    )
                )
                print(f"DEBUG: set {obj.name} CoM to body-local {tuple(com_offset_body.round(4).tolist())} (= world (0,0,{com_drop_world[2]:.4f}))")
            except Exception as e:
                print(f"DEBUG: could not set CoM on {obj.name}: {e}")

    # NOTE: foam_ball mass is set inside env_dynamic for pour_proxy (0.5 g
    # per ball), so no per-ball mass override is needed here.

    # TEMP DEBUG: bottle has now settled via warmup. Place each foam_ball
    # inside the bottle at its (now-settled) world position, with a small XY
    # jitter and Z spacing. Validate that the bottle stays upright after the
    # balls settle; if a ball bugs through a decomp seam and tips the bottle,
    # re-roll the jitter and retry — bad random placements should NEVER be
    # baked into the reset snapshot.
    if env.main_objects:
        foam_balls = [d for d in env.distractors if d is not None and "foam_ball" in d.name]
        if foam_balls:
            bottle = env.main_objects[0]
            ball_diameter = 0.01  # must match env_dynamic
            z_spacing = ball_diameter * 1.5
            xy_jitter = 0.002   # tightened to ±2 mm so balls stay near the bottle's center axis, away from wall-hull seams
            z_lift = 0.03
            max_attempts = 8

            # Capture the pristine post-warmup pose ONCE, before any retry can
            # tip the bottle. All subsequent attempts restore to THIS pose
            # (both position AND orientation), and ball positions are computed
            # from pristine_pos so a tipped bottle on a previous attempt
            # doesn't bias the next attempt's reference frame.
            pristine_pos, pristine_quat = bottle.get_position_orientation()
            if hasattr(pristine_pos, "cpu"):
                pristine_pos = pristine_pos.cpu().numpy()
            pristine_pos = np.asarray(pristine_pos, dtype=float)
            if hasattr(pristine_quat, "cpu"):
                pristine_quat = pristine_quat.cpu().numpy()
            pristine_quat = np.asarray(pristine_quat, dtype=float)

            for attempt in range(max_attempts):
                # Always restore bottle to pristine pose (pos AND quat) at
                # the start of each attempt.
                bottle.set_position_orientation(
                    position=pristine_pos.tolist(),
                    orientation=pristine_quat.tolist(),
                )
                bottle.keep_still()

                for i, ball in enumerate(foam_balls):
                    dx = np.random.uniform(-xy_jitter, xy_jitter)
                    dy = np.random.uniform(-xy_jitter, xy_jitter)
                    new_pos = [
                        float(pristine_pos[0]) + dx,
                        float(pristine_pos[1]) + dy,
                        float(pristine_pos[2]) + z_lift + i * z_spacing,
                    ]
                    ball.set_position_orientation(position=new_pos)
                    ball.keep_still()
                for _ in range(60):
                    og.sim.step()

                if env.is_source_upright():
                    print(f"DEBUG: balls placed successfully on attempt {attempt + 1}/{max_attempts}")
                    # Snap bottle exactly to pristine before snapshot so the
                    # captured initial_state has the canonical upright pose,
                    # not a borderline-drifted one.
                    bottle.set_position_orientation(
                        position=pristine_pos.tolist(),
                        orientation=pristine_quat.tolist(),
                    )
                    bottle.keep_still()
                    env.omnigibson_env.scene.update_initial_state()
                    print("DEBUG: snapshot captured with balls inside")
                    break
                print(f"DEBUG: bottle tipped on attempt {attempt + 1}/{max_attempts} — re-rolling jitter")
            else:
                print(f"DEBUG: FAILED to keep bottle upright after {max_attempts} attempts; snapshot NOT updated")

    # ---- Scripted trajectory: reach, grasp, lift, rotate the bottle. The point
    # of this script is to exercise the pipeline, not to solve the task.
    from realm.helpers import flip_pose_pointing_down
    from scipy.spatial.transform import Rotation
    reach_ee_pos = np.array([0.35, 0.0, 0.15])
    reach_ee_rot = flip_pose_pointing_down(np.array([3.14, 1.57, 0.0]))
    reach_gripper_open = np.array([-1.0])
    reach_ee_action = np.concatenate([reach_ee_pos, reach_ee_rot, reach_gripper_open])

    grasp_ee_pos = np.array([0.47, 0.0, 0.125])
    grasp_ee_rot = flip_pose_pointing_down(np.array([3.14, 1.57, 0.0]))
    grasp_gripper_open = np.array([-1.0])
    grasp_ee_action = np.concatenate([grasp_ee_pos, grasp_ee_rot, grasp_gripper_open])

    lift_ee_pos = np.array([0.47, 0.0, 0.35])
    lift_ee_rot = flip_pose_pointing_down(np.array([3.14, 1.57, 0.0]))
    lift_gripper_open = np.array([1.0])
    lift_ee_action = np.concatenate([lift_ee_pos, lift_ee_rot, lift_gripper_open])

    for t in range(max_steps):
        if t < 300:
            ee_action = reach_ee_action
        elif t < 350:
            ee_action = grasp_ee_action
        elif t < 400:
            ee_action = grasp_ee_action
            ee_action[-1] = 1
        else:
            ee_action = lift_ee_action
            if t > 450:
                base_rpy = flip_pose_pointing_down(np.array([3.14, 1.57, 0.0]))
                R_base = Rotation.from_euler('xyz', base_rpy)
                R_wrist_roll = Rotation.from_euler('z', np.deg2rad(130))
                ee_action[3:6] = (R_base * R_wrist_roll).as_euler('xyz')
        obs, task_progression, terminated, truncated, info = env.step(ee_action)
        print(t, task_progression)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbation", default="Default")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--no-water", action="store_true", help="Skip water spawning for fast arm debugging")
    args = parser.parse_args()
    run_pour_debug(perturbation=args.perturbation, max_steps=args.max_steps, spawn_water=not args.no_water)
    og.shutdown()
    sys.exit(0)
