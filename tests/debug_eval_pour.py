"""
Debug eval script for the pour_from_bottle task (task_id=10).

The pour task is wired through the standard env_dynamic plumbing now: env.warmup()
fills the source object with water, env.check_pour() reads particles in the
target. This script exists to exercise the task with a scripted EE trajectory
and to quickly check that perturbations (VB-POSE, S-LANG, ...) compose with the
fluid setup.

Usage:
    python tests/debug_eval_pour.py [--perturbation Default|VB-POSE|...]
"""

import argparse
import sys
import time
import os
import numpy as np

import omnigibson as og

from realm.eval import set_sim_config, SUPPORTED_TASKS
from realm.environments.env_dynamic import RealmEnvironmentDynamic


def run_pour_debug(perturbation: str = "Default", max_steps: int = 1500, log_dir: str = "/app/logs/debug_pour"):
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
    og.log.info(f"DEBUG: Env created in {time.perf_counter() - start:.2f}s")

    obs, _ = env.reset()
    obs, rew, terminated, truncated, info = env.warmup(obs)
    og.log.info(f"DEBUG: Warmup + water fill done in {time.perf_counter() - start:.2f}s")

    # ---- Scripted trajectory: move the EE in front of the source, then nudge
    # rightward as a placeholder for the actual pour motion. The point of this
    # script is to exercise the pipeline, not to solve the task.
    from realm.helpers import flip_pose_pointing_down
    target_ee_pos1 = np.array([0.6, 0.0, 0.15])
    target_ee_pos2 = np.array([0.8, 0.0, 0.15])
    target_ee_rot = flip_pose_pointing_down(np.array([0.0, 0.0, 0.0]))
    gripper_open = np.array([1.0])

    last_progression = 0.0
    last_pour_check = False
    for t in range(max_steps):
        ee_pos = target_ee_pos1 if t < 250 else target_ee_pos2
        ee_action = np.concatenate([ee_pos, target_ee_rot, gripper_open])
        obs, task_progression, terminated, truncated, info = env.step(ee_action)
        if task_progression != last_progression:
            og.log.info(f"  step {t}: progression -> {task_progression:.2f}")
            last_progression = task_progression
        cur_pour_check = env.check_pour(obs)
        if cur_pour_check and not last_pour_check:
            og.log.info(f"  step {t}: POUR success condition met")
        last_pour_check = cur_pour_check

    og.log.info(f"DEBUG: Final task progression={last_progression:.2f}, pour_success={last_pour_check}")
    og.log.info(f"DEBUG: Total time: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbation", default="Default")
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()
    run_pour_debug(perturbation=args.perturbation, max_steps=args.max_steps)
    og.shutdown()
    sys.exit(0)
