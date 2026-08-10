"""Механический стенд для гриппера: подать команду руками и посмотреть, что поехало.

Зачем. Вопрос «смыкается ли гриппер по команде сжать» неделю выяснялся двадцатиминутными
роллаутами с моделью, хотя он не требует ни модели, ни задачи, ни восьмисот шагов. Здесь
рука фиксируется в текущей позе, а на канал гриппера подаётся сначала «раскрыть», потом
«сжать», и печатаются реальные позы суставов. Три минуты вместо двадцати.

Что проверяет:
  * едет ли ВЕДУЩИЙ сустав (outer_knuckle) и в какую сторону;
  * следуют ли за ним ВЕДОМЫЕ (inner_knuckle, inner_finger_knuckle) через PhysxMimicJointAPI;
  * сходятся ли физически губки — по расстоянию между пальцевыми линками.

Чем это ловится, если не смотреть. Ход суставов Robotiq 2F-85 задан как 0..45 градусов, где
НОЛЬ — раскрытое положение. Движок же по умолчанию (`grasping_direction="lower"`) считает
закрытым нижний предел. Если параметр не поправлен, команда «сжать» гонит сустав в ноль,
то есть раскрывает его; сустав приезжает туда, где уже стоял, поза не меняется вовсе,
наблюдение модели застывает — и политика начинает повторять одну и ту же команду.
Стенд показывает это первой же строкой таблицы.

Запуск:
    OMNIGIBSON_HEADLESS=1 python tests/gripper_bench.py [--robot DROID] [--steps 40]
"""

import argparse

import torch as th

import omnigibson as og
from omnigibson.macros import gm


def finger_gap(robot):
    """Расстояние между пальцевыми линками, м. Прямая мера «сомкнулись или нет»."""
    links = [l for name, l in robot.links.items() if "inner_finger" in name and "knuckle" not in name]
    if len(links) < 2:
        return float("nan")
    a, b = links[0].get_position_orientation()[0], links[1].get_position_orientation()[0]
    return float(th.norm(a - b))


def gripper_joint_positions(robot):
    """Позы всех суставов гриппера, включая ведомые (их нет в логах роллаута)."""
    names = [n for n in robot.joints if "knuckle" in n or "finger" in n]
    qpos = robot.get_joint_positions()
    idx = {n: i for i, n in enumerate(robot.joints)}
    return {n: float(qpos[idx[n]]) for n in names}


def drive(env, robot, command, steps, label):
    """Держит руку на месте, подаёт на гриппер @command и печатает, что произошло."""
    action = th.zeros(robot.action_dim)
    action[:7] = robot.get_joint_positions()[:7]  # рука неподвижна
    action[-1] = command

    before = gripper_joint_positions(robot)
    gap_before = finger_gap(robot)
    for _ in range(steps):
        env.step(action)
    after = gripper_joint_positions(robot)
    gap_after = finger_gap(robot)

    print(f"\n=== команда {command:+.0f} ({label}), {steps} шагов ===")
    print(f"{'сустав':<40}{'было':>10}{'стало':>10}{'ход':>10}")
    for n in sorted(after):
        d = after[n] - before[n]
        mark = "  <-- не сдвинулся" if abs(d) < 1e-4 else ""
        print(f"{n:<40}{before[n]:>10.4f}{after[n]:>10.4f}{d:>+10.4f}{mark}")
    print(f"{'зазор между губками, м':<40}{gap_before:>10.4f}{gap_after:>10.4f}{gap_after - gap_before:>+10.4f}")
    return gap_after - gap_before


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="DROID")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--task_id", type=int, default=0)
    args = ap.parse_args()

    from realm.environments.env_dynamic import DynamicEnv  # noqa: E402

    with gm.unlocked():
        gm.ENABLE_HQ_RENDERING = False

    env = DynamicEnv(task_id=args.task_id, perturbation_id=0, robot=args.robot).env
    robot = env.robots[0]
    env.reset()

    print(f"робот: {robot.name}, суставов: {len(robot.joints)}, action_dim: {robot.action_dim}")
    print("суставы гриппера:", sorted(gripper_joint_positions(robot)))

    opened = drive(env, robot, +1, args.steps, "раскрыть")
    closed = drive(env, robot, -1, args.steps, "сжать")

    print("\n=== ВЕРДИКТ ===")
    if closed < -1e-3:
        print("губки сходятся по команде «сжать» — направление захвата задано верно")
    elif opened < -1e-3:
        print("губки сходятся по команде «РАСКРЫТЬ» — направление ИНВЕРТИРОВАНО.")
        print("Лечится строкой `grasping_direction: \"upper\"` в конфиге робота")
        print("(или closed_qpos/open_qpos в controller_config.gripper_0).")
    else:
        print("губки не двигаются ни на одну команду — смотреть привод, mimic и лимиты скорости")

    og.shutdown()
