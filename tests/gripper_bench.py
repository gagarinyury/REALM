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
import sys

import torch as th

_OUT = open("/tmp/gripper_bench_result.txt", "w", buffering=1)


def say(*a):
    """Isaac Sim перехватывает stdout, поэтому дублируем в файл."""
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    _OUT.write(line + "\n")

import omnigibson as og
from omnigibson.macros import gm


def finger_gap(robot):
    """Расстояние между пальцевыми линками, м.

    ОСТОРОЖНО: это расстояние между НАЧАЛАМИ КООРДИНАТ звеньев, а не между рабочими
    поверхностями губок. При вращении звенья могут расходиться, даже когда губки
    сближаются, поэтому по одному этому числу нельзя судить «сомкнулись или нет».
    Ниже считается и расстояние по геометрии коллизий — см. finger_surface_gap().
    """
    links = [l for name, l in robot.links.items() if "inner_finger" in name and "knuckle" not in name]
    if len(links) < 2:
        return float("nan")
    a, b = links[0].get_position_orientation()[0], links[1].get_position_orientation()[0]
    return float(th.norm(a - b))


def finger_surface_gap(robot):
    """Кратчайшее расстояние между коллизионными оболочками губок, м.

    Именно это отвечает на вопрос «сомкнулись ли губки»: берутся центры коллизионных
    мешей каждого пальца и меряется расстояние между ближайшей парой.
    """
    fingers = [l for name, l in robot.links.items() if "inner_finger" in name and "knuckle" not in name]
    if len(fingers) < 2:
        return float("nan")
    pts = []
    for link in fingers:
        try:
            meshes = list(link.collision_meshes.values())
            pts.append([m.get_position_orientation()[0] for m in meshes] or
                       [link.get_position_orientation()[0]])
        except Exception:
            pts.append([link.get_position_orientation()[0]])
    best = float("inf")
    for a in pts[0]:
        for b in pts[1]:
            best = min(best, float(th.norm(a - b)))
    return best


def pad_tilt(robot):
    """Наклон пальцевых колодок относительно основания гриппера, градусы.

    Смысл параллелограмма Robotiq 2F-85 в том, что колодка остаётся ПАРАЛЛЕЛЬНОЙ основанию
    на всём ходу — иначе губки берут предмет ребром, а не плоскостью. Если знак связи
    ведомого сустава задан неверно, колодка поворачивается вместе с коленом и набирает
    примерно двойной угол. Эталон знаков — panda_robotiq_85.urdf: follower (колодка)
    multiplier=-1, spring_link (колено) multiplier=+1; наш untangle_droid_gripper.py
    ставит одинаковый gearing всем четырём.
    """
    import omnigibson.utils.transform_utils as T

    base = [l for n, l in robot.links.items() if n.endswith("gripper_link_base")]
    pads = [l for n, l in robot.links.items() if "inner_finger" in n and "knuckle" not in n]
    if not base or len(pads) < 2:
        return {}
    _, bq = base[0].get_position_orientation()
    out = {}
    for pad in pads:
        _, q = pad.get_position_orientation()
        rel = T.quat2euler(T.quat_multiply(T.quat_inverse(bq), q))
        out[pad.name.split(":")[-1]] = float(th.rad2deg(th.abs(rel).max()))
    return out


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
    surf_before = finger_surface_gap(robot)
    tilt_before = pad_tilt(robot)
    for _ in range(steps):
        env.step(action)
    after = gripper_joint_positions(robot)
    gap_after = finger_gap(robot)
    surf_after = finger_surface_gap(robot)
    tilt_after = pad_tilt(robot)

    say(f"\n=== команда {command:+.0f} ({label}), {steps} шагов ===")
    say(f"{'сустав':<40}{'было':>10}{'стало':>10}{'ход':>10}")
    for n in sorted(after):
        d = after[n] - before[n]
        mark = "  <-- не сдвинулся" if abs(d) < 1e-4 else ""
        say(f"{n:<40}{before[n]:>10.4f}{after[n]:>10.4f}{d:>+10.4f}{mark}")
    say(f"{'зазор (начала звеньев), м':<40}{gap_before:>10.4f}{gap_after:>10.4f}{gap_after - gap_before:>+10.4f}")
    say(f"{'зазор (поверхности губок), м':<40}{surf_before:>10.4f}{surf_after:>10.4f}{surf_after - surf_before:>+10.4f}")
    say(f"наклон колодок к основанию, град: {tilt_before} -> {tilt_after}")
    # Условие метрики REALM (env_base.py:187): 0.45 - q > 1e-3 по proprio[7:9].
    # У авторов это призматика 0..0.05 м и условие тождественно истинно; после ремонта
    # ассета там углы 0..0.785 рад, и оно стало живым фильтром.
    proprio = robot.get_proprioception()[0]
    q78 = [float(proprio[7]), float(proprio[8])]
    cond = (0.45 - q78[0] > 1e-3) or (0.45 - q78[1] > 1e-3)
    say(f"метрика REALM: proprio[7:9] = {[round(x, 4) for x in q78]}  ->  is_either_finger_closing = {cond}")
    return surf_after - surf_before


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="DROID")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--task_id", type=int, default=0)
    args = ap.parse_args()

    import os

    from realm.environments.env_dynamic import RealmEnvironmentDynamic  # noqa: E402

    with gm.unlocked():
        gm.ENABLE_HQ_RENDERING = False

    from realm.eval import SUPPORTED_TASKS, set_sim_config  # noqa: E402

    set_sim_config(rendering_mode="r", robot=args.robot)

    realm_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    realm_env = RealmEnvironmentDynamic(
        config_path=os.path.join(realm_root, "realm", "config"),
        # тот же путь, что строит eval.py: REALM_DROID10/<task>/default.yaml
        task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[args.task_id]}/default.yaml",
        perturbations=[],
        multi_view=False,
        no_rendering=True,   # рендер стенду не нужен: смотрим только позы суставов
        rendering_mode="r",
        robot=args.robot,
    )
    env, robot = realm_env.omnigibson_env, realm_env.robot
    env.reset()

    say(f"робот: {robot.name}, суставов: {len(robot.joints)}, action_dim: {robot.action_dim}")
    say("суставы гриппера:", sorted(gripper_joint_positions(robot)))

    opened = drive(env, robot, +1, args.steps, "раскрыть")
    closed = drive(env, robot, -1, args.steps, "сжать")

    # --- ТЕСТ УДЕРЖАНИЯ: предмет между губками, три флага метрики GRASP ---
    # env_base.py:205 засчитывает GRASP только если ВСЕ ТРИ истинны одновременно.
    # Ни разу за отладку не проверялось, какой именно из них ложен.
    say("\n=== ТЕСТ УДЕРЖАНИЯ: предмет ставится между губками ===")
    obj = realm_env.main_objects[0]
    eef = robot.links[robot.eef_link_names[robot.default_arm]]

    drive(env, robot, +1, 10, "раскрыть перед постановкой")
    pos, orn = eef.get_position_orientation()
    obj.set_position_orientation(position=pos + th.tensor([0.0, 0.0, -0.09]), orientation=orn)
    for _ in range(10):
        env.step(th.cat([robot.get_joint_positions()[:7], th.tensor([1.0])]))

    say(f"предмет: {obj.name}, поставлен между губками")
    drive(env, robot, -1, args.steps, "сомкнуть на предмете")

    obs = env.get_obs()[0]
    proprio = robot.get_proprioception()[0]
    q78 = [float(proprio[7]), float(proprio[8])]
    from omnigibson.utils.usd_utils import RigidContactAPI  # noqa: E402

    touching = [
        bool(RigidContactAPI.is_in_contact(scene_idx=robot.scene.idx, query_set=[f],
                                           with_set=[obj], ignore_set=None, current_only=True))
        for f in realm_env.robot_finger_links
    ]
    cond_close = (0.45 - q78[0] > 1e-3) or (0.45 - q78[1] > 1e-3)
    cond_both = sum(touching) == 2
    cond_robot = bool(realm_env.is_touching(obs, obj))

    say("\nТРИ УСЛОВИЯ GRASP (env_base.py:205):")
    say(f"  обе колодки касаются предмета : {cond_both}   (касаний: {sum(touching)} из 2)")
    say(f"  робот касается предмета       : {cond_robot}")
    say(f"  0.45 - proprio[7:9] > 1e-3    : {cond_close}   (proprio[7:9] = {[round(x, 4) for x in q78]})")
    say(f"  ИТОГО GRASP засчитан          : {cond_both and cond_robot and cond_close}")
    say(f"  наклон колодок к основанию    : {pad_tilt(robot)}")

    say("\n=== УДЕРЖИВАЕТ ЛИ: поднимаем руку на 10 см ===")
    h0 = float(obj.get_position_orientation()[0][2])
    up = robot.get_joint_positions()[:7].clone()
    up[1] -= 0.25
    for _ in range(60):
        env.step(th.cat([up, th.tensor([-1.0])]))
    h1 = float(obj.get_position_orientation()[0][2])
    say(f"высота предмета: {h0:.4f} -> {h1:.4f} м  (изменение {h1 - h0:+.4f})")
    say("предмет УДЕРЖАН" if h1 - h0 > 0.02 else "предмет ВЫПАЛ или остался на месте")

    say("\n=== ВЕРДИКТ ===")
    if closed < -1e-3:
        say("губки сходятся по команде «сжать» — направление захвата задано верно")
    elif opened < -1e-3:
        say("губки сходятся по команде «РАСКРЫТЬ» — направление ИНВЕРТИРОВАНО.")
        say("Лечится строкой `grasping_direction: \"upper\"` в конфиге робота")
        say("(или closed_qpos/open_qpos в controller_config.gripper_0).")
    else:
        say("губки не двигаются ни на одну команду — смотреть привод, mimic и лимиты скорости")

    og.shutdown()
