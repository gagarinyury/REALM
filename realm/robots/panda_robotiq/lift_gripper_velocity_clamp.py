"""Снимает ограничение скорости с ведомых суставов гриппера.

ЭТО И БЫЛА ПРИЧИНА. В droid.usd у внутренних суставов стоит physxJoint:maxJointVelocity = 0.2,
у ведущих -- 120. Проверка арифметикой: 40 шагов при 15 Гц это 2.7 с, 0.2 град/с * 2.7 с =
0.0094 рад, а измерялось 0.0112 рад. Суставы не были заблокированы -- они честно отрабатывали
команду со скоростью, при которой полный ход в 45 градусов занимает почти четыре минуты.

У стокового Robotiq (models/ur5e/usd/ur5e.usda) те же суставы имеют maxJointVelocity = inf,
ограничение стоит только на ведущих (130). Ставим так же.

Ограничение было безобидным, пока суставы вёл констрейнт параллелограмма; как только ими стало
управлять напрямую, оно превратилось в стоп-кран.
"""

import sys

import omnigibson as og

og.launch()

from pxr import Sdf, Usd  # noqa: E402

USD = sys.argv[1] if len(sys.argv) > 1 else None
assert USD, "usage: python lift_gripper_velocity_clamp.py <path to droid.usd>"
# Все суставы гриппера: на mimic-конструкции лимит 0.2 стоит и на ведомых, и это ровно та
# причина, по которой mimic выглядел неработающим -- констрейнт вёл сустав, а лимит скорости
# не давал ему двигаться быстрее 0.2 град/с.
# ТОЛЬКО ВЕДОМЫЕ. Правка от 10.08.2026: прежний список включал и ведущие
# outer_knuckle, у которых в ассете стоит осмысленное ограничение 120 град/с --
# оно и задаёт нормальную скорость привода. Сняв его заодно с паразитным 0.2,
# мы получили гриппер, захлопывающийся за ОДИН шаг симуляции (измерено: 99.9%
# хода за шаг против 40.8% на авторском стенде, то есть ~680 град/с против ~275;
# настоящий Robotiq 2F-85 закрывается за полсекунды, то есть около 90 град/с).
# Следствие мгновенного срабатывания -- политика видит скачкообразную смену
# состояния и переключает команду каждые 8 шагов, губки клацают и не удерживают
# объект. Эталон тот же, что и был описан здесь изначально: у стокового ur5e
# ограничение стоит ТОЛЬКО на ведущих (130), у ведомых -- inf.
TARGETS = [
    "/panda/gripper_link_base/left_inner_knuckle_joint",
    "/panda/gripper_link_base/right_inner_knuckle_joint",
    "/panda/gripper_link_left_inner_knuckle/left_inner_finger_knuckle_joint",
    "/panda/gripper_link_right_inner_knuckle/right_inner_finger_knuckle_joint",
    "/panda/gripper_link_left_outer_knuckle/left_outer_finger_knuckle_joint",
    "/panda/gripper_link_right_outer_knuckle/right_outer_finger_knuckle_joint",
]

stage = Usd.Stage.Open(USD)
for path in TARGETS:
    prim = stage.GetPrimAtPath(path)
    if not (prim and prim.IsValid()):
        print(f"skip {path} (нет прима)", flush=True)
        continue
    attr = prim.GetAttribute("physxJoint:maxJointVelocity")
    before = attr.Get() if attr else None
    prim.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(float("inf"))
    print(f"{prim.GetName()}: maxJointVelocity {before} -> {prim.GetAttribute('physxJoint:maxJointVelocity').Get()}", flush=True)

stage.GetRootLayer().Save()
check = Usd.Stage.Open(USD)
for path in TARGETS:
    p = check.GetPrimAtPath(path)
    if p and p.IsValid():
        print(f"verify {p.GetName()}: {p.GetAttribute('physxJoint:maxJointVelocity').Get()}", flush=True)
print("=== MAXVEL COMPLETE ===", flush=True)
og.shutdown()
