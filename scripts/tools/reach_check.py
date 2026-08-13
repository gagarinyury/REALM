"""Дотягивался ли робот до предмета: считаем траекторию схвата из записанных qpos.

Симулятор не нужен -- берём записанные углы суставов, прогоняем прямую кинематику Panda
(параметры из panda_arm.urdf), переводим в мировые координаты по позе робота из scenes.yaml
и сравниваем с позицией предмета, вычисленной из конфига задачи.

REACH засчитывается при расстоянии < 0.1 м от НАЧАЛА пальцевого звена до центра предмета
(env_base.py:246). Схват (panda_link8) -- не то же самое, что пальцевое звено, оно ниже
примерно на длину гриппера, поэтому расстояние по схвату -- оценка сверху: если даже
схват не подошёл, то и пальцы тем более.
"""
import sys

import numpy as np
import pandas as pd

_PANDA_JOINT_ORIGINS = [
    ([0,       0,      0.333], [0,        0, 0]),
    ([0,       0,      0    ], [-np.pi/2, 0, 0]),
    ([0,      -0.316,  0    ], [ np.pi/2, 0, 0]),
    ([0.0825,  0,      0    ], [ np.pi/2, 0, 0]),
    ([-0.0825, 0.384,  0    ], [-np.pi/2, 0, 0]),
    ([0,       0,      0    ], [ np.pi/2, 0, 0]),
    ([0.088,   0,      0    ], [ np.pi/2, 0, 0]),
]
_PANDA_EE_OFFSET = [0, 0, 0.107]
DROID_BASE_HEIGHT = 0.86244


def _rot3(a, axis):
    ca, sa = np.cos(a), np.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    if axis == 'y':
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])


def panda_fk(q):
    def _ht(xyz, rpy, qi):
        r = _rot3(rpy[2], 'z') @ _rot3(rpy[1], 'y') @ _rot3(rpy[0], 'x') @ _rot3(qi, 'z')
        m = np.eye(4); m[:3, :3] = r; m[:3, 3] = xyz
        return m
    m = np.eye(4)
    for (xyz, rpy), qi in zip(_PANDA_JOINT_ORIGINS, q):
        m = m @ _ht(xyz, rpy, qi)
    ee = np.eye(4); ee[:3, 3] = _PANDA_EE_OFFSET
    return (m @ ee)[:3, 3]


def load_qpos(path):
    rows = pd.read_parquet(path)["data"].iloc[0]
    return np.array([np.asarray(r, dtype=float) for r in rows])


qpos_path, rx, ry, rot_deg, ox, oy, oz, label = sys.argv[1:9]
rx, ry, rot_deg, ox, oy, oz = map(float, (rx, ry, rot_deg, ox, oy, oz))

q = load_qpos(qpos_path)[:, :7]
obj = np.array([ox, oy, oz])
base = np.array([rx, ry, DROID_BASE_HEIGHT])
R = _rot3(np.radians(rot_deg), 'z')

d = []
pts = []
for row in q:
    ee_local = panda_fk(row)
    ee_world = base + R @ ee_local
    pts.append(ee_world)
    d.append(np.linalg.norm(ee_world - obj))
d = np.array(d); pts = np.array(pts)
i = int(np.argmin(d))

print(f"=== {label} ===")
print(f"  шагов                    : {len(d)}")
print(f"  предмет                  : [{obj[0]:+.3f} {obj[1]:+.3f} {obj[2]:+.3f}]")
print(f"  основание руки (мир)     : [{base[0]:+.3f} {base[1]:+.3f} {base[2]:+.3f}], поворот {rot_deg:.0f}°")
print(f"  МИНИМАЛЬНОЕ расстояние   : {d.min():.3f} м  (шаг {i})")
print(f"  схват в этот момент      : [{pts[i][0]:+.3f} {pts[i][1]:+.3f} {pts[i][2]:+.3f}]")
print(f"  расстояние в конце       : {d[-1]:.3f} м")
print(f"  порог REACH              : 0.100 м  ->  {'ДОСТИГНУТ' if d.min() < 0.1 else 'НЕ достигнут'}")
print(f"  шагов ближе 0.2 м        : {(d < 0.2).sum()}")
print(f"  разброс схвата по осям   : x {pts[:,0].min():+.2f}..{pts[:,0].max():+.2f}  "
      f"y {pts[:,1].min():+.2f}..{pts[:,1].max():+.2f}  z {pts[:,2].min():+.2f}..{pts[:,2].max():+.2f}")
