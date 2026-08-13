"""Проверяет точку установки робота REALM по геометрии сцены, без симулятора.

Читает JSON сцены BEHAVIOR-1K, берёт позу робота и спавн-бокс из scenes.yaml и отвечает
на три вопроса: (1) есть ли под точкой спавна предмета горизонтальная опора, (2) на какой
она высоте против заявленной z, (3) что стоит вплотную к роботу (не упирается ли он в стену).

Запуск:  python check_spot.py <SCENE> <ROBOT_X> <ROBOT_Y> <OBJ_X> <OBJ_Y> <OBJ_Z>
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = r"C:\thesis-yahor-pachkouski\repos\BEHAVIOR-1K-main\datasets\behavior-1k-assets\scenes"
SURFACE = ("table", "desk", "countertop", "counter", "shelf", "stand", "cabinet")

scene, rx, ry, ox, oy, oz = sys.argv[1], *map(float, sys.argv[2:7])

files = glob.glob(os.path.join(ROOT, scene, "json", "*_best.json"))
if not files:
    files = glob.glob(os.path.join(ROOT, scene, "json", "*.json"))
d = json.load(open(files[0], encoding="utf-8"))
objs = d["state"]["registry"]["object_registry"]
print(f"сцена {scene}: {os.path.basename(files[0])}, объектов {len(objs)}\n")

rows = []
for name, o in objs.items():
    p = o.get("root_link", {}).get("pos")
    if not p:
        continue
    p = np.array(p, dtype=float)
    rows.append((name, p, float(np.hypot(p[0] - rx, p[1] - ry)), float(np.hypot(p[0] - ox, p[1] - oy))))

print("ЧТО РЯДОМ С РОБОТОМ (радиус 1.5 м):")
for name, p, dr, do in sorted(rows, key=lambda r: r[2])[:10]:
    if dr > 1.5:
        break
    print(f"   {dr:5.2f} м  {name:40} z={p[2]:6.3f}")

print("\nОПОРЫ РЯДОМ С ТОЧКОЙ СПАВНА ПРЕДМЕТА (радиус 1.5 м):")
found = False
for name, p, dr, do in sorted(rows, key=lambda r: r[3]):
    if do > 1.5:
        break
    if any(w in name for w in SURFACE):
        found = True
        print(f"   {do:5.2f} м от предмета, {dr:5.2f} м от робота  {name:36} z={p[2]:6.3f}")
if not found:
    print("   НЕТ НИ ОДНОЙ ОПОРЫ -- предмет висит в воздухе")

nearest = min(rows, key=lambda r: r[2])
print(f"\nближайший объект к роботу : {nearest[0]} на {nearest[2]:.2f} м")
print(f"заявленная высота опоры   : z={oz:.3f}")
