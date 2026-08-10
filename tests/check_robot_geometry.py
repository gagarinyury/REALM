"""Проверка геометрии собранного робота: то, чего штатный аудит не смотрит.

ЗАЧЕМ. `urdf_audit` (omnigibson/utils/urdf_preprocessing.py:248) считает звенья, суставы,
mimic-связи, форматы мешей, битые пути и звенья без <inertial> — то есть разбирает XML.
Размеры получившейся геометрии не проверяет ни он, ни импортёр: слово `scale` в
`import_custom_robot.py` не встречается ни разу.

Из-за этого 10.08.2026 сборка `droid2` прошла импорт без единого замечания, загрузилась в
сцену и отработала полный прогон — при том, что база её гриппера имела размер 7.5 x 8.5 x
9.4 м визуально и 75 x 86 x 94 м по коллизии, а у левой половины губок коллизионной
геометрии не было вовсе. Захват был невозможен физически; наружу это выходило только как
низкий результат, который легко списать на симулятор.

Причина была в единицах: URDF подключает меши с `scale="0.1"` для визуала и `"0.001"` для
коллизии, а подставленные файлы (ros-industrial/robotiq) в других единицах, причём у
визуала и коллизии разных.

ЧТО ПРОВЕРЯЕТ. Три вещи, каждая из которых ловит эту ошибку за секунды и без GPU:

  1. звено крупнее разумного предела (по умолчанию 1 м — рука робота-манипулятора);
  2. визуальный и коллизионный меши одного звена расходятся в размере больше чем в N раз;
  3. у звена есть визуал, но нет коллизии — оно видимо и при этом бесплотно.

Запуск (нужен только usd-core, симулятор поднимать не надо):
    python tests/check_robot_geometry.py /путь/к/robot.usda [--max-size 1.0] [--ratio 2.0]

Возвращает ненулевой код, если что-то не сошлось, — чтобы вызывать из сборочного скрипта.
"""

import argparse
import sys

from pxr import Usd, UsdGeom


def link_extents(stage, root_name=None):
    """Габариты визуальной и коллизионной геометрии каждого звена в системе этого звена."""
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy],
    )
    out = {}
    for prim in stage.Traverse():
        parent = prim.GetParent()
        if root_name is not None and parent.GetName() != root_name:
            continue
        if parent.GetPath().pathString.count("/") > 1 and root_name is None:
            continue
        sizes = {}
        for child in prim.GetChildren():
            if child.GetName() not in ("visuals", "collisions"):
                continue
            r = cache.ComputeRelativeBound(child, prim).ComputeAlignedRange()
            if r.IsEmpty():
                continue
            d = r.GetMax() - r.GetMin()
            sizes[child.GetName()] = (float(d[0]), float(d[1]), float(d[2]))
        if sizes:
            out[prim.GetName()] = sizes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usd", help="USD собранного робота")
    ap.add_argument("--root", default=None, help="имя корневого prim'а робота (по умолчанию — угадать)")
    ap.add_argument("--max-size", type=float, default=1.0, help="предел размера звена, м")
    ap.add_argument("--ratio", type=float, default=2.0, help="во сколько раз визуал и коллизия могут расходиться")
    a = ap.parse_args()

    stage = Usd.Stage.Open(a.usd)
    if stage is None:
        sys.exit(f"не открылся: {a.usd}")

    root = a.root
    if root is None:
        # Корень робота — prim верхнего уровня, у которого больше всего потомков-звеньев.
        tops = [p for p in stage.GetPseudoRoot().GetChildren()]
        root = max(tops, key=lambda p: len(list(p.GetChildren()))).GetName() if tops else None

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"{a.usd}\nкорень: {root}, metersPerUnit = {mpu}\n")

    problems = []
    print(f"{'звено':36s} {'визуал (м)':26s} {'коллизия (м)':26s}")
    for name, sizes in link_extents(stage, root).items():
        vis = sizes.get("visuals")
        col = sizes.get("collisions")
        fmt = lambda s: "—" if s is None else "x".join(f"{v * mpu:.3f}" for v in s)
        mark = ""

        big = [s for s in (vis, col) if s is not None and max(s) * mpu > a.max_size]
        if big:
            problems.append(f"{name}: звено крупнее {a.max_size} м — {fmt(max(big, key=max))}")
            mark = "  <-- РАЗМЕР"

        if vis is not None and col is None:
            problems.append(f"{name}: есть визуал, нет коллизии — звено бесплотно")
            mark = "  <-- НЕТ КОЛЛИЗИИ"
        elif vis is not None and col is not None:
            r = max(max(vis) / max(col), max(col) / max(vis))
            if r > a.ratio:
                problems.append(f"{name}: визуал и коллизия расходятся в {r:.0f} раз")
                mark = f"  <-- РАСХОЖДЕНИЕ x{r:.0f}"

        print(f"{name:36s} {fmt(vis):26s} {fmt(col):26s}{mark}")

    if problems:
        print(f"\nНАЙДЕНО {len(problems)}:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nгеометрия в порядке")


if __name__ == "__main__":
    main()
