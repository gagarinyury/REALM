"""Переносит начало координат droid.usd в корневой линк base_link.

ЗАЧЕМ
Current OmniGibson требует, чтобы поза entity prim совпадала с позой корневого
линка -- prims/entity_prim.py::set_position_orientation, строка ~1019:

    assert th.allclose(this_position, root_link_position, atol=1e-2), \
        "Position mismatch between entity prim and root link"

В droid.usd это не так. Default prim /panda стоит в z = -0.00156, а base_link --
в z = -0.85227, потому что base_link изображает пол под подставкой робота, тогда
как начало координат ассета лежит на уровне столешницы. Расхождение 0.85 м при
допуске 1 см, и робот не загружается.

ЧТО ДЕЛАЕТСЯ
Чистая репараметризация: точка отсчёта переносится в base_link.

    /panda.translate      += base_link.translate
    <каждый прямой ребёнок>.translate -= base_link.translate

Мировые позы всех линков при этом не меняются -- меняется только то, относительно
чего они записаны. Относительные позы тел, а значит и локальные системы всех
суставов (localPos0/localPos1), сохраняются, поэтому физика идентична исходной.
После правки base_link оказывается ровно в начале координат /panda, и проверка
движка проходит.

Скрипт намеренно отказывается работать, если у /panda или base_link ненулевой
поворот либо масштаб: тогда сложение трансляций некорректно и нужен полноценный
пересчёт матриц. В droid.usd оба тождественны, что и проверяется.

СЛЕДСТВИЕ ДЛЯ REALM
После этого корень робота -- пол подставки, а рука оказывается на 0.8645 м выше
(при DROID_BASE_HEIGHT = 0.86244 в env_dynamic.py -- сходится с точностью 2 мм).
Это ровно то допущение, под которое написан оригинальный REALM: позы в scenes.yaml
для REALM_DROID10 заданы с z = 0, а камеры и world<->robot преобразования сами
добавляют DROID_BASE_HEIGHT. Значит для родного droid.usd поправку на высоту базы
к позиции робота добавлять НЕ нужно -- в отличие от стокового franka_robotiq,
который представляет собой одну лишь руку.

Запуск:  python align_droid_root.py <путь к droid.usd>
Идемпотентен: если base_link уже в начале координат, ничего не делает.
"""
import sys

from pxr import Gf, Usd, UsdGeom

usd_path = sys.argv[1]
stage = Usd.Stage.Open(usd_path)
TIME = Usd.TimeCode.Default()

root = stage.GetDefaultPrim()
base = stage.GetPrimAtPath(f"{root.GetPath()}/base_link")
assert base and base.IsValid(), "в ассете нет /panda/base_link"


def translate_op(prim):
    """xformOp:translate прима; None, если его нет."""
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            return op
    return None


def assert_trivial(prim):
    """Поворот и масштаб должны быть тождественными -- иначе сложение сдвигов неверно."""
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        name, val = op.GetOpName(), op.Get()
        if name == "xformOp:orient":
            q = Gf.Quatf(val)
            assert abs(q.GetReal() - 1.0) < 1e-6 and Gf.Vec3f(q.GetImaginary()).GetLength() < 1e-6, (
                f"{prim.GetPath()} имеет нетождественный поворот {val}; нужен полный пересчёт матриц"
            )
        elif name == "xformOp:scale":
            assert (Gf.Vec3d(val) - Gf.Vec3d(1, 1, 1)).GetLength() < 1e-6, (
                f"{prim.GetPath()} имеет масштаб {val}; нужен полный пересчёт матриц"
            )


assert_trivial(root)
assert_trivial(base)

base_op = translate_op(base)
assert base_op is not None, "у base_link нет xformOp:translate"
shift = Gf.Vec3d(base_op.Get())

print(f"локальный сдвиг base_link: {shift}")
if shift.GetLength() < 1e-9:
    print("base_link уже в начале координат -- нечего делать")
    sys.exit(0)

before = {
    p.GetName(): UsdGeom.Xformable(p).ComputeLocalToWorldTransform(TIME).ExtractTranslation()
    for p in root.GetChildren()
    if p.GetTypeName() == "Xform"
}

# Корень опускается на сдвиг base_link, все линки поднимаются на него же.
root_op = translate_op(root)
assert root_op is not None, "у default prim нет xformOp:translate"
root_op.Set(Gf.Vec3d(root_op.Get()) + shift)

moved = 0
for child in root.GetChildren():
    if child.GetTypeName() != "Xform":
        continue
    op = translate_op(child)
    if op is None:
        print(f"  ПРОПУСК (нет translate): {child.GetName()}")
        continue
    op.Set(Gf.Vec3d(op.Get()) - shift)
    moved += 1

stage.GetRootLayer().Save()
print(f"перепривязано линков: {moved}")

# --- Проверки ---------------------------------------------------------------
stage = Usd.Stage.Open(usd_path)
root = stage.GetDefaultPrim()
root_w = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(TIME).ExtractTranslation()
base_w = UsdGeom.Xformable(stage.GetPrimAtPath(f"{root.GetPath()}/base_link")).ComputeLocalToWorldTransform(TIME).ExtractTranslation()
print(f"\n/panda    -> {root_w}\nbase_link -> {base_w}")
assert (Gf.Vec3d(root_w) - Gf.Vec3d(base_w)).GetLength() < 1e-2, "корень и base_link всё ещё расходятся"

worst = 0.0
for p in root.GetChildren():
    if p.GetTypeName() != "Xform" or p.GetName() not in before:
        continue
    now = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(TIME).ExtractTranslation()
    worst = max(worst, (Gf.Vec3d(now) - Gf.Vec3d(before[p.GetName()])).GetLength())
print(f"максимальное смещение линка в мировых координатах: {worst:.3e} м")
assert worst < 1e-9, "мировые позы линков изменились -- это не репараметризация"
print("OK: корень совмещён с base_link, геометрия не тронута")
