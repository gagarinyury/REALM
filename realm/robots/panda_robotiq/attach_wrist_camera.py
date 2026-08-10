"""Перенести камеру запястья из эталонного droid.usd в импортированного робота.

ЗАЧЕМ. Ракурс камеры запястья — это то, на чём обучался pi0-FAST: смести её, и модель
получит вид, которого не видела ни разу, а прогон при этом отработает без единой ошибки
и покажет низкий результат, который спишут на симулятор. Поэтому положение камеры не
подбирается и не пересчитывается на глаз — оно ЧИТАЕТСЯ из эталонного ассета, на котором
получен опубликованный baseline 0.61, и переносится один в один.

ЧТО БЫЛО НЕ ТАК. В URDF, из которого собран droid2, камеры нет вообще — только рука и
гриппер. При первой сборке (10.08.2026) её подвесили к panda_link7, взяв смещение
(-0.031, 0.074, -0.011) прямо из эталона. Смещение это верное, но относится к ДРУГОМУ
уровню иерархии: в droid.usd так смещён prim `Camera` ВНУТРИ тела `gripper_link_camera`,
а само тело закреплено на panda_link8 почти без сдвига. В результате камера уехала на
12 см и получила лишний поворот. Замер расхождения — в проверке в конце файла.

ЧТО ДЕЛАЕТ СКРИПТ. Читает из droid.usd две трансформации:

    T1 = поза тела gripper_link_camera относительно panda_link7
    T2 = поза prim Camera внутри этого тела (плюс его масштаб)

и воспроизводит обе в droid2.usda. Родителем становится panda_link7, потому что
panda_link8 в URDF-версии робота нет; T1 читается относительно link7 именно поэтому —
чтобы отсутствие промежуточного звена ничего не меняло. Локальные системы координат
panda_link7 у обеих моделей совпадают (обе — Franka Panda, joint7 одинаков), это
проверено сравнением поз link5/link6/link7 в обоих ассетах.

ОПТИКУ скрипт не трогает намеренно: OmniGibson >= 3.9.1 при инициализации VisionSensor
переписывает focalLength и horizontalAperture значениями из конфига робота
(sensors/vision_sensor.py:260-263), поэтому что бы ни лежало в USD, действуют
focal_length 2.8 / horizontal_aperture 5.376 из realm/config/robots/DROID2.yaml.

Запуск (на машине с ассетами):
    python realm/robots/panda_robotiq/attach_wrist_camera.py \
        --reference /data/omnigibson-robot-assets/models/droid/usd/droid.usd \
        --target    /data/omnigibson-robot-assets/objects/robot/droid2/usd/droid2.usda
"""

import argparse

from pxr import Gf, Usd, UsdGeom


def local_xform(prim):
    """Снимок позы prim'а: (translate, orient) как записаны в файле."""
    t = prim.GetAttribute("xformOp:translate").Get()
    q = prim.GetAttribute("xformOp:orient").Get()
    m = Gf.Matrix4d(1.0)
    if q is not None:
        m.SetRotateOnly(Gf.Quatd(q.GetReal(), Gf.Vec3d(*q.GetImaginary())))
    if t is not None:
        m.SetTranslateOnly(Gf.Vec3d(*t))
    return m


def find(stage, name):
    for p in stage.Traverse():
        if p.GetName() == name:
            return p
    raise SystemExit(f"нет prim'а {name}")


def read_reference(path):
    """Из эталона: поза тела камеры относительно link7 и поза сенсора внутри тела."""
    st = Usd.Stage.Open(path)
    link7 = find(st, "panda_link7")
    body = find(st, "gripper_link_camera")
    cam = st.GetPrimAtPath(body.GetPath().AppendChild("Camera"))

    # USD-конвенция (вектор-строка): world = local * parent, значит local = world * parent^-1
    body_in_link7 = local_xform(body) * local_xform(link7).GetInverse()

    cam_t = cam.GetAttribute("xformOp:translate").Get()
    cam_q = cam.GetAttribute("xformOp:orient").Get()
    cam_s = cam.GetAttribute("xformOp:scale").Get()
    return body_in_link7, cam_t, cam_q, cam_s


def decompose(m):
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat().GetNormalized()
    return Gf.Vec3f(*t), Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*q.GetImaginary()))


def write_target(path, body_in_link7, cam_t, cam_q, cam_s):
    st = Usd.Stage.Open(path)
    link7 = find(st, "panda_link7")
    body = find(st, "gripper_link_camera")
    joint = find(st, "panda_link7_gripper_link_camera_joint")
    cam = st.GetPrimAtPath(body.GetPath().AppendChild("Camera"))

    t, q = decompose(body_in_link7)

    # 1. Сустав: точка и разворот крепления в системе panda_link7. Вторая сторона
    #    (localPos1/localRot1 на теле камеры) остаётся нулевой — тогда поза тела
    #    относительно звена в точности равна тому, что прочитано из эталона.
    joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(t))
    joint.GetAttribute("physics:localRot0").Set(q)
    for name, val in (("physics:localPos1", Gf.Vec3f(0, 0, 0)),
                      ("physics:localRot1", Gf.Quatf(1, 0, 0, 0))):
        a = joint.GetAttribute(name)
        if not a.IsValid():
            a = joint.CreateAttribute(name, Gf.Vec3f if "Pos" in name else Gf.Quatf)
        a.Set(val)

    # 2. Снимок позы самого тела — чтобы на первом кадре камера не оказалась в
    #    стороне и не «прилетала» к суставу рывком.
    world = body_in_link7 * local_xform(link7)
    wt, wq = decompose(world)
    body.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*wt))
    body.GetAttribute("xformOp:orient").Set(Gf.Quatd(float(wq.GetReal()), Gf.Vec3d(*wq.GetImaginary())))

    # 3. Сам сенсор внутри тела — ровно как в эталоне, включая масштаб.
    cam.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*cam_t))
    cam.GetAttribute("xformOp:orient").Set(Gf.Quatd(cam_q.GetReal(), Gf.Vec3d(*cam_q.GetImaginary())))
    cam.GetAttribute("xformOp:scale").Set(Gf.Vec3d(*cam_s))

    st.GetRootLayer().Save()
    return t, q


def verify(ref_path, tgt_path):
    """Поза сенсора относительно panda_link7 в обеих моделях — должна совпасть."""
    out = []
    for path, link_name in ((ref_path, "panda_link7"), (tgt_path, "panda_link7")):
        st = Usd.Stage.Open(path)
        link7 = find(st, link_name)
        body = find(st, "gripper_link_camera")
        cam = st.GetPrimAtPath(body.GetPath().AppendChild("Camera"))
        sensor_in_link7 = local_xform(cam) * local_xform(body) * local_xform(link7).GetInverse()
        t, q = decompose(sensor_in_link7)
        out.append((tuple(round(x, 5) for x in t), q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="droid.usd — ассет, на котором получен baseline")
    ap.add_argument("--target", required=True, help="droid2.usda — импортированный из URDF")
    a = ap.parse_args()

    before = verify(a.reference, a.target)
    print(f"ДО   эталон: t={before[0][0]}  q={before[0][1]}")
    print(f"ДО   droid2: t={before[1][0]}  q={before[1][1]}")

    body_in_link7, cam_t, cam_q, cam_s = read_reference(a.reference)
    t, q = write_target(a.target, body_in_link7, cam_t, cam_q, cam_s)
    print(f"\nперенесено: тело камеры на panda_link7, t={tuple(round(x,5) for x in t)}  q={q}")
    print(f"            сенсор внутри тела, t={cam_t}  q={cam_q}  scale={cam_s}")

    after = verify(a.reference, a.target)
    print(f"\nПОСЛЕ эталон: t={after[0][0]}  q={after[0][1]}")
    print(f"ПОСЛЕ droid2: t={after[1][0]}  q={after[1][1]}")
    d = max(abs(x - y) for x, y in zip(after[0][0], after[1][0]))
    print(f"\nрасхождение по положению: {d:.6f} м")


if __name__ == "__main__":
    main()
