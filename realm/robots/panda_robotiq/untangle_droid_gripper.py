"""Приводит артикуляцию гриппера в droid.usd к дереву -- так же, как это сделано
в стоковых Robotiq 2F-85 из BEHAVIOR-1K.

ЗАЧЕМ
Current OmniGibson (prims/entity_prim.py::_compute_articulation_tree) требует, чтобы
граф артикуляции был строгим деревом: in-degree 1 у каждого не-корневого линка.
В droid.usd Robotiq 2F-85 смоделирован как настоящий параллелограмм, и пять линков
имеют in-degree 2. Загрузить такой ассет движок отказывается.

Авторы REALM обошли это форком движка (realm/misc/modified_entity_prim.py -- копия
entity_prim.py с закомментированными ассертами). Здесь другой путь: чинится ассет,
а не движок.

Стоковый Robotiq из BEHAVIOR-1K (models/ur5e/usd/ur5e.usda, 17 суставов) замыкающее
звено параллелограмма просто не моделирует:

    base_gripper -> {left,right}_{inner,outer}_knuckle   (4 revolute)
    {inner,outer}_knuckle -> соответствующий finger      (4 revolute)

то есть дерево, и управление идёт двумя внешними суставами
(left/right_outer_knuckle_joint). Ровно к этому виду приводится и droid.usd.

ЧТО ДЕЛАЕТСЯ
1. Разворачиваются три сустава, заданные в направлении finger -> knuckle:
   left_outer_finger_knuckle_joint, left_inner_finger_knuckle_joint,
   right_inner_finger_knuckle_joint. (Разворот, а не удаление: именно они держат
   finger на своём knuckle, просто записаны наоборот.)
2. right_outer_finger_knuckle_joint перепривязывается с меш-примов
   (Defeatured_2F_85_...) на линки, с пересчётом относительной позы из фактических
   мировых трансформов -- это дефект экспорта, зеркальный левому.
3. Удаляются четыре сустава, которые и образуют петлю:
   {left,right}_inner_finger_prismatic_joint  (base -> inner_finger, второй путь
       к пальцу помимо inner_knuckle)
   {left,right}_inner_finger_joint            (outer_finger -> inner_finger,
       замыкающее звено параллелограмма)
4. Со всех суставов гриппера снимается physics:excludeFromArticulation.
   В исходном ассете он выставлен у всех шести revolute-суставов гриппера (у
   суставов руки -- нет), и это следствие той же петли: PhysX не поддерживает
   замкнутые контуры внутри артикуляции, поэтому весь гриппер был вынесен за её
   пределы и работал обычными констрейнтами. После разрыва петли флаг вреден --
   с ним движок видит в артикуляции только 7 суставов руки из 13 и падает на
   entity_prim.py::update_joints ("Number of joints inferred from prim tree (13)
   does not match number of joints found in the articulation view (7)").

ЦЕНА
Пальцы перестают быть кинематически связанными: параллельность губок теперь не
обеспечивается механизмом, как в реальном 2F-85, а определяется управлением. Это
ровно то допущение, на котором работают все стоковые Robotiq в BEHAVIOR-1K.
Управляемых суставов гриппера становится 2 (left/right_outer_knuckle_joint) вместо
объявленных REALM четырёх -- см. droid_robot_definition.yaml.

ЧТО СОХРАНЯЕТСЯ, И РАДИ ЧЕГО ВСЁ ЭТО
Камера запястья /panda/gripper_link_camera/Camera на panda_link8 -- та самая, под
которую обучалась pi0-FAST, и которой нет у стокового franka_robotiq (там камера
сидит на panda_link7 как camera_link). Плюс геометрия и eef-линк настоящего DROID.

Запуск:  python untangle_droid_gripper.py <путь к droid.usd>
Идемпотентен: повторный запуск на уже починенном файле ничего не меняет.
"""
import collections
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

usd_path = sys.argv[1]
stage = Usd.Stage.Open(usd_path)
TIME = Usd.TimeCode.Default()

# Суставы, записанные как finger -> knuckle; должны быть knuckle -> finger.
TO_REVERSE = [
    "/panda/gripper_link_left_outer_knuckle/left_outer_finger_knuckle_joint",
    "/panda/gripper_link_left_inner_knuckle/left_inner_finger_knuckle_joint",
    "/panda/gripper_link_right_inner_knuckle/right_inner_finger_knuckle_joint",
]

# Сустав с телами, указывающими на меш-примы вместо линков: (путь, body0, body1).
TO_REBIND = (
    "/panda/gripper_link_right_outer_knuckle/right_outer_finger_knuckle_joint",
    "/panda/gripper_link_right_outer_knuckle",
    "/panda/gripper_link_right_outer_finger",
)

# Суставы, образующие вторые пути к пальцам -- замыкание параллелограмма.
TO_REMOVE = [
    "/panda/gripper_link_base/left_inner_finger_prismatic_joint",
    "/panda/gripper_link_base/right_inner_finger_prismatic_joint",
    "/panda/gripper_link_left_outer_finger/left_inner_finger_joint",
    "/panda/gripper_link_right_outer_finger/right_inner_finger_joint",
]

# Ведомые суставы: (путь, ведущий сустав, gearing, нижний предел, верхний предел).
# Значения скопированы со стокового Robotiq 2F-85 (models/ur5e/usd/ur5e.usda), где
# ровно эта схема и применена: два ведущих outer_knuckle плюс mimic на остальных.
MIMIC = [
    ("/panda/gripper_link_base/left_inner_knuckle_joint",
     "/panda/gripper_link_base/left_outer_knuckle_joint", -1.0, 0.0, 45.0),
    ("/panda/gripper_link_base/right_inner_knuckle_joint",
     "/panda/gripper_link_base/right_outer_knuckle_joint", -1.0, 0.0, 45.0),
    ("/panda/gripper_link_left_inner_knuckle/left_inner_finger_knuckle_joint",
     "/panda/gripper_link_base/left_outer_knuckle_joint", -1.0, -45.0, 45.0),
    ("/panda/gripper_link_right_inner_knuckle/right_inner_finger_knuckle_joint",
     "/panda/gripper_link_base/right_outer_knuckle_joint", -1.0, -45.0, 45.0),
]


def joints():
    """(имя, prim, body0_link, body1_link) по всем суставам стадии."""
    out = []
    for p in stage.Traverse():
        if "Joint" not in str(p.GetTypeName()):
            continue
        rels = {r.GetName(): r for r in p.GetRelationships()}
        b0, b1 = rels.get("physics:body0"), rels.get("physics:body1")
        t0 = b0.GetTargets()[0].pathString.split("/")[-1] if b0 and b0.GetTargets() else None
        t1 = b1.GetTargets()[0].pathString.split("/")[-1] if b1 and b1.GetTargets() else None
        out.append((p.GetName(), p, t0, t1))
    return out


def report(label):
    js = joints()
    indeg = collections.Counter(t1 for _, _, _, t1 in js if t1)
    loops = {k: v for k, v in indeg.items() if v > 1}
    link_names = [
        p.GetName() for p in stage.GetDefaultPrim().GetChildren() if p.GetTypeName() == "Xform"
    ]
    roots = sorted(set(link_names) - set(indeg))
    print(f"{label}: суставов={len(js)}  корней={roots}  петли={loops or 'нет'}")
    return roots, loops


report("ДО ")

# --- 1. Развернуть суставы, записанные наоборот -------------------------------
for path in TO_REVERSE:
    prim = stage.GetPrimAtPath(path)
    assert prim and prim.IsValid(), f"нет прима {path}"
    j = UsdPhysics.Joint(prim)
    b0 = j.GetBody0Rel().GetTargets()[0]
    b1 = j.GetBody1Rel().GetTargets()[0]
    if b0.pathString.endswith("_knuckle"):
        print(f"  уже развёрнут, пропуск: {prim.GetName()}")
        continue
    p0, p1 = j.GetLocalPos0Attr().Get(), j.GetLocalPos1Attr().Get()
    r0, r1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
    j.GetBody0Rel().SetTargets([b1])
    j.GetBody1Rel().SetTargets([b0])
    j.GetLocalPos0Attr().Set(p1)
    j.GetLocalPos1Attr().Set(p0)
    j.GetLocalRot0Attr().Set(r1)
    j.GetLocalRot1Attr().Set(r0)
    print(f"  развёрнут: {prim.GetName()}  {b1.name} -> {b0.name}")

# --- 2. Перепривязать сустав с меш-примов на линки ----------------------------
jpath, knuckle_path, finger_path = TO_REBIND
prim = stage.GetPrimAtPath(jpath)
assert prim and prim.IsValid(), f"нет прима {jpath}"
j = UsdPhysics.Joint(prim)
if j.GetBody0Rel().GetTargets()[0].pathString == knuckle_path:
    print(f"  уже перепривязан, пропуск: {prim.GetName()}")
else:
    knuckle_w = UsdGeom.Xformable(stage.GetPrimAtPath(knuckle_path)).ComputeLocalToWorldTransform(TIME)
    finger_w = UsdGeom.Xformable(stage.GetPrimAtPath(finger_path)).ComputeLocalToWorldTransform(TIME)
    rel = finger_w * knuckle_w.GetInverse()
    rel_t, rel_q = rel.ExtractTranslation(), rel.ExtractRotationQuat()
    j.GetBody0Rel().SetTargets([Sdf.Path(knuckle_path)])
    j.GetBody1Rel().SetTargets([Sdf.Path(finger_path)])
    j.GetLocalPos0Attr().Set(Gf.Vec3f(rel_t))
    j.GetLocalRot0Attr().Set(Gf.Quatf(rel_q.GetReal(), Gf.Vec3f(rel_q.GetImaginary())))
    j.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    j.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    print(f"  перепривязан: {prim.GetName()}  t={rel_t}")

    # Сверка с зеркальным левым суставом -- относительная поза должна совпасть по модулю.
    lk = UsdGeom.Xformable(stage.GetPrimAtPath("/panda/gripper_link_left_outer_knuckle")).ComputeLocalToWorldTransform(TIME)
    lf = UsdGeom.Xformable(stage.GetPrimAtPath("/panda/gripper_link_left_outer_finger")).ComputeLocalToWorldTransform(TIME)
    print(f"    сверка, слева:                            t={(lf * lk.GetInverse()).ExtractTranslation()}")

# --- 3. Удалить замыкающие суставы -------------------------------------------
for path in TO_REMOVE:
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(Sdf.Path(path))
        print(f"  удалён: {path.split('/')[-1]}")
    else:
        print(f"  уже удалён, пропуск: {path.split('/')[-1]}")

# --- 4. Вернуть суставы гриппера в артикуляцию --------------------------------
# Флаг стоял из-за петли (PhysX не держит замкнутые контуры в артикуляции). Петли
# больше нет, и без этого шага движок насчитает в articulation view только суставы
# руки. Ограничиваемся суставами внутри гриппера, чтобы не трогать ничего лишнего.
freed = 0
for _, prim, _, _ in joints():
    if not prim.GetName().startswith(("left_", "right_")):
        continue
    attr = prim.GetAttribute("physics:excludeFromArticulation")
    if attr and attr.IsValid() and attr.Get():
        attr.Set(False)
        freed += 1
        print(f"  возвращён в артикуляцию: {prim.GetName()}")
print(f"снято excludeFromArticulation: {freed}")

# --- 5. Восстановить параллелограмм mimic-связями ------------------------------
# Разрыв петли сделал четыре сустава ничьими: ими никто не управляет, а привод на
# них остался, из-за чего движок падает на robot.py::update_controller_mode
# ("All unused joints not mapped to any controller should not have DriveAPI").
# Стоковый Robotiq решает это так: DriveAPI только у двух ведущих outer_knuckle,
# остальные суставы ведомые -- PhysxMimicJointAPI с gearing -1 относительно
# ведущего. Это и восстанавливает согласованное движение губок, которое раньше
# обеспечивала петля, но уже без петли в артикуляции.
layer = stage.GetRootLayer()
for jpath, ref_path, gearing, lo, hi in MIMIC:
    prim = stage.GetPrimAtPath(jpath)
    assert prim and prim.IsValid(), f"нет прима {jpath}"
    assert stage.GetPrimAtPath(ref_path), f"нет ведущего сустава {ref_path}"

    # Привод снимаем: сустав больше не управляется напрямую.
    if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
        prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")

    prim.CreateAttribute("physxMimicJoint:rotX:gearing", Sdf.ValueTypeNames.Float).Set(gearing)
    prim.CreateRelationship("physxMimicJoint:rotX:referenceJoint").SetTargets([Sdf.Path(ref_path)])

    # Пределы у ведомых суставов в исходном ассете бесконечны -- они были
    # констрейнтами петли. Ставим те же, что у стокового Robotiq.
    prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(lo)
    prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(hi)

    # PhysxMimicJointAPI живёт в плагине PhysxSchema, которого нет в чистом
    # usd-core, поэтому применяем её через список apiSchemas напрямую.
    spec = layer.GetPrimAtPath(jpath)
    assert spec is not None, f"нет prim spec для {jpath}"
    existing = list(spec.GetInfo("apiSchemas").GetAddedOrExplicitItems())
    existing = [s for s in existing if not s.startswith("PhysicsDriveAPI")]
    if "PhysxMimicJointAPI:rotX" not in existing:
        existing.append("PhysxMimicJointAPI:rotX")
    spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(existing))
    print(f"  ведомый: {prim.GetName():36s} -> {ref_path.split('/')[-1]}  gearing={gearing}")

stage.GetRootLayer().Save()

roots, loops = report("ПОСЛЕ")
assert not loops, f"петли остались: {loops}"
assert roots == ["base_link"], f"ожидался единственный корень base_link, получено: {roots}"
print("OK: артикуляция -- дерево с единственным корнем base_link")
