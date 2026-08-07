"""Чинит топологию артикуляции в droid.usd, чтобы у робота был ровно один root link.

OmniGibson (objects/usd_object.py::_preapply_articulation_root) определяет корневой
линк как единственный Xform-потомок, который не является body1 ни одного сустава.
В droid.usd таких оказывается три: base_link (настоящий корень) и оба
gripper_link_*_outer_finger, потому что соединяющие их суставы заданы некорректно:

  left_outer_finger_knuckle_joint   body0=outer_finger, body1=outer_knuckle  (перевёрнут)
  right_outer_finger_knuckle_joint  body0/body1 указывают на меш-примы, а не на линки

Оба сустава -- FixedJoint, соединяющий outer_knuckle с outer_finger. Левый чинится
разворотом (с обменом localPos/localRot, которые в нём и так почти идентичны).
Правый перепривязывается на линки, а его localPos0/localRot0 вычисляются из
фактических мировых трансформов линков в ассете, а не подбираются вручную.
"""
import sys

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

usd_path = sys.argv[1]
stage = Usd.Stage.Open(usd_path)


def prim(path):
    p = stage.GetPrimAtPath(path)
    assert p and p.IsValid(), f"нет прима {path}"
    return p


def root_candidates():
    default_prim = stage.GetDefaultPrim()
    link_names = [p.GetName() for p in default_prim.GetChildren() if p.GetTypeName() == "Xform"]
    joint_children = set()
    for p in stage.Traverse():
        if "Joint" not in str(p.GetTypeName()):
            continue
        rels = {r.GetName(): r for r in p.GetRelationships()}
        b0, b1 = rels.get("physics:body0"), rels.get("physics:body1")
        if b0 is None or b1 is None:
            continue
        if len(b0.GetTargets()) > 0 and len(b1.GetTargets()) > 0:
            joint_children.add(b1.GetTargets()[0].pathString.split("/")[-1])
    return sorted(set(link_names) - joint_children)


print("кандидаты в root ДО правки:", root_candidates())

# --- 1. Левый сустав: развернуть направление ---
lj = UsdPhysics.Joint(prim("/panda/gripper_link_left_outer_knuckle/left_outer_finger_knuckle_joint"))
b0 = lj.GetBody0Rel().GetTargets()[0]
b1 = lj.GetBody1Rel().GetTargets()[0]
p0, p1 = lj.GetLocalPos0Attr().Get(), lj.GetLocalPos1Attr().Get()
r0, r1 = lj.GetLocalRot0Attr().Get(), lj.GetLocalRot1Attr().Get()
lj.GetBody0Rel().SetTargets([b1])
lj.GetBody1Rel().SetTargets([b0])
lj.GetLocalPos0Attr().Set(p1)
lj.GetLocalPos1Attr().Set(p0)
lj.GetLocalRot0Attr().Set(r1)
lj.GetLocalRot1Attr().Set(r0)
print(f"левый сустав развёрнут: body0={b1} body1={b0}")

# --- 2. Правый сустав: перепривязать на линки, позу пересчитать из ассета ---
knuckle_path = "/panda/gripper_link_right_outer_knuckle"
finger_path = "/panda/gripper_link_right_outer_finger"
t = Usd.TimeCode.Default()
knuckle_w = UsdGeom.Xformable(prim(knuckle_path)).ComputeLocalToWorldTransform(t)
finger_w = UsdGeom.Xformable(prim(finger_path)).ComputeLocalToWorldTransform(t)
rel = finger_w * knuckle_w.GetInverse()  # поза finger в системе knuckle
rel_t = rel.ExtractTranslation()
rel_q = rel.ExtractRotationQuat()

rj = UsdPhysics.Joint(prim("/panda/gripper_link_right_outer_knuckle/right_outer_finger_knuckle_joint"))
rj.GetBody0Rel().SetTargets([Sdf.Path(knuckle_path)])
rj.GetBody1Rel().SetTargets([Sdf.Path(finger_path)])
rj.GetLocalPos0Attr().Set(Gf.Vec3f(rel_t))
rj.GetLocalRot0Attr().Set(Gf.Quatf(rel_q.GetReal(), Gf.Vec3f(rel_q.GetImaginary())))
rj.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
rj.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
print(f"правый сустав перепривязан на линки; относительная поза: t={rel_t} q={rel_q}")

# Сверка: у левого (эталонного) сустава относительная поза должна быть близкой по модулю
lk_w = UsdGeom.Xformable(prim("/panda/gripper_link_left_outer_knuckle")).ComputeLocalToWorldTransform(t)
lf_w = UsdGeom.Xformable(prim("/panda/gripper_link_left_outer_finger")).ComputeLocalToWorldTransform(t)
lrel = lf_w * lk_w.GetInverse()
print(f"для сверки, та же поза слева:            t={lrel.ExtractTranslation()} q={lrel.ExtractRotationQuat()}")

stage.GetRootLayer().Save()

after = root_candidates()
print("\nкандидаты в root ПОСЛЕ правки:", after)
assert after == ["base_link"], f"ожидался единственный корень base_link, получено: {after}"
print("OK: ровно один корневой линк")
