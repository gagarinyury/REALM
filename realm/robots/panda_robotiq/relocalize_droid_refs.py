"""Переписывает http-референсы Robotiq внутри droid.usd на локальные файлы,
лежащие рядом. REALM скачал эти файлы в репозиторий, но ссылки в USD оставил
указывающими на omniverse-content-production S3 (в их Docker Isaac резолвил
их через кеш Omniverse). Без этого гриппер грузится только при наличии сети.
"""
import os
import sys
from pxr import Sdf, Usd

usd_path = sys.argv[1]
layer = Sdf.Layer.FindOrOpen(usd_path)
assert layer is not None, f"не удалось открыть слой: {usd_path}"

usd_dir = os.path.dirname(os.path.abspath(usd_path))
refs = sorted(layer.GetExternalReferences())
changed = 0
for ref in refs:
    if not ref.startswith("http"):
        continue
    basename = os.path.basename(ref)
    local = os.path.join(usd_dir, basename)
    assert os.path.exists(local), f"локальная копия отсутствует: {local}"
    layer.UpdateExternalReference(ref, f"./{basename}")
    print(f"  {basename}\n    {ref}\n    -> ./{basename}")
    changed += 1

if changed:
    layer.Save()
print(f"\nпереписано ссылок: {changed}")

# Проверка: стадия должна открыться без нерезолвленных ассетов
stage = Usd.Stage.Open(usd_path)
remaining = [r for r in Sdf.Layer.FindOrOpen(usd_path).GetExternalReferences() if r.startswith("http")]
print("осталось http-ссылок:", len(remaining))
cams = [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == "Camera"]
meshes = sum(1 for p in stage.Traverse() if p.GetTypeName() == "Mesh")
print("камеры:", cams)
print("мешей в композиции:", meshes)
