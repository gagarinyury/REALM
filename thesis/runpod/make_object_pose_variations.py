"""Мини-бенчмарк с вариациями позы объекта — пункт 4 задания.

Бланк просит «simple task variations (e.g. object or camera pose)». Вариацию позы
камеры даёт сам бенчмарк (HardBench рандомизирует камеру запястья, DroidMini нет),
а вариации позы объекта в готовых наборах нет — её строим здесь.

Берём один эпизод DroidMini (тот, где π0-FAST показал лучший результат: картошка,
прогрессия 0.4) и создаём копии со сдвигом предмета по горизонтали. Сдвигаем и
стартовую, и целевую позу на один и тот же вектор, иначе изменилась бы сама задача,
а не только положение предмета.

Смещения по 3 и 6 см: заметно для политики, но предмет остаётся на той же
поверхности. Вертикаль не трогаем, чтобы объект не повис в воздухе и не провалился.
"""

import json
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/workspace/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/"
    "procthor-10k/FrankaPickandPlaceDroidMiniBench/"
    "FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark/benchmark.json"
)
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/workspace/bench_objpose")

# (метка, сдвиг по x, сдвиг по y) в метрах
OFFSETS = [
    ("base", 0.00, 0.00),
    ("x+3cm", 0.03, 0.00),
    ("x-3cm", -0.03, 0.00),
    ("y+3cm", 0.00, 0.03),
    ("y-3cm", 0.00, -0.03),
    ("x+6cm", 0.06, 0.00),
    ("y+6cm", 0.00, 0.06),
]

episodes = json.load(open(SRC))
base = episodes[0]  # house_0, «pick up the brown potato», лучший результат π0-FAST

out = []
for label, dx, dy in OFFSETS:
    ep = json.loads(json.dumps(base))  # глубокая копия
    for key in ("pickup_obj_start_pose", "pickup_obj_goal_pose"):
        pose = list(ep["task"][key])
        pose[0] += dx
        pose[1] += dy
        ep["task"][key] = pose
    out.append(ep)
    print(f"{label:8} сдвиг ({dx:+.2f}, {dy:+.2f}) -> "
          f"старт {[round(v, 3) for v in ep['task']['pickup_obj_start_pose'][:3]]}")

DST.mkdir(parents=True, exist_ok=True)
json.dump(out, open(DST / "benchmark.json", "w"))
meta = SRC.parent / "benchmark_metadata.json"
if meta.exists():
    (DST / "benchmark_metadata.json").write_text(meta.read_text())
print(f"\nзаписано {len(out)} эпизодов в {DST}")
print("порядок эпизодов соответствует списку OFFSETS выше")
