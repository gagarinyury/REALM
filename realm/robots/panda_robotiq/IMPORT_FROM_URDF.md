# Импорт DROID из URDF — штатный путь вместо ремонта бинарного ассета

Сделано 10.08.2026. Заменяет собой три скрипта ремонта (`untangle_droid_gripper.py`,
`align_droid_root.py`, `lift_gripper_velocity_clamp.py`) и всю конструкцию на mimic-связях.

## Почему

Документация OmniGibson (`docs/tutorials/custom_robot_import.md`, таблица чистильщиков URDF)
предписывает прямо противоположное тому, что делает наш ремонт ассета:

> `strip_mimic_joints(tree)` — удаляет все `<mimic>`. Импортёр **не создаёт привод** на
> mimic-суставе, из-за чего ведомый палец остаётся неприводным.
> **Пусть оба пальца ведёт контроллер во время работы.**

Наш ремонт строил гриппер именно на mimic-связях, и наблюдалось ровно то, о чём
предупреждает документация: ведомые суставы не дотягивают за ведущими (0.55 против 0.785),
пружинят, расходятся в противофазе, губки встают под углом ~90° друг к другу.

Проверено отдельно: вернуть авторскую схему (замкнутый параллелограмм + гриппер вне
артикуляции через `physics:excludeFromArticulation`) на OmniGibson 3.9.1 **невозможно** —
после отключения всех пяти мешающих проверок движка процесс падает в нативном коде
(`Segmentation fault`, без единого Python-кадра). То есть разрыв петли на новой версии
неизбежен, вопрос только в том, чем заменить механическую связь.

## Рецепт

```bash
# 1. Очистить URDF: убрать mimic-связи
python - <<'PY'
import xml.etree.ElementTree as ET
from omnigibson.utils.urdf_preprocessing import strip_mimic_joints, urdf_audit
t = ET.parse('realm/robots/panda_robotiq/panda_robotiq_85.urdf')
strip_mimic_joints(t)                      # -> 5 связей удалено, пальцы становятся приводными
t.write('panda_robotiq_85_clean.urdf', encoding='utf-8', xml_declaration=True)
PY

# 2. Достать геометрию гриппера (в репозитории её нет — только описание)
#    Источник: ros-industrial/robotiq, ветка kinetic-devel
#    Соответствие имён (наш URDF -> ros-industrial):
#      base -> base_link,  driver -> outer_knuckle,  coupler -> outer_finger,
#      follower -> inner_finger,  spring_link -> inner_knuckle,  pad -> pad

# 3. Перевести геометрию в .obj — конвертер Isaac Sim падает на .dae от Robotiq
python -c "import trimesh; [trimesh.load(f, force='mesh', process=False).export(f[:-4]+'.obj') for f in glob.glob('meshes/robotiq-2f/*/*.dae')]"

# 4. Абсолютные пути к мешам + инерция основанию (у panda_link0 её нет в URDF)

# 5. Импорт
python omnigibson/examples/robots/import_custom_robot.py --config droid_import.yaml
```

## Три причины, по которым импорт падает (все встречены и устранены)

| симптом | причина | лечение |
|---|---|---|
| `ValueError: string is not a file: .../link0.dae` | относительные пути к мешам | абсолютные пути |
| `No mass specified for link panda_link0`, затем segfault | у основания нет `<inertial>` — вырожденное тело | добавить массу и тензор инерции |
| segfault внутри `isaacsim.asset.importer.urdf` | конвертер не переваривает `.dae` от Robotiq | перевести геометрию в `.obj` |

Штатный аудит (`urdf_audit`) показывает первые две: поля `broken_relative_paths` и
`links_without_inertial`. Третья не диагностируется ничем — только падением.

## Результат

```
вращательных суставов 13, из них гриппера 6 — У ВСЕХ привод
   ведущие  robotiq_2f_85_{left,right}_driver_joint       0 .. 47.8 град
   колодки  robotiq_2f_85_{left,right}_follower_joint  -170 .. 170
   колена   robotiq_2f_85_{left,right}_spring_link_joint -170 .. 170
камера запястья gripper_link_camera на panda_link7 — ракурс сохранён из droid.usd
```

Управление гриппером сводится к таблице чисел в конфиге задачи
(`realm/config/robots/DROID2.yaml`), взятых из спецификации 2F-85, а не подобранных:

```
driver      (ведущий)  ход 0..0.834 рад,  множитель +1
spring_link (колено)                      множитель +1   -> вместе с ведущим
follower    (колодка)                     множитель -1   -> против ведущего
```

Зеркальность левой и правой половин в настоящем 2F-85 задана **посадочным разворотом на
180 град** (`rpy="0 0 3.14159"` у левой), а не противоположными знаками — множители у
обеих сторон одинаковые. Это тот факт, на подмене которого сломалась предыдущая
конструкция: в ремонтном ассете левая и правая половины получили разные посадочные
повороты (отличающиеся примерно на 90 град), из-за чего губки вставали под прямым углом.
