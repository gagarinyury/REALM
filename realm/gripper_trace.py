"""Трассировка гриппера: каждое движение губки и чем оно вызвано.

Зачем. Штатный лог REALM пишет ровно то, что уходит модели: семь суставов руки и один
обобщённый канал гриппера. Позы отдельных губок, ведомых суставов, зазор между пальцами
и контакты с объектом не логируются никуда. Поэтому вопросы вида «губки болтаются
влево-вправо» или «они смыкаются на предмете или мимо» проверялись чтением бинарного
ассета и двадцатиминутными прогонами, вместо того чтобы читаться из таблицы.

Здесь на каждом шаге записывается вся цепочка от намерения модели до физического
результата:

    что модель ВИДЕЛА      gripper_state      нормированное 0..1, 0 = раскрыт
    что СКОМАНДОВАЛА       cmd_raw            сырой последний канал действия
    что ушло в контроллер  cmd_binarized      после `1 if a > 0.5 else -1` (eval.py)
    как ИСТОЛКОВАНО        interpreted        "сомкнуть"/"разжать" с учётом inverted
    куда ПОЕХАЛО           q_<сустав>         поза каждого сустава гриппера, рад
    что ПОЛУЧИЛОСЬ         finger_gap         расстояние между пальцевыми линками, м
    коснулись ли           contact_left/right контакт губки с объектом задачи

Именно эти столбцы, поставленные рядом, отвечают на вопрос «чем мотивировано движение»:
видно, совпадает ли направление хода с намерением модели, и удерживается ли предмет.

Использование (в eval.py):

    tracer = GripperTracer(env.robot, task_object=env.task_object)
    ...
    tracer.record(gripper_state, action[-1], new_action[-1])
    ...
    tracer.to_parquet(log_dir / "gripper_trace" / f"{task}.parquet")
"""

import numpy as np


class GripperTracer:
    """Собирает пошаговую трассировку гриппера. Ничего не стоит, если не вызывать record()."""

    def __init__(self, robot, task_object=None, inverted=None):
        self.robot = robot
        self.task_object = task_object
        # inverted определяет, как контроллер понимает знак команды: см. robot.py,
        # "inverted": grasping_direction == "upper". Берём с самого робота, а не угадываем.
        self.inverted = (
            inverted if inverted is not None else getattr(robot, "_grasping_direction", "lower") == "upper"
        )
        self.joint_names = [n for n in robot.joints if "knuckle" in n or "finger" in n]
        self.finger_links = [
            l for name, l in robot.links.items() if "inner_finger" in name and "knuckle" not in name
        ]
        self.rows = []

    def _finger_gap(self):
        if len(self.finger_links) < 2:
            return float("nan")
        a = self.finger_links[0].get_position_orientation()[0]
        b = self.finger_links[1].get_position_orientation()[0]
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def _contacts(self):
        """Касается ли каждая губка объекта задачи."""
        if self.task_object is None:
            return [None] * len(self.finger_links)
        out = []
        for link in self.finger_links:
            try:
                bodies = link.states  # наличие контакта берём из состояния линка
                out.append(bool(self.task_object in getattr(link, "contact_list", lambda: [])()))
            except Exception:
                out.append(None)
        return out

    def record(self, gripper_state, cmd_raw, cmd_binarized):
        """Один шаг. Вызывать сразу после того, как действие отправлено в env.step()."""
        qpos = self.robot.get_joint_positions()
        idx = {n: i for i, n in enumerate(self.robot.joints)}

        # Как контроллер истолкует знак: при inverted команда переворачивается
        # (_preprocess_command), поэтому "сомкнуть" и "разжать" меняются местами.
        opens = (cmd_binarized > 0) if not self.inverted else (cmd_binarized < 0)
        interpreted = "к верхнему пределу (сомкнуть)" if opens else "к нижнему пределу (разжать)"

        row = {
            "gripper_state_seen": float(gripper_state),
            "cmd_raw": float(cmd_raw),
            "cmd_binarized": float(cmd_binarized),
            "interpreted": interpreted,
            "finger_gap": self._finger_gap(),
        }
        for n in self.joint_names:
            row[f"q_{n}"] = float(qpos[idx[n]])
        contacts = self._contacts()
        for i, c in enumerate(contacts):
            row[f"contact_finger_{i}"] = c
        self.rows.append(row)

    def summary(self):
        """Короткая сводка: согласовано ли намерение модели с направлением хода."""
        if len(self.rows) < 2:
            return "недостаточно данных"
        import pandas as pd

        df = pd.DataFrame(self.rows)
        lead = [c for c in df.columns if c.startswith("q_") and "outer_knuckle" in c]
        if not lead:
            return "не найден ведущий сустав"
        q = df[lead[0]].to_numpy()
        d = np.diff(q)
        want_close = df["cmd_binarized"].to_numpy()[:-1] > 0  # >0 = модель просит сомкнуть
        moved = np.abs(d) > 1e-4
        agree = np.sum(moved & (want_close == (d > 0)))
        disagree = np.sum(moved & (want_close != (d > 0)))
        lines = [
            f"шагов с движением ведущего сустава: {int(moved.sum())}",
            f"  ход СОВПАДАЕТ с намерением модели: {int(agree)}",
            f"  ход ПРОТИВОПОЛОЖЕН намерению:      {int(disagree)}",
            f"зазор между губками: мин {df.finger_gap.min():.4f} м, макс {df.finger_gap.max():.4f} м",
        ]
        if disagree > agree:
            lines.append("ВЫВОД: направление захвата инвертировано — команда даёт обратный ход")
        return "\n".join(lines)

    def to_parquet(self, path):
        import pandas as pd

        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_parquet(path)
        return path
