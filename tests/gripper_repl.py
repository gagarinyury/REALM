"""Живая сессия симулятора: загрузить сцену один раз, дальше проверять что угодно за секунды.

Зачем. Сцена REALM собирается около четырёх минут, и эта плата берётся ЗАНОВО при каждом
запуске любого теста. Отладка гриппера 10.08.2026 стоила шести таких загрузок и пяти полных
роллаутов по шестнадцать минут — при том, что все заданные вопросы (куда едет сустав, какой
знак у команды, касаются ли колодки предмета, удержится ли он при подъёме) требуют десятков
шагов симуляции, а не восьмисот.

Здесь симулятор поднимается ОДИН раз и остаётся жить, читая команды из файла. Каждая
проверка после этого стоит секунды, и гипотезы можно перебирать десятками, а не по одной
за прогон.

Запуск (на машине с симулятором):
    OMNIGIBSON_HEADLESS=1 python tests/gripper_repl.py &

Управление — записью строки в /tmp/gripper_cmd, ответ появляется в /tmp/gripper_out:
    echo 'close 40'        # сомкнуть, 40 шагов
    echo 'open 40'         # раскрыть
    echo 'state'           # позы всех суставов гриппера, зазор, наклон колодок
    echo 'grasp'           # три условия GRASP по отдельности (env_base.py:205)
    echo 'place'           # поставить предмет задачи между губками
    echo 'lift 0.25'       # поднять руку и проверить, удержался ли предмет
    echo 'joint <имя> <значение>'   # задать сустав напрямую, минуя контроллер
    echo 'quit'
"""

import os
import pathlib
import time

import torch as th

import omnigibson as og
from omnigibson.macros import gm

CMD = pathlib.Path("/tmp/gripper_cmd")
OUT = pathlib.Path("/tmp/gripper_out")


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with OUT.open("a") as f:
        f.write(line + "\n")


def main():
    from realm.environments.env_dynamic import RealmEnvironmentDynamic
    from realm.eval import SUPPORTED_TASKS, set_sim_config

    set_sim_config(rendering_mode="r", robot="DROID")
    with gm.unlocked():
        gm.ENABLE_HQ_RENDERING = False

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    realm_env = RealmEnvironmentDynamic(
        config_path=os.path.join(root, "realm", "config"),
        task_cfg_path=f"REALM_DROID10/{SUPPORTED_TASKS[0]}/default.yaml",
        perturbations=[],
        multi_view=False,
        no_rendering=False,   # рендер нужен для записи видео
        rendering_mode="r",
        robot="DROID",
    )
    env, robot = realm_env.omnigibson_env, realm_env.robot
    env.reset()

    # Запись видео: глазами видно то, чего не видно в числах — например, что губки
    # смыкаются мимо предмета или заваливаются. Отладка 10.08.2026 показала, что
    # визуальный контроль ловит такое быстрее любых метрик.
    frames = []

    # ЖИВАЯ ТРАНСЛЯЦИЯ. Кадры отдаются MJPEG-потоком на localhost пода; наружу он
    # выводится SSH-туннелем (ssh -L 8090:127.0.0.1:8090 ...), поэтому публичный порт
    # у пода не нужен. Смотреть: http://localhost:8090 в браузере.
    import io
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    _latest = {"jpg": None}

    class Stream(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            while True:
                jpg = _latest["jpg"]
                if jpg:
                    try:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                    except Exception:
                        return
                time.sleep(0.02)

    # ThreadingHTTPServer: однопоточный сервер занимался первым же клиентом навсегда,
    # потому что обработчик отдаёт кадры бесконечным циклом.
    threading.Thread(target=lambda: ThreadingHTTPServer(("127.0.0.1", 8090), Stream).serve_forever(),
                     daemon=True).start()

    def publish(img):
        try:
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=80)
            _latest["jpg"] = buf.getvalue()
        except Exception:
            pass

    def grab():
        """Кадр теми же камерами, что видит модель."""
        try:
            from realm.inference.utils import extract_from_obs

            obs = env.get_obs()[0]
            base_im, _, base2, _, wrist_im, _, _ = extract_from_obs(
                obs, robot.name, enable_depth=False,
                gripper_qpos_idx=int(robot.gripper_control_idx[robot.default_arm][0]),
                gripper_qpos_range=(0.0, 0.785),
            )
            import numpy as np

            pair = [im for im in (base_im, wrist_im) if im is not None and im.size]
            if pair:
                h = min(im.shape[0] for im in pair)
                img = np.concatenate([im[:h] for im in pair], axis=1)
                frames.append(img)
                publish(img)      # тот же кадр уходит в живую трансляцию
        except Exception as e:  # кадр не критичен, замер важнее
            say(f"(кадр не снят: {e})")

    from tests.gripper_bench import finger_gap, finger_surface_gap, gripper_joint_positions, pad_tilt

    OUT.write_text("")
    say("СЕССИЯ ГОТОВА. Сцена загружена один раз; дальше каждая команда стоит секунды.")
    say("Трансляция: http://localhost:8090 (через SSH-туннель -L 8090:127.0.0.1:8090)")
    say(f"робот {robot.name}, суставов {len(robot.joints)}, action_dim {robot.action_dim}")

    def hold(cmd_value, steps):
        a = th.zeros(robot.action_dim)
        a[:7] = robot.get_joint_positions()[:7]
        a[-1] = cmd_value
        for i in range(steps):
            env.step(a)
            grab()          # каждый шаг: кадров мало, узкое место — рендер, не передача

    while True:
        if not CMD.exists():
            time.sleep(0.3)
            continue
        line = CMD.read_text().strip()
        CMD.unlink()
        if not line:
            continue
        parts = line.split()
        op = parts[0]
        say(f"\n>>> {line}")

        if op == "quit":
            break
        elif op in ("open", "close"):
            hold(+1 if op == "open" else -1, int(parts[1]) if len(parts) > 1 else 40)
            say(f"суставы: { {k: round(v, 4) for k, v in gripper_joint_positions(robot).items()} }")
            say(f"зазор поверхностей: {finger_surface_gap(robot):.4f} м, наклон колодок: {pad_tilt(robot)}")
        elif op == "state":
            say(f"суставы: { {k: round(v, 4) for k, v in gripper_joint_positions(robot).items()} }")
            say(f"зазор звеньев {finger_gap(robot):.4f} / поверхностей {finger_surface_gap(robot):.4f} м")
            say(f"наклон колодок: {pad_tilt(robot)}")
        elif op == "grasp":
            from omnigibson.utils.usd_utils import RigidContactAPI

            obj = realm_env.main_objects[0]
            obs = env.get_obs()[0]
            touching = [
                bool(RigidContactAPI.is_in_contact(scene_idx=robot.scene.idx, query_set=[f],
                                                   with_set=[obj], ignore_set=None, current_only=True))
                for f in realm_env.robot_finger_links
            ]
            say(f"обе колодки касаются : {sum(touching) == 2}  (касаний {sum(touching)} из 2)")
            say(f"робот касается       : {bool(realm_env.is_touching(obs, obj))}")
            say(f"is_grasping (итог)   : {realm_env.is_grasping(obs, obj)}")
        elif op == "place":
            obj = realm_env.main_objects[0]
            eef = robot.links[robot.eef_link_names[robot.default_arm]]
            pos, orn = eef.get_position_orientation()
            obj.set_position_orientation(position=pos + th.tensor([0.0, 0.0, -0.09]), orientation=orn)
            hold(+1, 10)
            say(f"предмет {obj.name} поставлен между губками, высота {float(obj.get_position_orientation()[0][2]):.4f}")
        elif op == "lift":
            obj = realm_env.main_objects[0]
            d = float(parts[1]) if len(parts) > 1 else 0.25
            h0 = float(obj.get_position_orientation()[0][2])
            up = robot.get_joint_positions()[:7].clone()
            up[1] -= d
            for _ in range(60):
                env.step(th.cat([up, th.tensor([-1.0])]))
            h1 = float(obj.get_position_orientation()[0][2])
            say(f"высота предмета {h0:.4f} -> {h1:.4f} ({h1 - h0:+.4f} м)")
            say("УДЕРЖАН" if h1 - h0 > 0.02 else "ВЫПАЛ или остался на месте")
        elif op == "joint":
            name, val = parts[1], float(parts[2])
            q = robot.get_joint_positions().clone()
            idx = {n: i for i, n in enumerate(robot.joints)}[name]
            q[idx] = val
            robot.set_joint_positions(q)
            for _ in range(5):
                env.step(th.cat([robot.get_joint_positions()[:7], th.tensor([0.0])]))
            say(f"{name} = {float(robot.get_joint_positions()[idx]):.4f}; "
                f"зазор {finger_surface_gap(robot):.4f} м")
        elif op == "video":
            import imageio

            name = parts[1] if len(parts) > 1 else "bench"
            path = f"/workspace/videos_out/{name}.mp4"
            pathlib.Path("/workspace/videos_out").mkdir(exist_ok=True)
            if frames:
                imageio.mimsave(path, frames, fps=10)
                say(f"видео записано: {path}, кадров {len(frames)}")
                frames.clear()
            else:
                say("кадров нет — включи рендер (no_rendering=False)")
        else:
            say(f"неизвестная команда: {op}")

    og.shutdown()


if __name__ == "__main__":
    main()
