#!/usr/bin/env python3
"""Проверка на немую политику — минута работы, симулятор не нужен.

FASTTokenizer.extract_actions (openpi/models/tokenizer.py) при отсутствии маркера
"Action: " в сгенерированном тексте молча возвращает np.zeros. Дальше AbsoluteActions
прибавляет текущее состояние, и ответ выглядит как «почти твоя нынешняя поза» —
робот правдоподобно дрейфует, метрики и видео пишутся, а политика при этом молчит.

Два признака: (1) нулевой разброс внутри чанка, (2) побитово тот же ответ на
случайный шум, что и на осмысленное наблюдение.

    micromamba run -n omnigibson python probe_policy.py [порт]
"""
import sys
import numpy as np
from openpi_client import websocket_client_policy

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


def observation(seed):
    rng = np.random.default_rng(seed)
    return {
        "observation/exterior_image_1_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.array([0.0, -0.2, 0.0, -2.0, 0.0, 1.8, 0.7], dtype=np.float32),
        "observation/gripper_position": np.array([0.0], dtype=np.float32),
        "prompt": "put the green block in the bowl",
    }


client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=PORT)
a = np.asarray(client.infer(observation(1))["actions"])
b = np.asarray(client.infer(observation(2))["actions"])

print(f"форма чанка: {a.shape}")
print(f"разброс внутри чанка (std по шагам): {a.std(axis=0).round(5)}")
print(f"первый шаг: {a[0].round(4)}")
print(f"последний шаг: {a[-1].round(4)}")

same = np.allclose(a, b, atol=1e-6)
flat = a.std(axis=0).max() < 1e-6
print(f"\nответ на два РАЗНЫХ наблюдения совпадает: {same}")
print(f"чанк постоянен (нулевой разброс): {flat}")
print("\nВЕРДИКТ:", "ПОЛИТИКА МОЛЧИТ — дальше запускать бессмысленно" if (same or flat)
      else "политика отвечает осмысленно, можно запускать симулятор")
