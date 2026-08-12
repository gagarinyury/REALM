import time
import numpy as np
from openpi_client import websocket_client_policy as w

c = w.WebsocketClientPolicy(host="127.0.0.1", port=8000)
print("подключено, метаданные:", c.get_server_metadata(), flush=True)

obs = {
    "observation/exterior_image_1_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/joint_position": np.array([0.0, -0.2, 0.0, -2.0, 0.0, 1.8, 0.7], dtype=np.float32),
    "observation/gripper_position": np.array([0.0], dtype=np.float32),
    "prompt": "put the green block in the bowl",
}
print("отправляю запрос...", flush=True)
t = time.time()
a = np.asarray(c.infer(obs)["actions"])
print("ответ за %.1f с, форма %s" % (time.time() - t, a.shape), flush=True)
print("std по шагам:", a.std(axis=0).round(5), flush=True)
print("первый шаг:", a[0].round(4), flush=True)
