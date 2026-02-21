import numpy as np
from PIL import Image
from openpi_client import websocket_client_policy, image_tools
from gr00t_client import server_client
import omnigibson as og


def extract_from_obs(obs: dict):
    base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
    base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
    wrist_im = obs['franka']['franka:gripper_link_camera:Camera:0']['rgb'].cpu().numpy()[..., :3]
    proprio = obs['franka']['proprio'].cpu().numpy()
    robot_state = proprio[:7]
    gripper_state = proprio[7] / 0.05  # 0 = open, 0.05 = closed
    return base_im, base_im_second, wrist_im, robot_state, gripper_state


class InferenceClient:
    def __init__(self, model_type, port, host="localhost"):
        self.model_type = model_type
        self.client = None
        if model_type == "debug":
            pass
        elif model_type == "groot_n16":
            og.log.info("Connecting to GR00T N16 server...")
            self.client = server_client.PolicyClient(host=host, port=port)
            og.log.info("Connected!")
        else:
             og.log.info("Connecting to server...")
             self.client = websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port
            )
             og.log.info("Connected!")

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state, use_base_im_second=False):
        if self.model_type == "debug":
            pred_action_chunk = np.atleast_1d(np.zeros(8))
            return pred_action_chunk

        if self.model_type == "groot_n16":
            base_im_resized = np.asarray(Image.fromarray(base_im).resize((320, 180))).astype(np.uint8)
            wrist_im_resized = np.asarray(Image.fromarray(wrist_im).resize((320, 180))).astype(np.uint8)

            obs_dict = {
                "video.exterior_image_1_left": base_im_resized[None, None],  # (1, 1, H, W, 3)
                "video.wrist_image_left": wrist_im_resized[None, None],  # (1, 1, H, W, 3)
                "state.joint_position": np.array(robot_state).astype(np.float32).reshape(1, 1, 7),
                "state.gripper_position": np.atleast_1d(np.array(gripper_state)).astype(np.float32).reshape(1, 1, 1),
                "annotation.language.language_instruction": [instruction],
            }
            pred, _ = self.client.get_action(obs_dict)
            pred_action_chunk = np.concatenate(
                [pred["action.joint_position"],
                 pred["action.gripper_position"]], axis=-1)  # (1, 32, 8)
            return pred_action_chunk[0]  # (32, 8)

        if self.model_type == "GR00T":
            base_im_resized = np.asarray(Image.fromarray(base_im).resize((320, 180))).astype(np.uint8)
            base_im_second_resized = np.asarray(Image.fromarray(base_im_second).resize((320, 180))).astype(np.uint8)
            wrist_im_resized = np.asarray(Image.fromarray(wrist_im).resize((320, 180))).astype(np.uint8)

            obs_dict = {
                "prompt": [instruction],
                "state.joint_position": np.array(robot_state).astype(np.float32).reshape(1, 7),
                "state.gripper_position": np.atleast_1d(np.array(gripper_state)).astype(np.float32).reshape(1, 1),
                "video.exterior_image_1": base_im_resized[None],
                "video.exterior_image_2": base_im_second_resized[None],
                "video.wrist_image": wrist_im_resized[None]
            }
            pred = self.client.infer(obs_dict)
            pred_action_chunk = np.concatenate(
                [pred["action.joint_position"],
                 pred["action.gripper_position"].reshape(-1, 1)], axis=-1)
            return pred_action_chunk
        else:
            img_to_use = base_im_second if use_base_im_second else base_im

            obs_dict = {
                "prompt": instruction,
                "observation/joint_position": robot_state,
                "observation/gripper_position": np.atleast_1d(np.array(gripper_state)),
                "observation/exterior_image_1_left": image_tools.resize_with_pad(img_to_use, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(wrist_im, 224, 224)
            }
            pred = self.client.infer(obs_dict)
            pred_action_chunk = pred["actions"]
            return pred_action_chunk
