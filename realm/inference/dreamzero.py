import uuid
import time
import omnigibson as og


class DreamZeroClient:
    """
    Client for the DreamZero server.

    Wraps WebsocketClientPolicy with automatic reconnection on timeout.
    DreamZero's video diffusion model can hang mid-inference (CUDA/distributed
    rank deadlock). When that happens the websocket recv() would block forever.
    This client detects the timeout and reconnects transparently.
    """

    MAX_INFER_RETRIES = 3
    RETRY_BACKOFF = 5  # seconds between reconnection attempts

    def __init__(self, host="localhost", port=5000, recv_timeout=300.0):
        self._host = host
        self._port = port
        self._recv_timeout = recv_timeout
        self.session_id = str(uuid.uuid4())
        self._connect()

    def _connect(self):
        try:
            from eval_utils.policy_client import WebsocketClientPolicy
        except ImportError:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

        og.log.info(f"Connecting to DreamZero server at {self._host}:{self._port}...")
        self.client = WebsocketClientPolicy(
            host=self._host, port=self._port, recv_timeout=self._recv_timeout
        )

        try:
            metadata = self.client.get_server_metadata()
            og.log.info(f"Connected to DreamZero! Server metadata: {metadata}")
        except Exception as e:
            og.log.info(f"Warning: Could not fetch DreamZero metadata: {e}")

    def _reconnect(self):
        og.log.info("DreamZero: reconnecting after timeout...")
        try:
            self.client._reconnect()
            og.log.info("DreamZero: reconnected via websocket reconnect.")
        except Exception:
            og.log.info("DreamZero: websocket reconnect failed, rebuilding client...")
            self._connect()
        # New session after reconnect — server state is stale.
        self.session_id = str(uuid.uuid4())

    def infer(self, obs_dict):
        obs_dict["session_id"] = self.session_id
        obs_dict["endpoint"] = "infer"

        for attempt in range(1, self.MAX_INFER_RETRIES + 1):
            try:
                return self.client.infer(obs_dict)
            except TimeoutError:
                og.log.info(
                    f"DreamZero: inference timed out (attempt {attempt}/{self.MAX_INFER_RETRIES}). "
                    f"Server may be hung."
                )
                if attempt < self.MAX_INFER_RETRIES:
                    time.sleep(self.RETRY_BACKOFF)
                    self._reconnect()
                    obs_dict["session_id"] = self.session_id
                else:
                    raise TimeoutError(
                        f"DreamZero inference timed out after {self.MAX_INFER_RETRIES} attempts. "
                        f"Server at {self._host}:{self._port} is likely hung."
                    )

    def reset(self):
        """Tells the server to flush buffers and saves the generated video prediction to disk"""
        reset_obs = {"session_id": self.session_id}

        try:
            import inspect
            if hasattr(self.client, "reset"):
                sig = inspect.signature(self.client.reset)
                if len(sig.parameters) > 0:
                    self.client.reset(reset_obs)
                else:
                    self.client.reset()
            else:
                reset_obs["endpoint"] = "reset"
                self.client.infer(reset_obs)
        except Exception as e:
            og.log.info(f"Warning: DreamZero reset failed: {e}")

        self.session_id = str(uuid.uuid4())
