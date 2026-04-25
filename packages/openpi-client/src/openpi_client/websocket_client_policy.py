import logging
import time
from typing import Dict, Tuple

import websockets.sync.client
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000, recv_timeout: float = 300.0) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._recv_timeout = recv_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _connect(self) -> websockets.sync.client.ClientConnection:
        kwargs = dict(
            compression=None,
            max_size=None,
            open_timeout=60,
            close_timeout=30,
        )
        try:
            return websockets.sync.client.connect(self._uri, ping_timeout=120, **kwargs)
        except TypeError:
            # Older websockets versions don't support ping_timeout on the sync client
            return websockets.sync.client.connect(self._uri, **kwargs)

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = self._connect()
                metadata = msgpack_numpy.unpackb(conn.recv(timeout=self._recv_timeout))
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def _reconnect(self) -> None:
        logging.info(f"Reconnecting to server at {self._uri}...")
        try:
            self._ws.close()
        except Exception:
            pass
        self._ws, self._server_metadata = self._wait_for_server()
        logging.info("Reconnected successfully.")

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv(timeout=self._recv_timeout)
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
