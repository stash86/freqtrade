"""
This module manages webhook communication
"""

import logging
import time
from typing import Any

from requests import RequestException, post

from freqtrade.constants import Config
from freqtrade.enums import RPCMessageType
from freqtrade.rpc import RPC, RPCHandler
from freqtrade.rpc.rpc_types import RPCSendMsg


logger = logging.getLogger(__name__)

logger.debug("Included module rpc.webhook ...")


class Webhook(RPCHandler):
    """This class handles all webhook communication"""

    def __init__(self, rpc: RPC, config: Config) -> None:
        """
        Init the Webhook class, and init the super class RPCHandler
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        super().__init__(rpc, config)

        self._url = self._config["webhook"]["url"]
        self._format = self._config["webhook"].get("format", "form")
        self._retries = self._config["webhook"].get("retries", 0)
        self._retry_delay = self._config["webhook"].get("retry_delay", 0.1)
        self._timeout = self._config["webhook"].get("timeout", 10)

    def cleanup(self) -> None:
        """
        Cleanup pending module resources.
        This will do nothing for webhooks, they will simply not be called anymore
        """
        pass

    def _get_value_dict(self, msg: RPCSendMsg) -> dict[str, Any] | None:
        whconfig = self._config["webhook"]
        if msg["type"].value in whconfig:
            # Explicit types should have priority
            valuedict = whconfig.get(msg["type"].value)
        # Deprecated 2022.10 - only keep generic method.
        elif msg["type"] in [RPCMessageType.ENTRY]:
            valuedict = whconfig.get("webhookentry")
        elif msg["type"] in [RPCMessageType.ENTRY_CANCEL]:
            valuedict = whconfig.get("webhookentrycancel")
        elif msg["type"] in [RPCMessageType.ENTRY_FILL]:
            valuedict = whconfig.get("webhookentryfill")
        elif msg["type"] == RPCMessageType.EXIT:
            valuedict = whconfig.get("webhookexit")
        elif msg["type"] == RPCMessageType.EXIT_FILL:
            valuedict = whconfig.get("webhookexitfill")
        elif msg["type"] == RPCMessageType.EXIT_CANCEL:
            valuedict = whconfig.get("webhookexitcancel")
        elif msg["type"] in (
            RPCMessageType.STATUS,
            RPCMessageType.STARTUP,
            RPCMessageType.EXCEPTION,
            RPCMessageType.WARNING,
        ):
            valuedict = whconfig.get("webhookstatus")
        elif msg["type"] in (
            RPCMessageType.PROTECTION_TRIGGER,
            RPCMessageType.PROTECTION_TRIGGER_GLOBAL,
            RPCMessageType.WHITELIST,
            RPCMessageType.ANALYZED_DF,
            RPCMessageType.NEW_CANDLE,
            RPCMessageType.STRATEGY_MSG,
        ):
            # Don't fail for non-implemented types
            return None
        return valuedict

    def send_msg(self, msg: RPCSendMsg) -> None:
        """Send a message to telegram channel"""
        try:
            valuedict = self._get_value_dict(msg)

            if not valuedict:
                logger.debug("Message type '%s' not configured for webhooks", msg["type"])
                return

            payload = {key: value.format(**msg) for (key, value) in valuedict.items()}
            self._send_msg(payload)
        except KeyError as exc:
            logger.exception(
                "Problem calling Webhook. Please check your webhook configuration. Exception: %s",
                exc,
            )

    def _send_msg(self, payload: dict) -> None:
        """do the actual call to the webhook"""

        success = False
        attempts = 0
        while not success and attempts <= self._retries:
            if attempts:
                if self._retry_delay:
                    time.sleep(self._retry_delay)
                logger.info("Retrying webhook...")

            attempts += 1

            try:
                if isinstance(self._url, list):
                    for url in self._url:
                        response = self._post_msg(url, payload, self._timeout)
                else:
                    response = self._post_msg(self._url, payload, self._timeout)
                    # raise NotImplementedError(f'Unknown format: {self._format}')

                # Throw a RequestException if the post was not successful
                response.raise_for_status()
                success = True

            except RequestException as exc:
                logger.warning("Could not call webhook url. Exception: %s", exc)

    def _post_msg(self, url, payload, timeout):
        if self._format == "form":
            return post(url, data=payload, timeout=timeout)
        elif self._format == "json":
            return post(url, json=payload, timeout=timeout)
        elif self._format == "raw":
            return post(
                url, data=payload["data"], headers={"Content-Type": "text/plain"}, timeout=timeout
            )
        else:
            raise NotImplementedError(f"Unknown format: {self._format}")
