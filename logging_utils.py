from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

try:
    from pythonjsonlogger import jsonlogger  # type: ignore

    _JSON_LOG = True
except Exception:
    jsonlogger = None
    _JSON_LOG = False


class RedactingFormatter(logging.Formatter):
    _key_re = re.compile(
        r"(client_(?:id|secret)|youtube_api_key|spotify_client_id|spotify_client_secret)\s*[:=]\s*([^\s,'\"]+)",
        re.IGNORECASE,
    )
    _token_re = re.compile(r"[A-Za-z0-9_-]{24,}")

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        msg = self._key_re.sub(r"\1=***REDACTED***", msg)
        msg = self._token_re.sub(
            lambda m: "***REDACTED***" if len(m.group(0)) >= 32 else m.group(0),
            msg,
        )
        return msg


def configure_logging(verbose: bool, log_file: Optional[str]) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)

    if _JSON_LOG and jsonlogger:
        base_formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s")
    else:
        base_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    formatter = RedactingFormatter(base_formatter._fmt)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file:
        fh = RotatingFileHandler(
            log_file,
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        root.addHandler(fh)
