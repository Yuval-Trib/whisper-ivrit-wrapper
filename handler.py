import importlib
import logging
import os
import time
import uuid

import runpod

from app.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

_REGISTRY = {
    "stt_short": ("app.stt_short", "STTShort"),
    "stt_long": ("app.stt_long", "STTLong"),
    "tts": ("app.tts", "TTS"),
}

_MODE = os.environ.get("MODE")
if not _MODE:
    raise RuntimeError("MODE environment variable is required")
if _MODE not in _REGISTRY:
    raise RuntimeError(f"Unknown MODE: {_MODE!r}. Valid values: {list(_REGISTRY)}")

_module_path, _class_name = _REGISTRY[_MODE]
_endpoint = getattr(importlib.import_module(_module_path), _class_name)()
_endpoint.load_model()


def handler(job: dict) -> dict:
    request_id = str(uuid.uuid4())
    t0 = time.monotonic()

    try:
        result = _endpoint.dispatch(job.get("input", {}))
        logger.info({
            "event": "request_ok",
            "request_id": request_id,
            "mode": _MODE,
            "processing_ms": round((time.monotonic() - t0) * 1000),
        })
        return result
    except Exception as exc:
        logger.error({
            "event": "request_error",
            "request_id": request_id,
            "mode": _MODE,
            "error_class": type(exc).__name__,
            "processing_ms": round((time.monotonic() - t0) * 1000),
        }, exc_info=True)
        return {"error": "internal error"}


runpod.serverless.start({"handler": handler})
