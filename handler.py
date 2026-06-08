import base64
import logging
import time
import uuid

import runpod

from app.logging_config import configure_logging
from app.transcriber import load_model, transcribe

configure_logging()
logger = logging.getLogger(__name__)

load_model()


def handler(job: dict) -> dict:
    request_id = str(uuid.uuid4())
    t0 = time.monotonic()
    inp = job.get("input", {})

    audio_b64 = inp.get("audio")
    if not audio_b64:
        return {"error": "audio is required"}

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        return {"error": "invalid audio encoding"}

    if not audio_bytes:
        return {"error": "empty audio"}

    if len(audio_bytes) > 25 * 1024 * 1024:
        return {"error": "audio exceeds 25 MB limit"}

    language = inp.get("language", "he")
    initial_prompt = inp.get("initial_prompt")

    try:
        result = transcribe(audio_bytes, language=language, initial_prompt=initial_prompt)
        logger.info({
            "event": "transcription_ok",
            "request_id": request_id,
            "audio_duration": result["duration"],
            "processing_ms": round((time.monotonic() - t0) * 1000),
            "language": result["language"],
        })
        return result
    except Exception as exc:
        logger.error({
            "event": "transcription_error",
            "request_id": request_id,
            "error_class": type(exc).__name__,
            "processing_ms": round((time.monotonic() - t0) * 1000),
        }, exc_info=True)
        return {"error": "transcription failed"}


runpod.serverless.start({"handler": handler})
