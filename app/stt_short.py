import base64
import logging
import os
import tempfile

from faster_whisper import WhisperModel

from app.base import BaseHandler

logger = logging.getLogger(__name__)

_MAX_AUDIO_BYTES = 5 * 1024 * 1024


class STTShort(BaseHandler):

    def __init__(self) -> None:
        self._model: WhisperModel | None = None

    def load_model(self) -> None:
        model_id = os.environ.get("MODEL_SIZE", "ivrit-ai/whisper-large-v3-turbo-ct2")
        device = os.environ.get("DEVICE", "cuda")
        compute_type = os.environ.get("COMPUTE_TYPE", "int8_float16")
        logger.info({"event": "model_loading", "model": model_id, "device": device, "mode": "stt_short"})
        self._model = WhisperModel(model_id, device=device, compute_type=compute_type)
        logger.info({"event": "model_ready", "model": model_id})

    def handle(self, job_input: dict) -> dict:
        audio_b64 = job_input.get("audio")
        if not audio_b64:
            return {"error": "audio is required"}

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            return {"error": "invalid audio encoding"}

        if not audio_bytes:
            return {"error": "empty audio"}

        if len(audio_bytes) > _MAX_AUDIO_BYTES:
            return {"error": "audio exceeds 5 MB limit"}

        language = job_input.get("language", "he")
        initial_prompt = job_input.get("initial_prompt") or None

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, info = self._model.transcribe(
                tmp_path,
                language=language,
                initial_prompt=initial_prompt,
                beam_size=1,
                vad_filter=False,
            )
            text = "".join(seg.text for seg in segments)
            return {
                "text": text.strip(),
                "language": info.language,
                "duration": round(info.duration, 2),
            }
        finally:
            os.unlink(tmp_path)
