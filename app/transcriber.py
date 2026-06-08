import logging
import os
import tempfile
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_model: WhisperModel | None = None


def load_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model
    model_id = os.environ.get("MODEL_SIZE", "ivrit-ai/whisper-large-v3-turbo")
    device = os.environ.get("DEVICE", "cuda")
    compute_type = os.environ.get("COMPUTE_TYPE", "int8_float16")
    logger.info({"event": "model_loading", "model": model_id, "device": device})
    _model = WhisperModel(model_id, device=device, compute_type=compute_type)
    logger.info({"event": "model_ready", "model": model_id})
    return _model


def transcribe(
    audio_bytes: bytes,
    language: str = "he",
    initial_prompt: str | None = None,
) -> dict:
    model = load_model()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            initial_prompt=initial_prompt or None,
            beam_size=5,
            vad_filter=True,
        )
        text = "".join(seg.text for seg in segments)
        return {
            "text": text.strip(),
            "language": info.language,
            "duration": round(info.duration, 2),
        }
    finally:
        os.unlink(tmp_path)
