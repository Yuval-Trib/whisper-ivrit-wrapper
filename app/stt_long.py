import base64
import logging
import os
import tempfile

import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from app.base import BaseHandler

logger = logging.getLogger(__name__)

_MAX_AUDIO_BYTES = 25 * 1024 * 1024


class STTLong(BaseHandler):

    def __init__(self) -> None:
        self._model: WhisperModel | None = None
        self._diarizer: Pipeline | None = None

    def load_model(self) -> None:
        model_id = os.environ.get("MODEL_SIZE", "ivrit-ai/whisper-large-v3-turbo")
        device = os.environ.get("DEVICE", "cuda")
        compute_type = os.environ.get("COMPUTE_TYPE", "int8_float16")
        logger.info({"event": "model_loading", "model": model_id, "device": device, "mode": "stt_long"})
        self._model = WhisperModel(model_id, device=device, compute_type=compute_type)

        logger.info({"event": "diarizer_loading"})
        self._diarizer = Pipeline.from_pretrained(
            "ivrit-ai/pyannote-speaker-diarization-3.1",
        )
        self._diarizer.to(torch.device(device))
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
            return {"error": "audio exceeds 25 MB limit"}

        language = job_input.get("language", "he")
        initial_prompt = job_input.get("initial_prompt") or None
        num_speakers = job_input.get("num_speakers", 2)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, info = self._model.transcribe(
                tmp_path,
                language=language,
                initial_prompt=initial_prompt,
                beam_size=5,
                vad_filter=True,
            )
            segments = list(segments)

            diarization = self._diarizer(tmp_path, num_speakers=num_speakers)
        finally:
            os.unlink(tmp_path)

        return {
            "segments": self._merge(segments, diarization),
            "language": info.language,
            "duration": round(info.duration, 2),
        }

    def _merge(self, whisper_segments, diarization) -> list[dict]:
        speaker_map: dict[str, str] = {}

        def label(speaker_id: str) -> str:
            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = chr(ord("A") + len(speaker_map))
            return speaker_map[speaker_id]

        def find_speaker(start: float, end: float) -> str:
            mid = (start + end) / 2
            for turn, _, speaker_id in diarization.itertracks(yield_label=True):
                if turn.start <= mid <= turn.end:
                    return label(speaker_id)
            return "UNKNOWN"

        return [
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "speaker": find_speaker(seg.start, seg.end),
            }
            for seg in whisper_segments
            if seg.text.strip()
        ]
