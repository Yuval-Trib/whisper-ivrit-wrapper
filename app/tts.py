import base64
import io
import logging
import os

import soundfile as sf
from blue_onnx import BlueTTS
from huggingface_hub import hf_hub_download, snapshot_download

from app.base import BaseHandler

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 5000
_MODEL_REPO = "notmax123/blue-onnx-v2"
_RENIKUD_REPO = "thewh1teagle/renikud"


class TTS(BaseHandler):

    def __init__(self) -> None:
        self._model: BlueTTS | None = None
        self._steps = int(os.environ.get("TTS_STEPS", 8))
        self._speed = float(os.environ.get("TTS_SPEED", 0.95))

    def load_model(self) -> None:
        hf_home = os.environ.get("HF_HOME", "/runpod-volume/huggingface")
        model_dir = os.path.join(hf_home, "blue-onnx-v2")
        renikud_path = os.path.join(hf_home, "renikud.onnx")
        voice = os.environ.get("TTS_VOICE", "female1")

        logger.info({"event": "model_loading", "model": _MODEL_REPO, "mode": "tts"})

        snapshot_download(repo_id=_MODEL_REPO, local_dir=model_dir)

        if not os.path.exists(renikud_path):
            hf_hub_download(
                repo_id=_RENIKUD_REPO,
                filename="model.onnx",
                local_dir=hf_home,
            )

        style_path = os.path.join(model_dir, "voices", f"{voice}.json")
        if not os.path.exists(style_path):
            style_path = os.path.join(model_dir, f"{voice}.json")

        self._model = BlueTTS(
            onnx_dir=model_dir,
            style_json=style_path,
            renikud_path=renikud_path,
        )
        logger.info({"event": "model_ready", "model": _MODEL_REPO})

    def handle(self, job_input: dict) -> dict:
        text = job_input.get("text")
        if not text:
            return {"error": "text is required"}

        if len(text) > _MAX_TEXT_CHARS:
            return {"error": f"text exceeds {_MAX_TEXT_CHARS} character limit"}

        samples, sr = self._model.synthesize(text, lang="he", total_step=self._steps, speed=self._speed)

        buffer = io.BytesIO()
        sf.write(buffer, samples, sr, format="WAV")
        buffer.seek(0)

        return {"audio": base64.b64encode(buffer.read()).decode("utf-8")}
