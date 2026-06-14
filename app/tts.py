import base64
import io
import logging
import os

import numpy as np
import onnxruntime as ort
import soundfile as sf
from blue_onnx import (
    TextProcessor,
    TextToSpeech,
    UnicodeProcessor,
    load_cfgs,
    load_onnx_all,
    load_voice_style,
)
from huggingface_hub import hf_hub_download, snapshot_download

from app.base import BaseHandler

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 5000
_MODEL_REPO = "notmax123/blue-onnx-v2"
_DEFAULT_SAMPLE_RATE = 44100


def _load_tts_gpu(onnx_dir: str, renikud_path: str, config_path: str = "tts.json") -> TextToSpeech:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    cfgs = load_cfgs(onnx_dir, config_path)
    dp_ort, text_enc_ort, vector_est_ort, vocoder_ort = load_onnx_all(onnx_dir, opts, providers)
    text_processor = UnicodeProcessor(os.path.join(onnx_dir, "vocab.json"))
    return TextToSpeech(cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort, g2p=TextProcessor(renikud_path=renikud_path))


class TTS(BaseHandler):

    def __init__(self) -> None:
        self._model: TextToSpeech | None = None
        self._style = None
        self._sample_rate = _DEFAULT_SAMPLE_RATE
        self._steps = int(os.environ.get("TTS_STEPS", 8))
        self._speed = float(os.environ.get("TTS_SPEED", 0.95))
        self._cfg_scale = float(os.environ.get("TTS_CFG_SCALE", 4.0))

    def load_model(self) -> None:
        hf_home = os.environ.get("HF_HOME", "/runpod-volume/huggingface")
        model_dir = os.path.join(hf_home, "blue-onnx-v2")
        voice = os.environ.get("TTS_VOICE", "female1")

        logger.info({"event": "model_loading", "model": _MODEL_REPO, "mode": "tts"})
        snapshot_download(repo_id=_MODEL_REPO, local_dir=model_dir)

        renikud_path = hf_hub_download(
            repo_id="thewh1teagle/renikud",
            filename="model.onnx",
            local_dir=os.path.join(hf_home, "renikud"),
        )
        self._model = _load_tts_gpu(model_dir, renikud_path=renikud_path)

        style_path = os.path.join(model_dir, "voices", f"{voice}.json")
        if not os.path.exists(style_path):
            style_path = os.path.join(model_dir, f"{voice}.json")
        self._style = load_voice_style([style_path])

        logger.info({"event": "model_ready", "model": _MODEL_REPO})

    def handle(self, job_input: dict) -> dict:
        text = job_input.get("text")
        if not text:
            return {"error": "text is required"}

        if len(text) > _MAX_TEXT_CHARS:
            return {"error": f"text exceeds {_MAX_TEXT_CHARS} character limit"}

        audio_batch, _ = self._model.batch(
            [text], ["he"], self._style,
            total_step=self._steps,
            speed=self._speed,
            cfg_scale=self._cfg_scale,
        )

        audio = audio_batch[0]
        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = np.squeeze(audio)

        buffer = io.BytesIO()
        sf.write(buffer, audio, self._sample_rate, format="WAV")
        buffer.seek(0)

        return {"audio": base64.b64encode(buffer.read()).decode("utf-8")}
