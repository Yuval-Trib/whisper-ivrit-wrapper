"""Run TTS and write output.wav to /output (mount tests/ there).

Usage:
    docker run --rm --gpus all -e DEVICE=cuda -e HF_HOME=/tmp/hf \
        -v "%CD%\tests:/output" whisper-ivrit-wrapper \
        python3.12 /output/tts_to_file.py
"""
import base64
import sys
sys.path.insert(0, "/app")

from app.tts import TTS

handler = TTS()
handler.load_model()

result = handler.handle({"text": "שלום, איך אתה מרגיש היום? אני כאן כדי לעזור לך."})
if "error" in result:
    print(f"Error: {result['error']}", file=sys.stderr)
    sys.exit(1)

with open("/output/output.wav", "wb") as f:
    f.write(base64.b64decode(result["audio"]))
print("Saved /output/output.wav")
