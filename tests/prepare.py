"""Generate test_<mode>.json files for all handler modes.

Convention: one entry per mode in INPUTS below.
The runner discovers these files automatically by name.
Adding a new endpoint: add its mode and input payload here.
"""
import base64
import json
import sys
from pathlib import Path

here = Path(__file__).parent

audio_path = here / "transcript.mp3"
if not audio_path.exists():
    print(f"ERROR: {audio_path} not found", file=sys.stderr)
    sys.exit(1)

audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()

INPUTS: dict[str, dict] = {
    "stt_short": {"audio": audio_b64, "language": "he"},
    "stt_long":  {"audio": audio_b64, "language": "he"},
    "tts":       {"text": "שלום, איך אתה מרגיש היום? אני כאן כדי לעזור לך."},
}

for mode, payload in INPUTS.items():
    path = here / f"test_{mode}.json"
    path.write_text(json.dumps({"input": payload}), encoding="utf-8")
    print(f"wrote {path.name}")
