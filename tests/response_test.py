import base64
import json
from pathlib import Path

RESPONSES_DIR = Path(__file__).parent / "responses"
RESPONSES_DIR.mkdir(exist_ok=True)

response_path = RESPONSES_DIR / "r1.json"
response = json.loads(response_path.read_text(encoding="utf-8"))

out_path = RESPONSES_DIR / "output.wav"
out_path.write_bytes(base64.b64decode(response["output"]["audio"]))

print(f"Saved {out_path}")
