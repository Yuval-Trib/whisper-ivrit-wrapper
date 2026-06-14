"""Extract and save TTS audio from docker output.

Usage:
    docker run ... 2>&1 | Out-File -Encoding utf8 tests\raw_output.txt
    python tests\decode_audio.py tests\raw_output.txt
"""
import ast
import base64
import subprocess
import sys
from pathlib import Path

src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
text = src.read_text(encoding="utf-8") if src else sys.stdin.read()

for line in text.splitlines():
    if "Job result:" in line:
        dict_str = line.split("Job result: ", 1)[1]
        result = ast.literal_eval(dict_str)
        audio_b64 = result.get("output", {}).get("audio")
        if not audio_b64:
            print("No audio in output", file=sys.stderr)
            sys.exit(1)
        out_path = Path("tests/output.wav")
        out_path.write_bytes(base64.b64decode(audio_b64))
        print(f"Saved {out_path}")
        subprocess.Popen(["explorer", str(out_path)])
        sys.exit(0)

print("No 'Job result:' line found in output", file=sys.stderr)
sys.exit(1)
