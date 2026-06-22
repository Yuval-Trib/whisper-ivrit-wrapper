"""
TTS warmth/space enhancement. Builds on output_step2_pyworld+pedalboard.wav.
Change MODE to switch presets, then listen to output.

Run:
    python tests/enhance_test2.py

Output: tests/responses/output_step3_<MODE>.wav
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

INPUT = Path(__file__).parent / "responses" / "output_step2_pyworld+pedalboard.wav"

# Switch between presets to compare:
# "chorus+reverb", "chorus", "reverb", "full"
MODE = "chorus"


def apply_chorus_reverb(audio: np.ndarray, sr: int) -> np.ndarray:
    from pedalboard import Chorus, Pedalboard, Reverb
    board = Pedalboard([
        Chorus(rate_hz=1.0, depth=0.4, centre_delay_ms=7.0, mix=0.4),
        Reverb(room_size=0.25, damping=0.5, wet_level=0.2, dry_level=0.8),
    ])
    return board(audio.astype(np.float32), sample_rate=sr)


def apply_chorus(audio: np.ndarray, sr: int) -> np.ndarray:
    from pedalboard import Chorus, Pedalboard
    board = Pedalboard([
        Chorus(rate_hz=1.0, depth=0.4, centre_delay_ms=7.0, mix=0.4),
    ])
    return board(audio.astype(np.float32), sample_rate=sr)


def apply_reverb(audio: np.ndarray, sr: int) -> np.ndarray:
    from pedalboard import Pedalboard, Reverb
    board = Pedalboard([
        Reverb(room_size=0.25, damping=0.5, wet_level=0.2, dry_level=0.8),
    ])
    return board(audio.astype(np.float32), sample_rate=sr)


_PRESETS = {
    "chorus": apply_chorus,
    "reverb": apply_reverb,
    "chorus+reverb": apply_chorus_reverb,
}


def main() -> None:
    if MODE not in _PRESETS:
        print(f"Unknown MODE {MODE!r}. Options: {list(_PRESETS)}", file=sys.stderr)
        sys.exit(1)

    if not INPUT.exists():
        print(f"Input not found: {INPUT}", file=sys.stderr)
        sys.exit(1)

    audio, sr = sf.read(str(INPUT), dtype="float32")
    print(f"Loaded {INPUT.name}: {len(audio) / sr:.2f}s at {sr} Hz")

    print(f"Applying: {MODE}")
    result = _PRESETS[MODE](audio, sr)

    out = INPUT.parent / f"output_step3_{MODE}.wav"
    sf.write(str(out), result, sr)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
