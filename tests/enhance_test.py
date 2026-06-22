"""
TTS audio enhancement test. Change MODE to switch presets, then listen to output.

Install deps:
    pip install pedalboard pyworld librosa

Run:
    python tests/enhance_test.py

Output: tests/responses/output_step2_<MODE>.wav
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

INPUT = Path(__file__).parent / "responses" / "output_step1.wav"

# Switch between presets to compare:
# "original", "pedalboard", "pyworld", "librosa", "pyworld+pedalboard"
MODE = "pyworld+pedalboard"


def apply_pedalboard(audio: np.ndarray, sr: int) -> np.ndarray:
    from pedalboard import Compressor, Gain, HighShelfFilter, LowShelfFilter, Pedalboard
    board = Pedalboard([
        Compressor(threshold_db=-20, ratio=3.0, attack_ms=5, release_ms=100),
        LowShelfFilter(cutoff_frequency_hz=250, gain_db=-3),
        HighShelfFilter(cutoff_frequency_hz=4000, gain_db=2),
        Gain(gain_db=1),
    ])
    return board(audio.astype(np.float32), sample_rate=sr)


def apply_pyworld(audio: np.ndarray, sr: int) -> np.ndarray:
    import pyworld
    from scipy.ndimage import gaussian_filter1d

    y = audio.astype(np.float64)
    f0, t = pyworld.harvest(y, sr)
    sp = pyworld.cheaptrick(y, f0, t, sr)
    ap = pyworld.d4c(y, f0, t, sr)

    voiced = f0 > 0
    if voiced.any():
        mean_f0 = f0[voiced].mean()
        rng = np.random.default_rng(42)
        noise = rng.normal(0, mean_f0 * 0.03, f0.shape)
        smooth_variation = gaussian_filter1d(noise, sigma=50)
        f0_mod = np.where(voiced, f0 + smooth_variation, 0.0)

        end_start = int(len(f0) * 0.8)
        slope = np.linspace(0, -mean_f0 * 0.1, len(f0) - end_start)
        f0_mod[end_start:] = np.where(voiced[end_start:], f0_mod[end_start:] + slope, 0.0)
        f0_mod = np.maximum(f0_mod, 0.0)
    else:
        f0_mod = f0

    return pyworld.synthesize(f0_mod, sp, ap, sr).astype(np.float32)


def apply_librosa(audio: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    rng = np.random.default_rng(42)
    chunks = np.array_split(audio, 8)
    return np.concatenate([
        librosa.effects.pitch_shift(c.astype(np.float32), sr=sr, n_steps=float(rng.uniform(-1.0, 1.0)))
        for c in chunks
    ])


_PRESETS = {
    "original": lambda a, sr: a,
    "pedalboard": apply_pedalboard,
    "pyworld": apply_pyworld,
    "librosa": apply_librosa,
    "pyworld+pedalboard": lambda a, sr: apply_pedalboard(apply_pyworld(a, sr), sr),
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

    out = INPUT.parent / f"output_step2_{MODE}.wav"
    sf.write(str(out), result, sr)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
