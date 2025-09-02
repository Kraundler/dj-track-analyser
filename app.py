import os
import glob
import numpy as np
import librosa
import pyloudnorm as pyln
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="DJ Track Analyzer", version="0.5")

EXAMPLES_DIR = "examples"

# ===================== Utilities =====================

def load_audio_mono(path, target_sr=22050):
    """Downsampled mono load for faster analysis."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr

def robust_tempo(y: np.ndarray, sr: int):
    """
    DJ-focused tempo detection with half/double correction.
    Returns (bpm, confidence 0..1).
    """
    min_bpm, max_bpm = 124.0, 150.0
    duration = len(y) / sr
    seg_len = int(min(60.0, max(20.0, duration / 4.0)) * sr)
    positions = [0, max(0, len(y)//2 - seg_len//2), max(0, len(y) - seg_len)]

    cands_all = []
    for pos in positions:
        seg = y[pos:pos+seg_len]
        if len(seg) < 5 * sr:
            continue
        onset_env = librosa.onset.onset_strength(y=seg, sr=sr)
        # librosa >=0.10 moved tempo; keep backwards compat
        try:
            from librosa.feature.rhythm import tempo as rhythm_tempo
            cands = rhythm_tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        except Exception:
            cands = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        if isinstance(cands, np.ndarray):
            cands = cands.tolist()
        cands_all.extend(cands[:5])

    if not cands_all:
        return 0.0, 0.0

    corrected = []
    for t in cands_all:
        for mult in (0.5, 1.0, 2.0):
            bpm = t * mult
            if min_bpm <= bpm <= max_bpm:
                corrected.append(bpm)

    if not corrected:
        t = float(cands_all[0])
        while t < min_bpm:
            t *= 2.0
        while t > max_bpm:
            t /= 2.0
        corrected = [t]

    bpm = float(np.median(corrected))
    bpm = round(bpm * 2) / 2.0
    spread = float(np.std(corrected)) if len(corrected) > 1 else 0.0
    conf = max(0.0, min(1.0, 1.0 - spread / 8.0))
    return bpm, conf

def seconds_to_bars(seconds, bpm):
    if bpm <= 0:
        return 0.0
    beats = seconds * (bpm / 60.0)
    return beats / 4.0

def bars_to_seconds(bars, bpm):
    if bpm <= 0:
        return 0.0
    return (bars * 4.0) * 60.0 / bpm

def moving_avg(x, k):
    if k <= 1 or len(x) == 0:
        return x
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    y = np.convolve(xp, ker, mode="same")[pad:-pad]
    return y

# ===================== Beat-aware intro/outro detection =====================

def detect_intro_outro(y, sr, bpm):
    """
    Beat/bar-aware intro/outro detection using HPSS:
    - Compute harmonic/percussive RMS per beat
    - Aggregate per bar (4 beats)
    - Intro ends at first sustained rise of harmonic ratio (>= threshold)
    - Outro starts at last sustained drop to drums-only (<= threshold)
    - Snap boundaries to bar edges
    Returns (intro_end_sec, outro_start_sec)
    """
    duration = len(y) / sr
    if bpm <= 0 or duration < 30:
        return 0.0, max(0.0, duration - 30.0)

    # Beat track
    hop = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=oenv, hop_length=hop)
    if beat_frames is None or len(beat_frames) < 16:
        return detect_intro_outro_rms_fallback(y, sr)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    # add track end as last boundary for safe indexing
    beat_times = np.append(beat_times, duration)

    # HPSS and per-beat RMS
    y_harm, y_perc = librosa.effects.hpss(y)

    def beat_rms(sig, t0, t1):
        s0 = int(max(0, np.floor(t0 * sr)))
        s1 = int(min(len(sig), np.floor(t1 * sr)))
        if s1 <= s0:
            return 0.0
        seg = sig[s0:s1]
        return float(np.sqrt(np.mean(seg * seg) + 1e-12))

    harm_rms = []
    perc_rms = []
    for i in range(len(beat_times) - 1):
        t0, t1 = beat_times[i], beat_times[i + 1]
        harm_rms.append(beat_rms(y_harm, t0, t1))
        perc_rms.append(beat_rms(y_perc, t0, t1))
    harm_rms = np.asarray(harm_rms)
    perc_rms = np.asarray(perc_rms)

    # Ratio (harmonic over percussive). Intros/outros are drum-heavy => low ratio.
    ratio_beats = harm_rms / (perc_rms + 1e-8)

    # Aggregate by bars (4 beats)
    nb = len(ratio_beats) // 4
    if nb < 8:
        return detect_intro_outro_rms_fallback(y, sr)
    ratio_bars = ratio_beats[:nb * 4].reshape(nb, 4).mean(axis=1)

    # Baselines / thresholds
    start_win = min(8, nb // 4) or 4
    end_win = min(8, nb // 4) or 4
    start_base = float(np.median(ratio_bars[:start_win]))
    end_base = float(np.median(ratio_bars[-end_win:]))
    r_med = float(np.median(ratio_bars))
    r_p70 = float(np.percentile(ratio_bars, 70))
    r_p30 = float(np.percentile(ratio_bars, 30))

    # Thresholds tuned for techno/house intros/outros
    thr_intro_high = max(start_base + 0.15, r_p70 * 0.85, 0.45)
    thr_outro_low  = min(end_base + 0.10, r_p30 * 1.05, 0.50)

    sustain_bars = 2  # require 2 bars of evidence

    # Intro end: first sustained >= thr_intro_high within first 64 bars
    intro_end_bar = 0
    run = 0
    search_end = min(nb, 64)
    for b in range(search_end):
        if ratio_bars[b] >= thr_intro_high:
            run += 1
            if run >= sustain_bars:
                intro_end_bar = max(0, b - sustain_bars + 1)
                break
        else:
            run = 0

    # Outro start: last sustained <= thr_outro_low within last 64 bars
    outro_start_bar = nb - 1
    run = 0
    search_start = max(0, nb - 64)
    for b in range(nb - 1, search_start - 1, -1):
        if ratio_bars[b] <= thr_outro_low:
            run += 1
            if run >= sustain_bars:
                outro_start_bar = min(nb - 1, b)  # start of drums-only region
                break
        else:
            run = 0

    # Snap to bar boundaries and convert to seconds (bar -> beat index -> time)
    intro_end_bar = max(0, int(round(intro_end_bar / 1.0)))  # already bar-accurate
    outro_start_bar = max(0, int(round(outro_start_bar / 1.0)))

    def bar_to_time(bar_idx):
        beat_idx = min(bar_idx * 4, len(beat_times) - 1)
        return float(beat_times[beat_idx])

    intro_end_sec = bar_to_time(intro_end_bar)
    outro_start_sec = bar_to_time(outro_start_bar)

    # clamps
    intro_end_sec = max(0.0, min(intro_end_sec, duration))
    outro_start_sec = max(0.0, min(outro_start_sec, duration))
    return intro_end_sec, outro_start_sec

# ===================== Examples model =====================

_examples_cache = {"mtimes": {}, "summary": None}

def analyze_example_file(path):
    y, sr = load_audio_mono(path, target_sr=22050)
    bpm, _ = robust_tempo(y, sr)
    intro_end_sec, outro_start_sec = detect_intro_outro(y, sr, bpm)
    duration = len(y) / sr
    intro_bars = seconds_to_bars(intro_end_sec, bpm)
    outro_bars = seconds_to_bars(duration - outro_start_sec, bpm)
    return dict(
        file=os.path.basename(path),
        bpm=bpm,
        duration_sec=round(duration, 2),
        intro_end_sec=round(intro_end_sec, 2),
        outro_start_sec=round(outro_start_sec, 2),
        intro_bars=round(intro_bars, 1),
        outro_bars=round(outro_bars, 1),
    )

def build_examples_summary():
    """Scan examples/, analyze, and derive target intro/outro in bars (median).
       Ignores broken/very short values; falls back to 16 bars if nothing valid."""
    files = []
    if os.path.isdir(EXAMPLES_DIR):
        for ext in ("*.wav", "*.mp3", "*.aiff", "*.flac"):
            files.extend(glob.glob(os.path.join(EXAMPLES_DIR, ext)))

    results = []
    for f in sorted(files):
        try:
            results.append(analyze_example_file(f))
        except Exception:
            pass

    if not results:
        return {
            "count": 0,
            "target_intro_bars": 16.0,
            "target_outro_bars": 16.0,
            "examples": [],
        }

    # Keep only reasonable values (>= 4 bars)
    intro_bars = [r["intro_bars"] for r in results if r["intro_bars"] and r["intro_bars"] >= 4.0]
    outro_bars = [r["outro_bars"] for r in results if r["outro_bars"] and r["outro_bars"] >= 4.0]

    tgt_intro = float(np.median(intro_bars)) if intro_bars else 16.0
    tgt_outro = float(np.median(outro_bars)) if outro_bars else tgt_intro  # default to same as intro

    # Round to nearest 4 bars
    def round_to_4(b):
        return float(int(round(b / 4.0)) * 4)

    return {
        "count": len(results),
        "target_intro_bars": round_to_4(tgt_intro),
        "target_outro_bars": round_to_4(tgt_outro),
        "examples": results,
    }


def get_examples_summary():
    """Simple cache: rebuild if folder mtimes changed."""
    global _examples_cache
    mtimes = {}
    if os.path.isdir(EXAMPLES_DIR):
        for p in glob.glob(os.path.join(EXAMPLES_DIR, "*")):
            try:
                mtimes[p] = os.path.getmtime(p)
            except Exception:
                pass
    if mtimes != _examples_cache["mtimes"]:
        _examples_cache["mtimes"] = mtimes
        _examples_cache["summary"] = build_examples_summary()
    return _examples_cache["summary"] or {"count": 0, "target_intro_bars": None, "target_outro_bars": None, "examples": []}

# ===================== Core analysis (DJ only) =====================

def analyze_track(path: str):
    y, sr = load_audio_mono(path, target_sr=22050)
    duration = len(y) / sr

    tempo_bpm, tempo_conf = robust_tempo(y, sr)

    meter = pyln.Meter(sr)
    loudness = float(meter.integrated_loudness(y))
    peak_dbfs = float(20 * np.log10(max(1e-12, np.max(np.abs(y)))))

    intro_end_sec, outro_start_sec = detect_intro_outro(y, sr, tempo_bpm)
    intro_bars = seconds_to_bars(intro_end_sec, tempo_bpm)
    outro_bars = seconds_to_bars(max(0.0, duration - outro_start_sec), tempo_bpm)

    ex = get_examples_summary()
    tgt_intro = ex["target_intro_bars"]
    tgt_outro = ex["target_outro_bars"]

    def verdict(measured, target):
        if (measured is None) or (target is None) or (measured <= 0):
            return "unknown"
        diff = abs(measured - target)
        if diff <= 4.0:
            return "good"   # green
        elif diff >= 8.0:
            return "bad"    # red
        else:
            return "warn"   # amber

    intro_verdict = verdict(intro_bars, tgt_intro)
    outro_verdict = verdict(outro_bars, tgt_outro)

    notes = []
    if duration < 150:
        notes.append("⚠️ Track is short for club use (<4:00).")
    if not (120 <= tempo_bpm <= 160):
        notes.append("⚠️ Tempo outside common DJ range (120–160 BPM).")
    if loudness > -6.5:
        notes.append("⚠️ Very loud; risk of clipping on club systems.")
    if loudness < -9.5:
        notes.append("⚠️ Quiet; consider stronger limiting/level.")

    return {
        "file": os.path.basename(path),
        "duration_sec": round(duration, 2),

        # Tempo / levels
        "tempo_bpm": round(tempo_bpm, 1),
        "tempo_confidence": round(tempo_conf * 100, 1),
        "loudness_lufs": round(loudness, 2),
        "peak_dbfs": round(peak_dbfs, 2),

        # Section times for overlays
        "intro_end_sec": round(float(intro_end_sec), 2),
        "outro_start_sec": round(float(outro_start_sec), 2),

        # Lengths: numeric + beats + formatted string (bars (beats))
        "intro_bars": round(float(intro_bars), 1) if tempo_bpm > 0 else None,
        "intro_beats": int(round(float(intro_bars * 4))) if tempo_bpm > 0 else None,
        "intro_length": (
        f"{round(float(intro_bars),1)} bars ({int(round(float(intro_bars*4)))} beats)"
        if tempo_bpm > 0 else None
        ),

        "outro_bars": round(float(outro_bars), 1) if tempo_bpm > 0 else None,
        "outro_beats": int(round(float(outro_bars * 4))) if tempo_bpm > 0 else None,
        "outro_length": (
        f"{round(float(outro_bars),1)} bars ({int(round(float(outro_bars*4)))} beats)"
        if tempo_bpm > 0 else None
        ),


        # Examples model (targets from your reference tracks)
        "examples_count": ex["count"],
        "target_intro_bars": ex["target_intro_bars"],
        "target_outro_bars": ex["target_outro_bars"],

        # Verdicts for coloring
        "intro_verdict": intro_verdict,
        "outro_verdict": outro_verdict,

        "notes": notes,
    }

# ===================== Routes =====================

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    tmp_path = f"temp_upload{suffix}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        result = analyze_track(tmp_path)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
