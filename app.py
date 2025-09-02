import os
import glob
import numpy as np
import librosa
import pyloudnorm as pyln
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response

app = FastAPI(title="DJ Track Analyzer", version="0.5")

EXAMPLES_DIR = "examples"

# ====== Beats/targets ======
GREEN_TARGET_BEATS = 192                 # perfect beats-to-break target
INTRO_TARGET_BEATS = 128                 # 32 bars = preferred intro
def _beats_to_break_status(n, target=GREEN_TARGET_BEATS):
    if n is None:
        return "unknown"
    if n == target:
        return "green"
    if n < target:
        return "red"
    return "orange"
# ===========================

# Silence icon 404s (optional)
@app.get("/apple-touch-icon.png")
def apple_icon():
    return Response(status_code=204)

@app.get("/apple-touch-icon-precomposed.png")
def apple_icon_pre():
    return Response(status_code=204)

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# ===================== Utilities =====================

def load_audio_mono(path, target_sr=22050):
    """Downsampled mono load for faster analysis."""
    y, sr = librosa.load(path, sr=target_sr, mono=True, res_type="kaiser_fast")
    return y, sr

def robust_tempo(y: np.ndarray, sr: int):
    """
    DJ-focused tempo detection with robust half/double correction.
    Returns (bpm, confidence 0..1).
    """
    # Wider range so hard techno (e.g., 156 BPM) is valid
    min_bpm, max_bpm = 120.0, 180.0

    def fit_range(bpm: float, lo: float, hi: float) -> float:
        if bpm <= 0:
            return 0.0
        # Move by factors of 2 until inside [lo, hi]
        while not (lo <= bpm <= hi):
            bpm = bpm * 2.0 if bpm < lo else bpm / 2.0
        return bpm

    duration = len(y) / sr
    seg_len = int(min(60.0, max(20.0, duration / 4.0)) * sr)
    positions = [0, max(0, len(y)//2 - seg_len//2), max(0, len(y) - seg_len)]

    cands_all = []
    for pos in positions:
        seg = y[pos:pos+seg_len]
        if len(seg) < 5 * sr:
            continue
        onset_env = librosa.onset.onset_strength(y=seg, sr=sr)
        try:
            from librosa.feature.rhythm import tempo as rhythm_tempo
            cands = rhythm_tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        except Exception:
            cands = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        if isinstance(cands, np.ndarray):
            cands = cands.tolist()
        cands_all.extend([float(x) for x in cands[:5]])

    if not cands_all:
        return 0.0, 0.0

    # Generate half/double variants and fold into range
    corrected = []
    for t in cands_all:
        for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
            bpm = fit_range(t * mult, min_bpm, max_bpm)
            if min_bpm <= bpm <= max_bpm:
                corrected.append(bpm)

    if not corrected:
        t0 = fit_range(float(cands_all[0]), min_bpm, max_bpm)
        corrected = [t0]

    bpm = float(np.median(corrected))
    bpm = round(bpm * 2.0) / 2.0   # round to 0.5 BPM for stability
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

# --- Beat grid refinement: snap tempo + phase to onsets (first 120s) ---
def refine_tempo_and_phase(y, sr, rough_bpm):
    hop = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if oenv is None or len(oenv) < 16:
        return float(rough_bpm), 0.0
    times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=hop)

    # score only the first 120s to avoid outro bias
    max_t = 120.0
    limit = min(len(times)-1, int(np.searchsorted(times, max_t)))
    if limit < 16:
        return float(rough_bpm), 0.0

    # Wider window around rough BPM
    bpm_lo = max(110.0, float(rough_bpm) - 12.0)
    bpm_hi = min(190.0, float(rough_bpm) + 12.0)
    bpm_cands = np.linspace(bpm_lo, bpm_hi, 31)

    best_bpm, best_phase, best_score = float(rough_bpm), 0.0, -1.0
    for bpm in bpm_cands:
        spb = 60.0 / bpm
        # 32 phase tests across one beat
        for k in range(32):
            t0 = (k / 32.0) * spb
            grid = t0 + np.arange(0.0, times[limit], spb)
            if len(grid) == 0:
                continue
            idx = np.clip(np.searchsorted(times[:limit], grid), 0, limit-1)
            score = float(np.mean(oenv[idx]))
            if score > best_score:
                best_bpm, best_phase, best_score = float(bpm), float(t0), score
    return best_bpm, best_phase

# ===================== Beat-aware intro/outro detection =====================

def detect_intro_outro(y, sr, bpm, y_harm=None, y_perc=None):
    """
    Beat/bar-aware intro/outro detection using HPSS.
    Accepts optional precomputed HPSS (y_harm, y_perc) to avoid recomputation.
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
    beat_times = np.append(beat_times, duration)  # safe end boundary

    # HPSS (use precomputed if provided)
    if y_harm is None or y_perc is None:
        y_harm, y_perc = librosa.effects.hpss(y)

    def beat_rms(sig, t0, t1):
        s0 = int(max(0, np.floor(t0 * sr)))
        s1 = int(min(len(sig), np.floor(t1 * sr)))
        if s1 <= s0:
            return 0.0
        seg = sig[s0:s1]
        return float(np.sqrt(np.mean(seg * seg) + 1e-12))

    # Per-beat harmonic/percussive RMS
    harm_rms = []
    perc_rms = []
    for i in range(len(beat_times) - 1):
        t0, t1 = beat_times[i], beat_times[i + 1]
        harm_rms.append(beat_rms(y_harm, t0, t1))
        perc_rms.append(beat_rms(y_perc, t0, t1))
    harm_rms = np.asarray(harm_rms)
    perc_rms = np.asarray(perc_rms)

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
    r_p70 = float(np.percentile(ratio_bars, 70))
    r_p30 = float(np.percentile(ratio_bars, 30))

    thr_intro_high = max(start_base + 0.15, r_p70 * 0.85, 0.45)
    thr_outro_low  = min(end_base + 0.10, r_p30 * 1.05, 0.50)
    sustain_bars = 2  # require 2 bars of evidence

    # Intro end
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

    # Outro start
    outro_start_bar = nb - 1
    run = 0
    search_start = max(0, nb - 64)
    for b in range(nb - 1, search_start - 1, -1):
        if ratio_bars[b] <= thr_outro_low:
            run += 1
            if run >= sustain_bars:
                outro_start_bar = min(nb - 1, b)
                break
        else:
            run = 0

    def bar_to_time(bar_idx):
        beat_idx = min(bar_idx * 4, len(beat_times) - 1)
        return float(beat_times[beat_idx])

    intro_end_sec = bar_to_time(intro_end_bar)
    outro_start_sec = bar_to_time(outro_start_bar)

    # clamps
    intro_end_sec = max(0.0, min(intro_end_sec, duration))
    outro_start_sec = max(0.0, min(outro_start_sec, duration))
    return intro_end_sec, outro_start_sec

def detect_intro_outro_rms_fallback(y, sr):
    # very basic fallback if beat tracking fails
    duration = len(y) / sr
    return min(32.0, duration * 0.25), max(0.0, duration - 32.0)

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
    """Scan examples/, analyze, and derive target intro/outro in bars (median)."""
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

    intro_bars = [r["intro_bars"] for r in results if r["intro_bars"] and r["intro_bars"] >= 4.0]
    outro_bars = [r["outro_bars"] for r in results if r["outro_bars"] and r["outro_bars"] >= 4.0]

    tgt_intro = float(np.median(intro_bars)) if intro_bars else 16.0
    tgt_outro = float(np.median(outro_bars)) if outro_bars else tgt_intro

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

    # Tempo (rough) → refine BPM and phase for a stable beat grid
    tempo_bpm_raw, tempo_conf = robust_tempo(y, sr)
    tempo_bpm, beat_phase_sec = refine_tempo_and_phase(y, sr, tempo_bpm_raw)

    # Precompute HPSS once (big speed win)
    y_h, y_p = librosa.effects.hpss(y)

    meter = pyln.Meter(sr)
    loudness = float(meter.integrated_loudness(y))
    peak_dbfs = float(20 * np.log10(max(1e-12, np.max(np.abs(y)))))

    intro_end_sec, outro_start_sec = detect_intro_outro(y, sr, tempo_bpm, y_harm=y_h, y_perc=y_p)
    intro_bars = seconds_to_bars(intro_end_sec, tempo_bpm)
    outro_bars = seconds_to_bars(max(0.0, duration - outro_start_sec), tempo_bpm)

    ex = get_examples_summary()
    tgt_intro = INTRO_TARGET_BEATS / 4.0   # 32 bars target
    tgt_outro = ex["target_outro_bars"]

    def verdict(measured, target):
        if (measured is None) or (target is None) or (measured <= 0):
            return "unknown"
        diff = abs(measured - target)
        if diff <= 4.0:
            return "good"
        elif diff >= 8.0:
            return "bad"
        else:
            return "warn"

    intro_verdict = verdict(intro_bars, tgt_intro)
    outro_verdict = verdict(outro_bars, tgt_outro)

    # ====== First-break detection (grid-anchored, ≥15s, drum pullback, earliest phrase) ======
    beats_before_break = None
    break_time_sec = None
    try:
        if tempo_bpm > 0:
            # Use refined grid (BPM + phase) so counting is exact
            spb = 60.0 / float(tempo_bpm)

            def seg_rms_db(sig, t0, t1):
                s0 = max(0, int(t0 * sr))
                s1 = min(len(sig), int(t1 * sr))
                if s1 <= s0:
                    return -120.0
                seg = sig[s0:s1].astype(np.float32)
                rms = np.sqrt(float(np.mean(seg * seg)) + 1e-12)
                return 20.0 * np.log10(rms + 1e-12)

            # Tunables
            PHRASE_BEATS   = 32
            PREV_BEATS     = 16        # baseline window before candidate
            MIN_BREAK_SEC  = 15.0      # break must sustain ≥15s
            DROP_DB        = 5.0       # overall energy drop (was 5.5)
            PERC_DROP_DB   = 4.0       # percussive drop (was 4.5)
            TIME_MIN_SEC   = 20.0      # allow earlier breaks
            TIME_MAX_SEC   = min(180.0, duration * 0.7)  # dynamic upper bound

            # Build candidate phrase starts on the refined grid, earliest first
            candidates = []
            b = PHRASE_BEATS * 2  # start at 64 beats
            while True:
                t = beat_phase_sec + b * spb
                if t > duration:
                    break
                if TIME_MIN_SEC <= t <= TIME_MAX_SEC:
                    candidates.append((b, t))
                elif t > TIME_MAX_SEC:
                    break
                b += PHRASE_BEATS  # 32-beat phrases: 64, 96, 128, 160, 192, ...

            # Evaluate candidates; pick the FIRST that passes
            for beats, t_start in candidates:
                prev_t0 = max(0.0, t_start - PREV_BEATS * spb)
                prev_t1 = t_start
                next_t0 = t_start
                next_t1 = min(duration, t_start + MIN_BREAK_SEC)

                prev_tot_db = seg_rms_db(y,   prev_t0, prev_t1)
                next_tot_db = seg_rms_db(y,   next_t0, next_t1)
                prev_per_db = seg_rms_db(y_p, prev_t0, prev_t1)
                next_per_db = seg_rms_db(y_p, next_t0, next_t1)

                if (prev_tot_db - next_tot_db) >= DROP_DB and (prev_per_db - next_per_db) >= PERC_DROP_DB:
                    break_time_sec = float(t_start)
                    beats_before_break = int(round((break_time_sec - beat_phase_sec) / spb))
                    break
    except Exception:
        pass

    beats_to_break_status = _beats_to_break_status(beats_before_break, GREEN_TARGET_BEATS)

    # --- Override intro length to match the green selection: start → (break - 64 beats)
    # If we detected the first break, define intro as (beats_before_break - 64) beats.
    if beats_before_break is not None and tempo_bpm > 0:
        WHITE_BEATS = 64
        intro_bars = round(max(0, beats_before_break - WHITE_BEATS) / 4.0, 1)
        # Recompute verdict for the new intro length
        intro_verdict = verdict(intro_bars, tgt_intro)
    # ==========================================================================================

    notes = []
    if duration < 150:
        notes.append("⚠️ Track is short for club use (<4:00).")
    if not (120 <= tempo_bpm <= 180):
        notes.append("⚠️ Tempo outside common range for techno/hard techno (120–180 BPM).")
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

        # Section times for overlays (from HPSS detector)
        "intro_end_sec": round(float(intro_end_sec), 2),
        "outro_start_sec": round(float(outro_start_sec), 2),

        # Lengths (intro possibly overridden above)
        "intro_bars": round(float(intro_bars), 1) if tempo_bpm > 0 else None,
        "intro_beats": int(round(float(intro_bars * 4))) if tempo_bpm > 0 else None,
        "intro_length": (f"{round(float(intro_bars),1)} bars ({int(round(float(intro_bars*4)))} beats)"
                         if tempo_bpm > 0 else None),

        "outro_bars": round(float(outro_bars), 1) if tempo_bpm > 0 else None,
        "outro_beats": int(round(float(outro_bars * 4))) if tempo_bpm > 0 else None,
        "outro_length": (f"{round(float(outro_bars),1)} bars ({int(round(float(outro_bars*4)))} beats)"
                         if tempo_bpm > 0 else None),

        # Examples model / targets
        "examples_count": ex["count"],
        "target_intro_bars": round(tgt_intro, 1),
        "target_outro_bars": ex["target_outro_bars"],

        # Verdicts
        "intro_verdict": intro_verdict,
        "outro_verdict": outro_verdict,

        # Beats-to-break
        "beats_before_break": beats_before_break,  # int or None
        "break_time_sec": round(break_time_sec, 3) if break_time_sec is not None else None,
        "target_beats_to_break": GREEN_TARGET_BEATS,
        "beats_to_break_status": beats_to_break_status,

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