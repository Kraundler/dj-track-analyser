import sys, numpy as np
import librosa, pyloudnorm as pyln, soundfile as sf

# === Small additions for beats-to-break check ===
GREEN_TARGET_BEATS = 192

def classify_beats_to_break(n, target=GREEN_TARGET_BEATS):
    if n is None:
        return "unknown"
    if n == target:
        return "green"   # exactly target
    if n < target:
        return "red"     # less than target
    return "orange"      # more than target
# ===============================================


def analyze(path):
    # Load audio (mono for simplicity)
    y, sr = librosa.load(path, mono=True)
    duration = len(y) / sr

    # Tempo (BPM) + get beat frames (changed: keep frames)
    # Using onset envelope helps stabilize beat tracking
    hop_length = 512
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=o_env, sr=sr, hop_length=hop_length)

    # Loudness (LUFS)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)

    # Peak level in dBFS
    peak = 20*np.log10(max(1e-12, np.max(np.abs(y))))

    # ---------- Tiny first-break detector (energy drop heuristic) ----------
    beats_before_break = None
    break_time = None

    try:
        if len(beat_frames) >= 32:
            # Per-frame RMS (energy), convert to dB for smoother comparison
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
            rms_db = librosa.power_to_db(rms**2 + 1e-12)

            # Average energy per beat (map frame-level RMS to beat spans)
            per_beat_rms_db = []
            for i in range(len(beat_frames)):
                start = beat_frames[i]
                end = beat_frames[i + 1] if i + 1 < len(beat_frames) else min(len(rms_db), beat_frames[i] + 1)
                if end <= start:
                    # repeat last average if a degenerate span happens
                    per_beat_rms_db.append(per_beat_rms_db[-1] if per_beat_rms_db else float(np.mean(rms_db)))
                else:
                    per_beat_rms_db.append(float(np.mean(rms_db[start:end])))

            # Look for earliest sustained energy drop across an 8-beat window
            look = 8       # beats to compare before vs after
            drop_db = 6.0  # threshold for energy drop in dB
            break_idx = None

            for i in range(look, len(per_beat_rms_db) - look):
                prev_mean = float(np.mean(per_beat_rms_db[i - look:i]))
                next_mean = float(np.mean(per_beat_rms_db[i:i + look]))
                if (prev_mean - next_mean) >= drop_db:
                    break_idx = i
                    break

            if break_idx is not None:
                beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
                beats_before_break = int(break_idx)            # beats BEFORE the break starts
                break_time = float(beat_times[break_idx])      # seconds
    except Exception:
        # If anything goes wrong, we just keep beats_before_break/break_time as None
        pass
    # ---------------------------------------------------------------------

    # Some simple rules for feedback
    dj_notes = []
    if duration < 150:
        dj_notes.append("⚠️ Track is short for club use (<2:30).")
    if loudness > -7.5:
        dj_notes.append("⚠️ Track is very loud, check clipping.")
    if loudness < -10:
        dj_notes.append("⚠️ Track is quiet, could use more limiting.")
    if tempo < 120 or tempo > 155:
        dj_notes.append("⚠️ Tempo outside typical techno range (124–150 BPM).")

    # Color classification vs 192-beat target
    status_color = classify_beats_to_break(beats_before_break, GREEN_TARGET_BEATS)

    return {
        "file": path,
        "duration_sec": round(duration, 2),
        "tempo_bpm": round(float(tempo), 1),
        "loudness_lufs": round(float(loudness), 2),
        "peak_dbfs": round(float(peak), 2),
        "dj_notes": dj_notes,

        # ---- new fields for the UI ----
        "beats_before_break": beats_before_break,       # int or None
        "break_time_sec": round(break_time, 3) if break_time is not None else None,
        "target_beats_to_break": GREEN_TARGET_BEATS,    # 192
        "beats_to_break_status": status_color           # "green" | "red" | "orange" | "unknown"
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python analyze.py "/path/to/your/track.wav"')
        sys.exit(1)

    result = analyze(sys.argv[1])
    print(result)
