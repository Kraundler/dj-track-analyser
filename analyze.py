import sys, numpy as np
import librosa, pyloudnorm as pyln, soundfile as sf

def analyze(path):
    # Load audio (mono for simplicity)
    y, sr = librosa.load(path, mono=True)
    duration = len(y) / sr

    # Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Loudness (LUFS)
    meter = pyln.Meter(sr)  
    loudness = meter.integrated_loudness(y)

    # Peak level in dBFS
    peak = 20*np.log10(max(1e-12, np.max(np.abs(y))))

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

    return {
        "file": path,
        "duration_sec": round(duration, 2),
        "tempo_bpm": round(float(tempo), 1),
        "loudness_lufs": round(float(loudness), 2),
        "peak_dbfs": round(float(peak), 2),
        "dj_notes": dj_notes
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python analyze.py "/path/to/your/track.wav"')
        sys.exit(1)

    result = analyze(sys.argv[1])
    print(result)

