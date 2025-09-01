import numpy as np
import librosa, pyloudnorm as pyln
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse

app = FastAPI()


# --- Robust Tempo Detection ---
def robust_tempo(y, sr, mode="dj"):
    if mode == "streaming":
        preferred_min, preferred_max = 60.0, 200.0
    else:
        preferred_min, preferred_max = 124.0, 150.0  # DJ range

    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    cands = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                               aggregate=None, max_tempo=preferred_max * 2)

    if isinstance(cands, np.ndarray):
        cands = cands.tolist()

    corrected = []
    for c in cands[:5]:
        for mult in (0.5, 1.0, 2.0):
            t = c * mult
            if preferred_min <= t <= preferred_max:
                corrected.append(t)

    if corrected:
        return float(np.median(corrected))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    while tempo < preferred_min:
        tempo *= 2.0
    while tempo > preferred_max:
        tempo /= 2.0
    return float(tempo)


# --- Track Analysis ---
def analyze_track(path, mode="dj"):
    y, sr = librosa.load(path, mono=True)
    duration = len(y) / sr

    tempo = robust_tempo(y, sr, mode)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    peak = 20 * np.log10(max(1e-12, np.max(np.abs(y))))

    notes = []
    if mode == "dj":
        if duration < 150:
            notes.append("⚠️ Short for club use (<2:30).")
        if tempo < 124 or tempo > 150:
            notes.append("⚠️ Tempo outside DJ range.")
    else:  # streaming mode
        if duration < 120:
            notes.append("⚠️ Very short for streaming platforms.")

    if loudness > -7.5:
        notes.append("⚠️ Loud; check clipping.")
    if loudness < -10:
        notes.append("⚠️ Quiet, could use more limiting.")

    return {
        "duration_sec": round(duration, 2),
        "tempo_bpm": round(float(tempo), 1),
        "loudness_lufs": round(float(loudness), 2),
        "peak_dbfs": round(float(peak), 2),
        "notes": notes
    }


# --- Web Routes ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
      <head>
        <title>DJ Track Analyzer</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; text-align: center; background: #111; color: #eee; }
          h1 { color: #0f0; }
          form { margin-top: 20px; }
          input, select { margin: 10px; padding: 8px; }
          .report { margin-top: 40px; padding: 20px; background: #222; border-radius: 8px; text-align: left; display: inline-block; }
        </style>
      </head>
      <body>
        <h1>DJ Track Analyzer</h1>
        <form action="/analyze" enctype="multipart/form-data" method="post">
          <input type="file" name="file" required>
          <select name="mode">
            <option value="dj">DJ Mix Ready</option>
            <option value="streaming">Streaming Ready</option>
          </select>
          <button type="submit">Analyze</button>
        </form>
      </body>
    </html>
    """


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...), mode: str = Form("dj")):
    path = f"temp_{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    result = analyze_track(path, mode)
    minutes = int(result["duration_sec"] // 60)
    seconds = int(result["duration_sec"] % 60)

    notes_html = "<br>".join(result["notes"]) if result["notes"] else "✅ No major issues detected."

    return f"""
    <html>
      <head>
        <title>DJ Track Analyzer - Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 40px; text-align: center; background: #111; color: #eee; }}
          h1 {{ color: #0f0; }}
          .report {{ margin-top: 40px; padding: 20px; background: #222; border-radius: 8px; text-align: left; display: inline-block; }}
        </style>
      </head>
      <body>
        <h1>DJ Track Analyzer</h1>
        <div class="report">
          <h2>Analysis Report</h2>
          <p><b>File:</b> {file.filename}</p>
          <p><b>Mode:</b> {"DJ Mix Ready" if mode=="dj" else "Streaming Ready"}</p>
          <p><b>Length:</b> {minutes} min {seconds} sec</p>
          <p><b>Tempo:</b> {result["tempo_bpm"]} BPM</p>
          <p><b>Loudness:</b> {result["loudness_lufs"]} LUFS</p>
          <p><b>Peak:</b> {result["peak_dbfs"]} dBFS</p>
          <p><b>Notes:</b><br>{notes_html}</p>
        </div>
        <form action="/analyze" enctype="multipart/form-data" method="post" style="margin-top:30px;">
          <input type="file" name="file" required>
          <select name="mode">
            <option value="dj">DJ Mix Ready</option>
            <option value="streaming">Streaming Ready</option>
          </select>
          <button type="submit">Analyze Another Track</button>
        </form>
      </body>
    </html>
    """
