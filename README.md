# Audio Auth Inference (FastAPI)

A FastAPI service that runs **audio classification** inference using a model stored locally (default folder: `audio-auth-best`).

It accepts an uploaded **`.mp3`** file, decodes it with **ffmpeg**, resamples to the model’s target sample rate,
clips to a max duration, and returns the predicted label + top-k probabilities.

---

## Project Structure

```
.
├── app.py
├── requirements.txt
└── audio-auth-best/          # HuggingFace model folder (~1.4GB)
```

> **Note:** Because `audio-auth-best/` is large, push it to GitHub using **Git LFS**.

---

## Requirements

- Python 3.9+
- `ffmpeg` installed and available in `PATH`
- The local model folder exists (default: `audio-auth-best/`)

Install Python deps:

```bash
pip install -r requirements.txt
```

Install ffmpeg (Ubuntu/Debian):

```bash
sudo apt update && sudo apt install -y ffmpeg
```

---

## Run the API

```bash
# optional: set custom model directory
export MODEL_DIR="audio-auth-best"

uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict`

---

## Usage

### Health

```bash
curl http://localhost:8000/health
```

### Predict (upload .mp3)

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample.mp3"
```

Example response:

```json
{
  "predicted_label": "label_name",
  "predicted_id": 0,
  "score": 0.92,
  "top_k": [
    {"id": 0, "label": "label_name", "score": 0.92}
  ],
  "audio_seconds_used": 5.0,
  "target_sr": 16000
}
```

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---:|---|
| `MODEL_DIR` | `audio-auth-best` | Path to local HF model directory |
| `MAX_DURATION_SEC` | `5.0` | Maximum audio duration used |
| `MIN_DURATION_SEC` | `0.20` | Minimum allowed duration |
| `MAX_UPLOAD_MB` | `25` | Max upload file size |
| `TOP_K` | `5` | Number of top predictions to return |
| `RETURN_ALL_PROBS` | `0` | If `1/true/yes`, include full probability list |
