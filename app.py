import os
import io
import subprocess
from typing import Dict, Any, List

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# ---------------- CONFIG ----------------
MODEL_DIR = os.environ.get("MODEL_DIR", "audio-auth-best")
MAX_DURATION_SEC = float(os.environ.get("MAX_DURATION_SEC", "5.0"))
MIN_DURATION_SEC = float(os.environ.get("MIN_DURATION_SEC", "0.20"))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))

# Return only Top-K by default to avoid "stacking"
TOP_K = int(os.environ.get("TOP_K", "5"))
RETURN_ALL_PROBS = os.environ.get("RETURN_ALL_PROBS", "0").strip().lower() in ("1", "true", "yes")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# ---------------- LOAD MODEL ON STARTUP ----------------
app = FastAPI(title="Audio Auth Inference", version="1.1")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR)

model.to(DEVICE)
model.eval()

TARGET_SR = int(getattr(feature_extractor, "sampling_rate", 16000))
MAX_SAMPLES = int(TARGET_SR * MAX_DURATION_SEC)
MIN_SAMPLES = int(TARGET_SR * MIN_DURATION_SEC)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _decode_audio_bytes_to_f32_mono(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Two-pass strategy:
    # 1) strict decode
    # 2) lenient decode for mp4/m4a partial-file cases
    base_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-f", "f32le",
        "pipe:1",
    ]

    def run_ffmpeg(cmd):
        try:
            return subprocess.run(cmd, input=file_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="ffmpeg not found. Please install ffmpeg and ensure it's in PATH.")

    # Pass 1: strict
    p = run_ffmpeg(base_cmd)

    # Pass 2: lenient for partial mp4/m4a (tries to salvage audio)
    if p.returncode != 0 or not p.stdout:
        lenient_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            # tolerate some container issues
            "-err_detect", "ignore_err",
            # probe more to find streams
            "-analyzeduration", "200M",
            "-probesize", "200M",
            "-i", "pipe:0",
            "-vn",
            "-ac", "1",
            "-ar", str(TARGET_SR),
            "-f", "f32le",
            "pipe:1",
        ]
        p2 = run_ffmpeg(lenient_cmd)
        if p2.returncode == 0 and p2.stdout:
            p = p2  # salvage success

    if p.returncode != 0 or not p.stdout:
        err = p.stderr.decode("utf-8", errors="ignore").strip()
        err_short = err[:300] if err else "Unknown decode error"

        # If ffmpeg says "partial file", give a clearer API message
        if "partial file" in err.lower() or "invalid data" in err.lower():
            raise HTTPException(
                status_code=400,
                detail="Audio file appears incomplete/corrupted (partial upload). Please re-upload the file or export it again."
            )

        raise HTTPException(status_code=400, detail=f"Could not decode audio (ffmpeg/codec issue?): {err_short}")

    audio = np.frombuffer(p.stdout, dtype=np.float32)

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Decoded audio is empty (file may be corrupted).")

    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if audio.shape[0] < MIN_SAMPLES:
        raise HTTPException(status_code=400, detail=f"Audio too short (< {MIN_DURATION_SEC:.2f}s).")

    if audio.shape[0] > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]

    return audio.astype(np.float32)



def _make_batch(audio_1d: np.ndarray) -> Dict[str, torch.Tensor]:
    """
    Create input_values + attention_mask.
    Pads to MAX_SAMPLES.
    """
    x = np.zeros((1, MAX_SAMPLES), dtype=np.float32)
    attn = np.zeros((1, MAX_SAMPLES), dtype=np.int64)

    L = min(audio_1d.shape[0], MAX_SAMPLES)
    if L > 0:
        x[0, :L] = audio_1d[:L]
        attn[0, :L] = 1

    return {
        "input_values": torch.from_numpy(x).to(DEVICE),
        "attention_mask": torch.from_numpy(attn).to(DEVICE),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": DEVICE,
        "target_sr": TARGET_SR,
        "max_duration_sec": MAX_DURATION_SEC,
        "min_duration_sec": MIN_DURATION_SEC,
        "num_labels": int(model.config.num_labels),
        "top_k": TOP_K,
        "return_all_probs": RETURN_ALL_PROBS,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    # Enforce mp3-only by filename (your requested behavior)
    filename = (file.filename or "").lower()
    if not filename.endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only .mp3 files are accepted (by filename).")

    # Size guard
    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed is {MAX_UPLOAD_MB}MB."
        )

    # Robust decode (works even if bytes are m4a/mp4 but named .mp3)
    audio = _decode_audio_bytes_to_f32_mono(raw)
    batch = _make_batch(audio)

    with torch.no_grad():
        autocast_enabled = torch.cuda.is_available()
        autocast_dtype = torch.bfloat16 if USE_BF16 else torch.float16

        if autocast_enabled:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                outputs = model(**batch)
        else:
            outputs = model(**batch)

        logits = outputs.logits[0].detach().float().cpu().numpy()
        probs = _softmax(logits)

    top_idx = int(np.argmax(probs))
    label = model.config.id2label.get(top_idx, str(top_idx))
    score = float(probs[top_idx])

    # Build distribution sorted desc
    dist: List[Dict[str, Any]] = [
        {"id": i, "label": model.config.id2label.get(i, str(i)), "score": float(p)}
        for i, p in enumerate(probs.tolist())
    ]
    dist.sort(key=lambda x: x["score"], reverse=True)

    resp: Dict[str, Any] = {
        "predicted_label": label,
        "predicted_id": top_idx,
        "score": score,
        "top_k": dist[:TOP_K],
        "audio_seconds_used": float(len(audio) / TARGET_SR),
        "target_sr": TARGET_SR,
    }

    # Avoid "stacking results" unless explicitly requested
    if RETURN_ALL_PROBS:
        resp["all_probs"] = dist

    return JSONResponse(resp)
