import librosa
import numpy as np
from pathlib import Path
import requests
from io import BytesIO

def extract_mfcc(path: Path, duration: float = 12.0) -> np.ndarray:
    """Extract MFCC embedding of the first X seconds."""
    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1)  # compress to 20 dims


def fetch_spotify_preview(url: str) -> np.ndarray | None:
    """Download Spotify 30s preview MP3 and extract embeddings."""
    if not url:
        return None

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        audio = BytesIO(r.content)
        y, sr = librosa.load(audio, sr=22050, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return np.mean(mfcc, axis=1)
    except Exception:
        return None


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity: higher = better."""
    if a is None or b is None:
        return 0.0
    dot = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / denom)
