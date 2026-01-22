import numpy as np
import librosa

def analyze_spectrogram(path: str) -> dict:
    """
    Analyze audio quality using spectrogram characteristics.
    Returns metrics:
        - hf_energy_ratio: high-frequency energy ratio (0–1)
        - spectral_entropy: flatness/chaos (0–1)
        - lowpass_detected: True if audio seems lowpassed (<15kHz)
        - noise_detected: True if over-noisy / non-music
    """

    y, sr = librosa.load(path, sr=44100, mono=True, duration=20)

    #  Calculate STFT 
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    #  Energy by frequency band 
    total_energy = np.sum(S)
    if total_energy == 0:
        return {"invalid": True}

    hf_band = S[freqs > 14000]  # above 14 kHz
    hf_ratio = float(np.sum(hf_band) / total_energy)

    #  Spectral Entropy 
    psd = np.mean(S, axis=1)
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-9))

    # Normalize entropy to 0–1
    spectral_entropy = float(spectral_entropy / np.log2(len(psd_norm)))

    #  Heuristics 
    lowpass_detected = hf_ratio < 0.015        # almost no high end → bad source
    noise_detected = spectral_entropy > 0.82   # chaotic noisy audio → live/concert/video audio

    return {
        "hf_energy_ratio": hf_ratio,
        "spectral_entropy": spectral_entropy,
        "lowpass_detected": lowpass_detected,
        "noise_detected": noise_detected,
        "invalid": False,
    }


def looks_like_bad_source(metrics: dict) -> bool:
    """
    Decide if the audio is low-quality or "video audio".
    """

    if metrics.get("invalid"):
        return True

    lowpass = metrics["lowpass_detected"]
    noisy = metrics["noise_detected"]

    # Combined logic
    if lowpass and noisy:
        return True  # Movie audio, live audio, or terrible upload

    # Very aggressive lowpass (128kbps or fake 320)
    if metrics["hf_energy_ratio"] < 0.01:
        return True

    # Extremely noisy (crowd, talking, reverb)
    if metrics["spectral_entropy"] > 0.87:
        return True

    return False
