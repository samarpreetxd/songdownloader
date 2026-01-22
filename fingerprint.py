import subprocess
import json
import numpy as np
from pathlib import Path
from config_utils import get_app_config

# Load config
CFG = get_app_config()

# Load fpcalc path from config, fallback to "fpcalc" in PATH
FPCALC = (
    CFG.get("fingerprinting", {})
       .get("fpcalc_path", "fpcalc")
)

def chromaprint_fingerprint(file: Path) -> np.ndarray | None:
    """
    Runs fpcalc to produce a Chromaprint fingerprint.
    Uses the configured fpcalc path from config.json or PATH.
    Returns a numpy array of ints.
    """
    try:
        result = subprocess.run(
            [FPCALC, "-json", str(file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        fp_str = data.get("fingerprint")
        if not fp_str:
            return None

        # Convert space-separated ints into numpy array
        fp = np.fromstring(fp_str, dtype=int, sep=' ')
        return fp

    except Exception:
        return None


def fingerprint_similarity(fp1: np.ndarray | None, fp2: np.ndarray | None) -> float:
    """
    Computes similarity between two fingerprints using a match ratio.
    Range: 0 to 1
    """
    if fp1 is None or fp2 is None:
        return 0.0

    # Trim to same length
    n = min(len(fp1), len(fp2))
    if n == 0:
        return 0.0

    fp1 = fp1[:n]
    fp2 = fp2[:n]

    matches = np.sum(fp1 == fp2)
    return float(matches / n)
