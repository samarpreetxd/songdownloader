from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict

from models import Track


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def score_candidate_for_track(candidate: Dict[str, Any], track: Track) -> float:
    """
    Heuristic "ML-style" scorer.
    Higher score = better match. Typical good matches 50–100.
    """
    title = candidate.get("title") or ""
    channel = candidate.get("channel") or ""
    cand_dur = candidate.get("duration") or 0
    target_dur = (track.duration_ms or 0) / 1000.0

    # 1) Title similarity
    full_ref = f"{track.artist_str} - {track.title}".strip()
    sim1 = _title_similarity(title, track.title or "")
    sim2 = _title_similarity(title, full_ref)
    title_score = max(sim1, sim2) * 60.0  # up to ~60 pts

    # 2) Duration closeness
    dur_score = 0.0
    if target_dur > 0 and cand_dur > 0:
        diff = abs(cand_dur - target_dur)
        # Allow up to 20s grace, then decay
        if diff <= 5:
            dur_score = 25.0
        elif diff <= 10:
            dur_score = 20.0
        elif diff <= 20:
            dur_score = 12.0
        elif diff <= 40:
            dur_score = 5.0
        else:
            dur_score = max(0.0, 10.0 - diff / 10.0)

    # 3) Channel / artist hints
    channel_score = 0.0
    norm_channel = _normalize(channel)
    for artist in track.artists:
        if not artist:
            continue
        if _normalize(artist) in norm_channel:
            channel_score = 15.0
            break

    # 4) Penalties for suspicious words
    bad_words = ["live", "remix", "cover", "nightcore", "slowed", "reverb", "8d"]
    for w in bad_words:
        if re.search(rf"\b{w}\b", _normalize(title)):
            channel_score -= 10.0

    score = title_score + dur_score + channel_score
    return float(score)
