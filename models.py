from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Track:
    # Basic metadata
    title: str
    artists: List[str]
    album: Optional[str] = None
    duration_ms: Optional[int] = None
    track_number: Optional[int] = None
    total_tracks: Optional[int] = None
    release_year: Optional[int] = None
    genres: List[str] = field(default_factory=list)
    cover_url: Optional[str] = None
    id: Optional[str] = None
    preview_url: Optional[str] = None

    # Source-related
    source: str = "unknown"            # "spotify" or "soundcloud" or "unknown"
    direct_url: Optional[str] = None   # For direct SC download via yt-dlp
    waveform_url: Optional[str] = None # Optional SC waveform

    @property
    def artist_str(self) -> str:
        return ", ".join(self.artists) if self.artists else ""
