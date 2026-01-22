from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from yt_dlp import YoutubeDL

from config_utils import get_app_config
from models import Track

#  simple retry utilities 


def simple_retry(max_attempts: int = 3, base_wait: float = 1.0, max_wait: float = 10.0):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = base_wait
            while True:
                try:
                    return fn(*args, **kwargs)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    time.sleep(min(delay, max_wait))
                    delay *= 2

        return wrapper

    return decorator


try:
    from tenacity import retry as _tenacity_retry, stop_after_attempt, wait_exponential  # type: ignore

    _TENACITY = True
except Exception:
    _tenacity_retry = None
    _TENACITY = False


if _TENACITY:

    def retryable(fn=None, *, attempts=3):
        if fn is None:
            return lambda f: _tenacity_retry(
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=1, min=1, max=10),
            )(f)
        return _tenacity_retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
        )(fn)

else:

    def retryable(fn=None, *, attempts=3):
        if fn is None:
            return lambda f: simple_retry(max_attempts=attempts)(f)
        return simple_retry(max_attempts=attempts)(fn)


try:
    from ratelimit import limits, sleep_and_retry  # type: ignore

    _RATE_LIMIT = True
except Exception:
    limits = None
    sleep_and_retry = None
    _RATE_LIMIT = False


if _RATE_LIMIT:

    @sleep_and_retry
    @limits(calls=10, period=1)
    def _rl_pass():
        return True

else:

    def _rl_pass():
        return True


CRED_JSON = Path("credentials.json")


def load_credentials() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Returns (spotify_client_id, spotify_client_secret, spotify_redirect_uri, youtube_api_key).
    Order of precedence:
    1) environment variables
    2) config.json
    3) credentials.json (legacy)
    """
    cfg = get_app_config()

    spotify_cfg = cfg.get("spotify", {})
    youtube_cfg = cfg.get("youtube", {})

    env_id = os.getenv("SPOTIFY_CLIENT_ID") or spotify_cfg.get("client_id") or ""
    env_secret = os.getenv("SPOTIFY_CLIENT_SECRET") or spotify_cfg.get("client_secret") or ""
    env_redirect = os.getenv("SPOTIFY_REDIRECT_URI") or spotify_cfg.get("redirect_uri") or ""
    env_yt = os.getenv("YOUTUBE_API_KEY") or youtube_cfg.get("api_key") or ""

    client_id = env_id or None
    client_secret = env_secret or None
    redirect_uri = env_redirect or None
    yt_key = env_yt or None

    # Legacy credentials.json (optional)
    if (not client_id or not client_secret) and CRED_JSON.exists():
        try:
            data = json.loads(CRED_JSON.read_text("utf-8"))
            client_id = client_id or data.get("client_id")
            client_secret = client_secret or data.get("client_secret")
            redirect_uri = redirect_uri or data.get("redirect_uri")
            yt_key = yt_key or data.get("youtube_api_key")
        except Exception as e:
            logging.error("Failed reading credentials.json: %s", e)

    if not client_id or not client_secret:
        logging.warning("Spotify credentials not fully configured.")

    return client_id, client_secret, redirect_uri, yt_key


def get_spotify_client() -> spotipy.Spotify:
    client_id, client_secret, _redir, _yt = load_credentials()
    if not client_id or not client_secret:
        logging.error(
            "Spotify credentials missing. Configure env variables, config.json, or credentials.json."
        )
        raise SystemExit(1)
    creds = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(client_credentials_manager=creds)


def parse_playlist_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    if "playlist" in s:
        m = re.search(r"playlist[/:]([A-Za-z0-9]+)", s)
        if m:
            return m.group(1)
    if s.startswith("spotify:playlist:"):
        return s.split(":")[-1]
    return s


#  Spotify playlist fetching 


@retryable(attempts=3)
def fetch_spotify_playlist_tracks(sp: spotipy.Spotify, playlist_id: str) -> List[Track]:
    items: List[Track] = []
    fields = "items.track(id,name,artists(id,name),album(name,images,release_date,total_tracks),duration_ms,preview_url),next"
    res = sp.playlist_items(playlist_id, fields=fields, limit=100)

    while res:
        _rl_pass()
        for it in res.get("items", []):
            t = it.get("track")
            if not t:
                continue

            artists = [a.get("name") for a in t.get("artists", []) if a]
            artist_ids = [a.get("id") for a in t.get("artists", []) if a and a.get("id")]

            album = t.get("album") or {}
            imgs = album.get("images", [])
            cover_url = imgs[0]["url"] if imgs else None

            release_date = album.get("release_date") or ""
            release_year = release_date.split("-")[0] if release_date else None

            preview_url = t.get("preview_url")
            total_tracks = album.get("total_tracks")

            tr = Track(
                id=t.get("id"),
                title=t.get("name"),
                artists=artists,
                album=album.get("name"),
                duration_ms=t.get("duration_ms") or 0,
                cover_url=cover_url,
                release_year=int(release_year) if release_year and release_year.isdigit() else None,
                total_tracks=total_tracks,
                genres=[],
                source="spotify",
                preview_url=preview_url,   # Store preview URL from Spotify
            )

            # Attach artist IDs temporarily
            setattr(tr, "_artist_ids", artist_ids)

            # For Chromaprint
            tr.preview_fp_path = None
            tr.preview_fp = None

            items.append(tr)

        res = sp.next(res) if res.get("next") else None

    enrich_genres(sp, items)
    return items


@retryable(attempts=3)
def enrich_genres(sp: spotipy.Spotify, tracks: List[Track]) -> None:
    unique_ids: List[str] = []
    for tr in tracks:
        for aid in (getattr(tr, "_artist_ids", []) or []):
            if aid and aid not in unique_ids:
                unique_ids.append(aid)
    for i in range(0, len(unique_ids), 50):
        _rl_pass()
        batch = unique_ids[i : i + 50]
        data = sp.artists(batch)
        id_to_genres = {a["id"]: a.get("genres", []) for a in data.get("artists", [])}
        for tr in tracks:
            gset = set(tr.genres or [])
            for aid in (getattr(tr, "_artist_ids", []) or []):
                gset.update(id_to_genres.get(aid, []))
            tr.genres = sorted(gset)


def fetch_spotify_playlist_with_name(sp: spotipy.Spotify, playlist_id: str) -> Tuple[List[Track], str]:
    meta = sp.playlist(playlist_id, fields="name")
    pname = (meta or {}).get("name") or playlist_id
    tracks = fetch_spotify_playlist_tracks(sp, playlist_id)
    return tracks, pname


#  SoundCloud playlist fetching via yt-dlp 


def _extract_with_yt_dlp(url: str, verbose: bool = False) -> Dict[str, Any]:
    opts = {
        "quiet": not verbose,
        "skip_download": True,
        "noplaylist": False,
        "extract_flat": False,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info


def fetch_soundcloud_playlist(url: str, verbose: bool = False) -> Tuple[List[Track], str]:
    """
    Fetches SoundCloud playlist/“set” using yt-dlp only.
    No SoundCloud API keys required.
    """
    info = _extract_with_yt_dlp(url, verbose=verbose)
    playlist_title = info.get("title") or "soundcloud_playlist"

    entries = info.get("entries") or []
    if not entries:
        entries = [info]

    total = len(entries)
    tracks: List[Track] = []

    for e in entries:
        if not e:
            continue
        uploader = e.get("uploader") or e.get("uploader_id") or "Unknown"
        duration_s = e.get("duration") or 0
        thumbnail = e.get("thumbnail") or info.get("thumbnail")
        tags = e.get("tags") or info.get("tags") or []
        direct_url = e.get("url") or e.get("webpage_url") or e.get("original_url")
        waveform_url = e.get("waveform_url")

        tr = Track(
            id=str(e.get("id") or e.get("url") or ""),
            title=e.get("title") or "Untitled",
            artists=[uploader],
            album=playlist_title,
            duration_ms=int(duration_s * 1000),
            cover_url=thumbnail,
            release_year=None,
            total_tracks=total,
            genres=list(tags) if tags else [],
            source="soundcloud",
            direct_url=direct_url,
            waveform_url=waveform_url,
        )
        tracks.append(tr)

    logging.info("Fetched %d tracks from SoundCloud playlist '%s'.", len(tracks), playlist_title)
    return tracks, playlist_title


#  unified helper 


def detect_source(url_or_id: str) -> str:
    s = url_or_id.strip().lower()
    if "soundcloud.com" in s:
        return "soundcloud"
    return "spotify"


def get_playlist_tracks(url_or_id: str, verbose: bool = False) -> Tuple[List[Track], str, str]:
    """
    Returns (tracks, playlist_name, source) where source is 'spotify' or 'soundcloud'.
    """
    source = detect_source(url_or_id)
    if source == "spotify":
        sp = get_spotify_client()
        playlist_id = parse_playlist_id(url_or_id)
        tracks, pname = fetch_spotify_playlist_with_name(sp, playlist_id)
        logging.info("Detected Spotify playlist '%s' (%d tracks)", pname, len(tracks))
        return tracks, pname, "spotify"
    else:
        tracks, pname = fetch_soundcloud_playlist(url_or_id, verbose=verbose)
        logging.info("Detected SoundCloud playlist '%s' (%d tracks)", pname, len(tracks))
        return tracks, pname, "soundcloud"
