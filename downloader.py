from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from mutagen.easyid3 import EasyID3
from mutagen.id3 import (
    APIC,
    ID3,
    ID3NoHeaderError,
    TALB,
    TCON,
    TDRC,
    TIT2,
    TPE1,
    TPE2,
    TPOS,
    TRCK,
    TSOA,
    TSOP,
    TSOT,
    TXXX,
    TYER,
)
from mutagen.mp4 import MP4, MP4Cover
from mutagen.flac import FLAC, Picture
from mutagen.aiff import AIFF
from mutagen.wave import WAVE
from yt_dlp import YoutubeDL

from models import Track
from ml_search import score_candidate_for_track
from source import get_playlist_tracks

# Optional libs 
try:  # progress UI
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    _RICH = True
except Exception:
    from tqdm import tqdm  

    _RICH = False

try:  # async optional
    import asyncio
    import aiohttp  

    _ASYNC_OK = True
except Exception:
    asyncio = None
    aiohttp = None
    _ASYNC_OK = False

try:
    from pathvalidate import sanitize_filename as _sanitize_filename  
except Exception:
    _sanitize_filename = None

try:
    from PIL import Image  
    from io import BytesIO  

    _PIL_OK = True
except Exception:
    Image = None
    BytesIO = None
    _PIL_OK = False

INVALID_FN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

def looks_like_music_video(title: str, channel: str) -> bool:
    """Return True if video appears to be a music video rather than audio."""
    t = title.lower()
    c = channel.lower()

    bad_words = [
        "official video",
        "music video",
        "mv",
        "cover",
        "promo",
        "video out now",
        "clip officiel",
        "clip video",
        "videoclip",
        "teaser",
        "trailer",
        "live performance",
        "performance video",
    ]

    if any(w in t for w in bad_words):
        return True

    if "(official video)" in t or "[mv]" in t:
        return True

    suspicious_channels = ["studios", "films", "movies", "trailer"]
    if any(w in c for w in suspicious_channels):
        return True

    return False

def safe_filename(name: str, maxlen: int = 120) -> str:
    name = (name or "").strip()
    if _sanitize_filename:
        cleaned = _sanitize_filename(name)
    else:
        cleaned = INVALID_FN.sub("", name)
        cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) > maxlen:
        cut = cleaned[:maxlen]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        cleaned = cut
    return cleaned or "untitled"


class HybridSmartSearch:
    """
    Uses YouTube Data API v3 (optional) + yt-dlp fallback.
    Ranking is driven by ml_search.score_candidate_for_track().
    """

    def __init__(self, api_key: Optional[str], verbose: bool = False):
        self.api_key = api_key
        self.verbose = verbose
        self.session = requests.Session()

    def _log(self, msg: str):
        if self.verbose:
            logging.info(f"[SmartSearch] {msg}")

    def _iso_to_seconds(self, duration: str) -> int:
        m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
        if not m:
            return 0
        h = int(m.group(1) or 0)
        mm = int(m.group(2) or 0)
        ss = int(m.group(3) or 0)
        return h * 3600 + mm * 60 + ss

    def _api_candidates(self, query: str) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        search_url = (
            "https://www.googleapis.com/youtube/v3/search"
            f"?part=snippet&type=video&maxResults=10&q={requests.utils.quote(query)}&key={self.api_key}"
        )
        r = self.session.get(search_url, timeout=10)
        if r.status_code != 200:
            self._log(f"YouTube API search failed: {r.status_code}")
            return []
        items = r.json().get("items", [])
        ids = ",".join(i.get("id", {}).get("videoId", "") for i in items if i.get("id"))
        if not ids:
            return []

        details_url = (
            "https://www.googleapis.com/youtube/v3/videos"
            f"?part=contentDetails,snippet&id={ids}&key={self.api_key}"
        )
        r2 = self.session.get(details_url, timeout=10)
        if r2.status_code != 200:
            self._log(f"YouTube API details failed: {r2.status_code}")
            return []
        videos = r2.json().get("items", [])

        out: List[Dict[str, Any]] = []
        for v in videos:
            snippet = v.get("snippet", {})
            duration = self._iso_to_seconds(v.get("contentDetails", {}).get("duration", ""))
            out.append(
                {
                    "title": snippet.get("title", ""),
                    "channel": snippet.get("channelTitle", ""),
                    "video_id": v.get("id"),
                    "duration": duration,
                }
            )
        return out
    def get_ranked_candidates(self, query: str, track: Track) -> list:
        """Get up to 10 ranked YouTube candidates using API."""
        try:
            cands = self._api_candidates(query)
        except Exception:
            cands = []

        if not cands:
            return []

        ranked = [
            (score_candidate_for_track(c, track), c)
            for c in cands
        ]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked]

    def _yt_dlp_fallback(self, query: str, target_duration: int) -> Optional[str]:
        try:
            ydl_opts = {
                "quiet": True,
                "default_search": "ytsearch10",
                "extract_flat": True,
                "noplaylist": True,
                "concurrent_fragment_downloads": 8,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(query, download=False)
            entries = (info or {}).get("entries") or []
            best = None
            best_diff = 1e9
            for e in entries:
                dur = e.get("duration") or 0
                diff = abs(dur - target_duration) if target_duration else 9999
                if diff < best_diff:
                    best_diff = diff
                    best = e
            if best and best.get("url"):
                return best["url"]
        except Exception as e:
            self._log(f"yt-dlp fallback failed: {e}")
        return None

    def search_url(self, query: str, track: Track) -> Optional[str]:
        duration_s = (track.duration_ms or 0) // 1000

        try:
            cands = self._api_candidates(query)
        except Exception as e:
            self._log(f"API candidates error: {e}")
            cands = []

        if cands:
            scored = [
                (score_candidate_for_track(c, track), c)
                for c in cands
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_score, top = scored[0]
            self._log(
                f"Selected via ML+API: {top['title']} | {top['channel']} | score={top_score:.2f}"
            )
            if top_score >= 40.0:
                return f"https://www.youtube.com/watch?v={top['video_id']}"

        return self._yt_dlp_fallback(query, duration_s)


class YTDLPWrapper:
    def __init__(
        self,
        ffmpeg_path: Optional[str],
        verbose: bool,
        smart: Optional[HybridSmartSearch] = None,
        codec: str = "mp3",
        max_video_seconds: Optional[int] = None,
        duration_tolerance_seconds: int = 45,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.verbose = verbose
        self.smart = smart
        self.codec = codec if codec in ("mp3", "m4a", "flac", "alac", "wav", "aiff") else "mp3"
        self.max_video_seconds = max_video_seconds
        self.duration_tolerance_seconds = max(0, int(duration_tolerance_seconds))

    def _final_extension(self) -> str:
        mapping = {
            "mp3": "mp3",
            "m4a": "m4a",
            "alac": "m4a",  # ALAC in MP4 container
            "flac": "flac",
            "wav": "wav",
            "aiff": "aiff",
        }
        return mapping.get(self.codec, "mp3")

    def _opts(
        self,
        outtmpl: str,
        bitrate_kbps: int,
        format_expr: Optional[str] = None,
        expected_duration_s: Optional[int] = None,
    ) -> Dict[str, Any]:
        if format_expr:
            fmt = format_expr
        else:
            fmt = "bestaudio/best"

        opts: Dict[str, Any] = {
            "format": fmt,
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": not self.verbose,
            "no_warnings": True,
            "default_search": "ytsearch1",
            "retries": 3,
            "ignoreerrors": True,
            "noprogress": True,
            "http_chunk_size": 10 * 1024 * 1024,
            "concurrent_fragment_downloads": 4,
            "source_address": "0.0.0.0",
            "extractor_args": {"youtube": {"player_client": ["android"]}},
            "prefer_ffmpeg": True,
        }

        postprocessors: List[Dict[str, Any]] = []

        if self.codec in ("mp3", "m4a"):
            postprocessors.append(
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": self.codec,
                    "preferredquality": str(bitrate_kbps),
                }
            )
        else:
            postprocessors.append(
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": self.codec,
                    "preferredquality": "0",
                }
            )

        opts["postprocessors"] = postprocessors

        # Duration constraints
        max_len = self.max_video_seconds
        tol = self.duration_tolerance_seconds
        exp = expected_duration_s

        if max_len is not None or exp is not None:
            def _match_filter(info):
                dur = info.get("duration")
                if dur is None:
                    return None
                if max_len is not None and dur > max_len:
                    return f"duration {dur}s exceeds max {max_len}s"
                if exp is not None:
                    low = max(1, exp - tol)
                    high = exp + tol
                    if dur < low or dur > high:
                        return f"duration {dur}s outside [{low},{high}]"
                return None

            opts["match_filter"] = _match_filter

        if self.ffmpeg_path:
            opts["ffmpeg_location"] = self.ffmpeg_path

        try:
            from shutil import which as _which

            if _which("aria2c"):
                opts["external_downloader"] = "aria2c"
                opts["external_downloader_args"] = ["-x", "8", "-s", "8", "-k", "2M"]
        except Exception:
            pass

        return opts

    def _download_url(
        self,
        url: str,
        out_path_template: str,
        bitrate_kbps: int,
        expected_duration_s: Optional[int] = None,
    ) -> Optional[Path]:
        out_dir = Path(out_path_template).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with YoutubeDL(self._opts(out_path_template, bitrate_kbps, expected_duration_s=expected_duration_s)) as ydl:
            ydl.extract_info(url, download=True)
        final_ext = self._final_extension()
        final_path = Path(str(out_path_template).replace("%(ext)s", final_ext))
        return final_path if final_path.exists() else None

    def download_to_audio(
        self,
        search_query: str,
        out_path_template: str,
        bitrate_kbps: int,
        track: Optional[Track] = None,
        enable_smart: bool = False,
    ) -> Optional[Path]:
        # Smart search first (if enabled)
        if enable_smart and self.smart and track is not None:
            url = self.smart.search_url(search_query, track)
            if url:
                try:
                    exp = (track.duration_ms or 0) // 1000 if track else None
                    return self._download_url(url, out_path_template, bitrate_kbps, expected_duration_s=exp)
                except Exception:
                    pass

        # Legacy/basic yt-dlp search
        out_dir = Path(out_path_template).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        exp = (track.duration_ms or 0) // 1000 if track else None

        with YoutubeDL(self._opts(out_path_template, bitrate_kbps, expected_duration_s=exp)) as ydl:
            info = ydl.extract_info(search_query, download=True)
            if isinstance(info, dict) and info.get("entries"):
                info = info["entries"][0]

        final_ext = self._final_extension()
        final_path = Path(str(out_path_template).replace("%(ext)s", final_ext))
        return final_path if final_path.exists() else None


class Tagger:
    """
    Pro tagger for MP3 (ID3v2.x), M4A/ALAC (MP4), FLAC (Vorbis), WAV/AIFF (ID3).
    """

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()

    # cover fetching & processing 

    def _fetch_cover_bytes(self, url: Optional[str]) -> Optional[bytes]:
        if not url:
            return None
        try:
            r = self.session.get(url, timeout=10)
            if r.status_code != 200:
                return None
            data = r.content
            if not data:
                return None
            return data
        except Exception:
            return None

    def _process_image_for_mp3(
        self, raw: bytes
    ) -> Tuple[Optional[bytes], Optional[bytes], str]:
        """
        Returns (full_img_bytes, thumb_img_bytes, mime).
        If Pillow is not available, full_img_bytes = raw, thumb = None, mime guessed.
        """
        if not _PIL_OK:
            mime = "image/jpeg"
            return raw, None, mime

        try:
            img = Image.open(BytesIO(raw))
            mime = "image/jpeg"
            if img.format == "PNG":
                mime = "image/png"

            if img.mode in ("P", "RGBA"):
                bg = Image.new("RGB", img.size, (0, 0, 0))
                bg.paste(img, mask=img.split()[-1])
                img = bg
                mime = "image/jpeg"
            elif img.mode != "RGB":
                img = img.convert("RGB")
                mime = "image/jpeg"

            # High-res cover
            max_dim = 1200
            img_full = img.copy()
            img_full.thumbnail((max_dim, max_dim), Image.LANCZOS)
            buf_full = BytesIO()
            fmt_full = "JPEG" if mime == "image/jpeg" else "PNG"
            img_full.save(buf_full, fmt_full, quality=90)
            full_bytes = buf_full.getvalue()

            # Thumbnail
            img_thumb = img.copy()
            img_thumb.thumbnail((300, 300), Image.LANCZOS)
            buf_thumb = BytesIO()
            img_thumb.save(buf_thumb, fmt_full, quality=85)
            thumb_bytes = buf_thumb.getvalue()

            return full_bytes, thumb_bytes, mime
        except Exception:
            return raw, None, "image/jpeg"

    def _process_image_for_mp4(self, raw: bytes) -> Tuple[bytes, int]:
        """
        Returns (img_bytes, MP4Cover.FORMAT_*).
        If Pillow is available, normalize; otherwise guess JPEG.
        """
        if not _PIL_OK:
            return raw, MP4Cover.FORMAT_JPEG

        try:
            img = Image.open(BytesIO(raw))
            fmt = img.format.upper()
            if fmt == "PNG":
                cover_fmt = MP4Cover.FORMAT_PNG
            else:
                cover_fmt = MP4Cover.FORMAT_JPEG

            if img.mode in ("P", "RGBA"):
                bg = Image.new("RGB", img.size, (0, 0, 0))
                bg.paste(img, mask=img.split()[-1])
                img = bg
                cover_fmt = MP4Cover.FORMAT_JPEG
            elif img.mode != "RGB":
                img = img.convert("RGB")
                cover_fmt = MP4Cover.FORMAT_JPEG

            max_dim = 1200
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            buf = BytesIO()
            if cover_fmt == MP4Cover.FORMAT_PNG:
                img.save(buf, "PNG")
            else:
                img.save(buf, "JPEG", quality=90)
            return buf.getvalue(), cover_fmt
        except Exception:
            return raw, MP4Cover.FORMAT_JPEG

    #  MP3 / WAV / AIFF tagging (ID3) 

    def _ensure_id3(self, path: Path) -> ID3:
        try:
            return ID3(str(path))
        except ID3NoHeaderError:
            id3 = ID3()
            id3.save(str(path), v2_version=3)
            return ID3(str(path))

    def _make_sort_name(self, name: str) -> str:
        if not name:
            return ""
        tokens = name.split()
        if tokens[0].lower() in ("the", "a", "an") and len(tokens) > 1:
            return f'{" ".join(tokens[1:])}, {tokens[0]}'
        return name

    def _tag_mp3_like(
        self,
        media_path: Path,
        track: Track,
        index: int,
        cover_bytes: Optional[bytes],
        cover_thumb: Optional[bytes],
        cover_mime: str,
    ) -> None:
        id3 = self._ensure_id3(media_path)

        id3["TIT2"] = TIT2(encoding=3, text=track.title or "")
        id3["TPE1"] = TPE1(encoding=3, text=track.artists or [])
        id3["TPE2"] = TPE2(encoding=3, text=track.artist_str or "")
        if track.album:
            id3["TALB"] = TALB(encoding=3, text=track.album)

        total_tracks = track.total_tracks or 0
        if total_tracks > 0:
            trk_str = f"{index}/{total_tracks}"
        else:
            trk_str = str(index)
        id3["TRCK"] = TRCK(encoding=3, text=trk_str)
        id3["TPOS"] = TPOS(encoding=3, text="1/1")

        if track.genres:
            joined_genres = "; ".join(track.genres[:4])
            id3["TCON"] = TCON(encoding=3, text=joined_genres)

        if track.release_year:
            id3["TYER"] = TYER(encoding=3, text=str(track.release_year))

        main_artist = (track.artists[0] if track.artists else "").strip()
        if main_artist:
            sort_artist = self._make_sort_name(main_artist)
            id3["TSOP"] = TSOP(encoding=3, text=sort_artist)
        if track.album:
            sort_album = self._make_sort_name(track.album)
            id3["TSOA"] = TSOA(encoding=3, text=sort_album)
        if track.title:
            sort_title = self._make_sort_name(track.title)
            id3["TSOT"] = TSOT(encoding=3, text=sort_title)

        if track.id:
            id3.add(
                TXXX(
                    encoding=3,
                    desc="SPOTIFY_TRACK_ID",
                    text=str(track.id),
                )
            )

        if cover_bytes:
            id3.delall("APIC")
            id3.add(
                APIC(
                    encoding=3,
                    mime=cover_mime,
                    type=3,
                    desc="Cover",
                    data=cover_bytes,
                )
            )
            if cover_thumb:
                id3.add(
                    APIC(
                        encoding=3,
                        mime=cover_mime,
                        type=1,
                        desc="Thumbnail",
                        data=cover_thumb,
                    )
                )

        id3.save(str(media_path), v2_version=3)

    #  M4A/ALAC tagging 

    def _tag_m4a(
        self,
        media_path: Path,
        track: Track,
        index: int,
        cover_bytes: Optional[bytes],
    ) -> None:
        audio = MP4(str(media_path))

        audio["\xa9nam"] = track.title or ""
        audio["\xa9ART"] = track.artist_str or ""
        audio["aART"] = track.artist_str or ""
        if track.album:
            audio["\xa9alb"] = track.album

        total_tracks = track.total_tracks or 0
        audio["trkn"] = [(int(index), int(total_tracks))]
        audio["disk"] = [(1, 1)]

        if track.release_year:
            audio["\xa9day"] = str(track.release_year)

        if track.genres:
            audio["\xa9gen"] = "; ".join(track.genres[:4])

        main_artist = (track.artists[0] if track.artists else "").strip()
        if main_artist:
            audio["soar"] = self._make_sort_name(main_artist)
        if track.title:
            audio["sonm"] = self._make_sort_name(track.title)
        if track.album:
            audio["soal"] = self._make_sort_name(track.album)

        if track.id:
            audio["----:com.apple.iTunes:SPOTIFY_TRACK_ID"] = [
                str(track.id).encode("utf-8")
            ]

        if cover_bytes:
            img_bytes, fmt = self._process_image_for_mp4(cover_bytes)
            audio["covr"] = [MP4Cover(img_bytes, imageformat=fmt)]

        audio.save()

    #  FLAC tagging 

    def _tag_flac(
        self,
        media_path: Path,
        track: Track,
        index: int,
        cover_bytes: Optional[bytes],
        cover_mime: str,
    ) -> None:
        audio = FLAC(str(media_path))

        audio["TITLE"] = [track.title or ""]
        audio["ARTIST"] = track.artists or [track.artist_str or ""]
        if track.album:
            audio["ALBUM"] = [track.album]
        audio["ALBUMARTIST"] = [track.artist_str or ""]

        total_tracks = track.total_tracks or 0
        audio["TRACKNUMBER"] = [str(index)]
        if total_tracks > 0:
            audio["TRACKTOTAL"] = [str(total_tracks)]

        if track.release_year:
            audio["DATE"] = [str(track.release_year)]

        if track.genres:
            audio["GENRE"] = track.genres[:4]

        if track.id:
            audio["SPOTIFY_TRACK_ID"] = [str(track.id)]

        if cover_bytes:
            pic = Picture()
            pic.data = cover_bytes
            pic.type = 3  # front cover
            pic.mime = cover_mime or "image/jpeg"
            pic.desc = "Cover"

            if _PIL_OK:
                try:
                    img = Image.open(BytesIO(cover_bytes))
                    pic.width, pic.height = img.size
                    pic.depth = 24
                except Exception:
                    pass

            audio.clear_pictures()
            audio.add_picture(pic)

        audio.save()

    #  public API 

    def embed_tags(self, media_path: Path, track: Track, index: int) -> None:
        """
        Handles MP3, M4A/ALAC, FLAC, WAV, AIFF.
        """

        # Wait for file to stabilize on disk (Windows AV, slow FS, etc.)
        time.sleep(0.5)
        current_size = 0
        for _ in range(10):
            try:
                new_size = media_path.stat().st_size
                if new_size == current_size and new_size > 0:
                    break
                current_size = new_size
                time.sleep(0.3)
            except FileNotFoundError:
                logging.debug("File not found during stabilization check: %s", media_path)
                return
            except Exception:
                pass

        cover_raw = self._fetch_cover_bytes(track.cover_url)

        try:
            suffix = media_path.suffix.lower()
            full_cover = None
            thumb_cover = None
            cover_mime = "image/jpeg"
            if cover_raw:
                full_cover, thumb_cover, cover_mime = self._process_image_for_mp3(cover_raw)

            if suffix in (".m4a", ".mp4"):
                self._tag_m4a(media_path, track, index, full_cover)
            elif suffix == ".flac":
                self._tag_flac(media_path, track, index, full_cover, cover_mime)
            elif suffix in (".wav", ".wave", ".aiff", ".aif"):
                self._tag_mp3_like(media_path, track, index, full_cover, thumb_cover, cover_mime)
            else:
                # default: MP3
                self._tag_mp3_like(media_path, track, index, full_cover, thumb_cover, cover_mime)
        except Exception:
            logging.exception("CRITICAL: Tagging failed for %s", media_path)


async def fetch_cover_bytes_async(url: str, timeout: int = 10) -> Optional[bytes]:
    if not (_ASYNC_OK and aiohttp):
        return None
    try:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.read()
    except Exception:
        return None
    return None


class PlaylistDownloader:
    def __init__(
        self,
        out_dir: Path,
        bitrate: int,
        workers: int,
        skip_existing: bool,
        verbose: bool,
        ffmpeg_path: Optional[str],
        log_file: Optional[Path],
        use_async: bool = False,
        smart_search: bool = False,
        youtube_api_key: Optional[str] = None,
        dry_run: bool = False,
        codec: str = "mp3",
        max_video_seconds: Optional[int] = 900,
        duration_tolerance_seconds: int = 45,
    ):
        self.out_dir = out_dir
        self.bitrate = bitrate
        self.workers = max(1, workers)
        self.skip_existing = skip_existing
        self.verbose = verbose
        self.ffmpeg_path = ffmpeg_path
        self.log_file = log_file
        self.use_async = use_async and _ASYNC_OK
        self.http = requests.Session()
        self.smart_search_enabled = smart_search
        self.smart = HybridSmartSearch(youtube_api_key, verbose=verbose) if smart_search else None
        self.codec = codec if codec in ("mp3", "m4a", "flac", "alac", "wav", "aiff") else "mp3"
        self.ytdlp = YTDLPWrapper(
            ffmpeg_path=self.ffmpeg_path,
            verbose=self.verbose,
            smart=self.smart,
            codec=self.codec,
            max_video_seconds=max_video_seconds,
            duration_tolerance_seconds=duration_tolerance_seconds,
        )
        self.tagger = Tagger(self.http)
        self.dry_run = dry_run
        self.max_video_seconds = max_video_seconds
        self.duration_tolerance_seconds = duration_tolerance_seconds

    def _track_filename(self, index: int, track: Track) -> str:
        base = safe_filename(track.title)
        ext_map = {
            "mp3": ".mp3",
            "m4a": ".m4a",
            "alac": ".m4a",
            "flac": ".flac",
            "wav": ".wav",
            "aiff": ".aiff",
        }
        ext = ext_map.get(self.codec, ".mp3")
        return base + ext

    def _queries_for(self, track: Track) -> List[str]:
        return [
            f"{track.artist_str} - {track.title} official audio",
            f"{track.title} {track.artist_str} audio",
            f"{track.title} {track.artist_str} lyrics",
        ]

    def _process_one(
        self,
        payload: Tuple[int, Track, Path],
    ) -> Tuple[int, Track, Dict[str, Any]]:
        index, track, target_dir = payload

        # === DIRECT SOUNDCLOUD HANDLING ===
        if track.source == "soundcloud" and track.direct_url:
            try:
                result = self.ytdlp._download_url(
                    track.direct_url,
                    str(target_dir / (track.title + ".%(ext)s")),
                    self.bitrate,
                    expected_duration_s=(track.duration_ms or 0) // 1000,
                )
                if result:
                    media_path = target_dir / self._track_filename(index, track)
                    if result.resolve() != media_path.resolve():
                        result.rename(media_path)
                    self.tagger.embed_tags(media_path, track, index)
                    return index, track, {"ok": True, "path": media_path}
            except Exception as e:
                logging.exception("Direct SC download failed: %s", e)

        filename = self._track_filename(index, track)
        media_path = target_dir / filename

        # === SKIP EXISTING ===
        if self.skip_existing and media_path.exists():
            return index, track, {"ok": True, "path": media_path, "skipped": True}

        temp_template = str(target_dir / (Path(filename).stem + ".%(ext)s"))
        result: Optional[Path] = None

        # === PREVIEW FINGERPRINT (GENERATE ONCE PER TRACK) ===
        if track.preview_url and track.preview_fp is None:
            try:
                import requests
                from tempfile import NamedTemporaryFile
                from fingerprint import chromaprint_fingerprint

                r = requests.get(track.preview_url, timeout=10)
                if r.status_code == 200:
                    with NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                        f.write(r.content)
                        track.preview_fp_path = Path(f.name)
                    track.preview_fp = chromaprint_fingerprint(track.preview_fp_path)
            except Exception as e:
                logging.warning("Preview fingerprint failed: %s", e)
                track.preview_fp = None

        # ====================================================================================
        # SMART SEARCH (ML RANKED) + MUSIC VIDEO SKIP + SPECTROGRAM + CHROMAPRINT + MFCC
        # ====================================================================================
        if self.smart_search_enabled and self.smart:
            for q in self._queries_for(track):
                candidates = self.smart.get_ranked_candidates(q, track)
                if not candidates:
                    continue

                for cand in candidates:
                    title = cand.get("title", "")
                    channel = cand.get("channel", "")

                    # === MUSIC VIDEO DETECTION ===
                    if looks_like_music_video(title, channel):
                        logging.info("Skipping music video: %s | %s", title, channel)
                        continue

                    url = f"https://www.youtube.com/watch?v={cand['video_id']}"
                    try:
                        expected = (track.duration_ms or 0) // 1000
                        result = self.ytdlp._download_url(
                            url,
                            temp_template,
                            bitrate_kbps=self.bitrate,
                            expected_duration_s=expected,
                        )
                        if not (result and result.exists()):
                            result = None
                            continue

                        # === SPECTROGRAM QUALITY CHECK ===
                        try:
                            from spectrogram_quality import analyze_spectrogram, looks_like_bad_source
                            metrics = analyze_spectrogram(str(result))
                            if looks_like_bad_source(metrics):
                                logging.warning(
                                    "Rejected via Spectrogram HF=%.4f ENT=%.4f",
                                    metrics.get("hf_energy_ratio", 0),
                                    metrics.get("spectral_entropy", 0),
                                )
                                result.unlink(missing_ok=True)
                                result = None
                                continue
                        except Exception as e:
                            logging.warning("Spectrogram error: %s", e)

                        # === CHROMAPRINT FINGERPRINT MATCH ===
                        if track.preview_fp is not None:
                            try:
                                from fingerprint import chromaprint_fingerprint, fingerprint_similarity
                                cand_fp = chromaprint_fingerprint(result)

                                if cand_fp is not None:
                                    fp_score = fingerprint_similarity(cand_fp, track.preview_fp)
                                    logging.info("Chromaprint similarity: %.3f", fp_score)

                                    if fp_score < 0.60:
                                        logging.warning("Rejected by Chromaprint (%.3f): %s",
                                                        fp_score, title)
                                        result.unlink(missing_ok=True)
                                        result = None
                                        continue
                            except Exception as e:
                                logging.warning("Chromaprint failed: %s", e)

                        # === MFCC MATCH CHECK ===
                        if track.preview_url:
                            try:
                                from audio_verify import extract_mfcc, fetch_spotify_preview, similarity

                                cand_vec = extract_mfcc(result)
                                ref_vec = fetch_spotify_preview(track.preview_url)
                                score = similarity(cand_vec, ref_vec)

                                logging.info("MFCC similarity: %.3f", score)

                                if score < 0.55:
                                    logging.warning("Rejected by MFCC (%.3f): %s", score, title)
                                    result.unlink(missing_ok=True)
                                    result = None
                                    continue

                            except Exception as e:
                                logging.warning("MFCC verification failed: %s", e)

                        # === SUCCESSFUL CANDIDATE ===
                        break

                    except Exception as e:
                        logging.warning("Candidate failed: %s", e)
                        result = None
                        continue

                if result and result.exists():
                    break

        # ====================================================================================
        # FALLBACK (yt-dlp normal search)
        # ====================================================================================
        if not result:
            for q in self._queries_for(track):
                try:
                    result = self.ytdlp.download_to_audio(
                        q, temp_template, self.bitrate, track=track
                    )
                except Exception:
                    result = None

                if result and result.exists():
                    break

        # ====================================================================================
        # FINALIZE OR FAIL
        # ====================================================================================
        if result and result.exists():
            try:
                if result.resolve() != media_path.resolve():
                    result.rename(media_path)

                self.tagger.embed_tags(media_path, track, index)

                return index, track, {"ok": True, "path": media_path}
            except Exception as e:
                logging.exception("Finalize error: %s", e)
                return index, track, {"ok": False, "reason": f"finalize error: {e}"}

        return index, track, {"ok": False, "reason": "not found"}

    async def _process_one_async(
        self,
        payload: Tuple[int, Track, Path],
    ) -> Tuple[int, Track, Dict[str, Any]]:
        index, track, target_dir = payload
        loop = asyncio.get_event_loop()

        # === DIRECT SOUNDCLOUD HANDLING (same as sync, but offloaded) ===
        if track.source == "soundcloud" and track.direct_url:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.ytdlp._download_url(
                        track.direct_url,
                        str(target_dir / (track.title + ".%(ext)s")),
                        self.bitrate,
                        expected_duration_s=(track.duration_ms or 0) // 1000,
                    ),
                )
                if result:
                    media_path = target_dir / self._track_filename(index, track)
                    if result.resolve() != media_path.resolve():
                        result.rename(media_path)
                    await loop.run_in_executor(
                        None, self.tagger.embed_tags, media_path, track, index
                    )
                    return index, track, {"ok": True, "path": media_path}
            except Exception as e:
                logging.exception("Direct SC download failed (async): %s", e)

        filename = self._track_filename(index, track)
        media_path = target_dir / filename

        # === SKIP EXISTING ===
        if self.skip_existing and media_path.exists():
            return index, track, {"ok": True, "path": media_path, "skipped": True}

        temp_template = str(target_dir / (Path(filename).stem + ".%(ext)s"))
        result: Optional[Path] = None

        # === PREVIEW FINGERPRINT (GENERATE ONCE PER TRACK) ===
        if track.preview_url and getattr(track, "preview_fp", None) is None:
            try:
                import requests
                from tempfile import NamedTemporaryFile
                from fingerprint import chromaprint_fingerprint

                def _gen_preview_fp() -> None:
                    r = requests.get(track.preview_url, timeout=10)
                    if r.status_code == 200:
                        with NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                            f.write(r.content)
                            track.preview_fp_path = Path(f.name)
                        track.preview_fp = chromaprint_fingerprint(track.preview_fp_path)
                    else:
                        track.preview_fp = None

                await loop.run_in_executor(None, _gen_preview_fp)
            except Exception as e:
                logging.warning("Async preview fingerprint failed: %s", e)
                track.preview_fp = None

        # ====================================================================================
        # SMART SEARCH (ML RANKED) + MUSIC VIDEO SKIP + SPECTROGRAM + CHROMAPRINT + MFCC
        # ====================================================================================
        if self.smart_search_enabled and self.smart:
            for q in self._queries_for(track):
                try:
                    candidates = self.smart.get_ranked_candidates(q, track)
                except Exception:
                    candidates = []

                if not candidates:
                    continue

                for cand in candidates:
                    title = cand.get("title", "")
                    channel = cand.get("channel", "")

                    # === MUSIC VIDEO DETECTION ===
                    if looks_like_music_video(title, channel):
                        logging.info(
                            "[async] Skipping music video: %s | %s", title, channel
                        )
                        continue

                    url = f"https://www.youtube.com/watch?v={cand['video_id']}"

                    try:
                        expected = (track.duration_ms or 0) // 1000

                        # Download candidate via executor
                        result = await loop.run_in_executor(
                            None,
                            lambda: self.ytdlp._download_url(
                                url,
                                temp_template,
                                bitrate_kbps=self.bitrate,
                                expected_duration_s=expected,
                            ),
                        )
                        if not (result and result.exists()):
                            result = None
                            continue

                        # === SPECTROGRAM QUALITY CHECK ===
                        try:
                            from spectrogram_quality import (
                                analyze_spectrogram,
                                looks_like_bad_source,
                            )

                            def _analyze() -> Dict[str, Any]:
                                return analyze_spectrogram(str(result))

                            metrics = await loop.run_in_executor(None, _analyze)
                            if looks_like_bad_source(metrics):
                                logging.warning(
                                    "[async] Rejected via Spectrogram HF=%.4f ENT=%.4f",
                                    metrics.get("hf_energy_ratio", 0.0),
                                    metrics.get("spectral_entropy", 0.0),
                                )
                                result.unlink(missing_ok=True)
                                result = None
                                continue
                        except Exception as e:
                            logging.warning("Async spectrogram error: %s", e)

                        # === CHROMAPRINT FINGERPRINT MATCH ===
                        if getattr(track, "preview_fp", None) is not None:
                            try:
                                from fingerprint import (
                                    chromaprint_fingerprint,
                                    fingerprint_similarity,
                                )

                                def _cand_fp():
                                    return chromaprint_fingerprint(result)

                                cand_fp = await loop.run_in_executor(None, _cand_fp)

                                if cand_fp is not None:
                                    fp_score = fingerprint_similarity(
                                        cand_fp, track.preview_fp
                                    )
                                    logging.info(
                                        "[async] Chromaprint similarity: %.3f", fp_score
                                    )
                                    if fp_score < 0.60:
                                        logging.warning(
                                            "[async] Rejected by Chromaprint (%.3f): %s",
                                            fp_score,
                                            title,
                                        )
                                        result.unlink(missing_ok=True)
                                        result = None
                                        continue
                            except Exception as e:
                                logging.warning(
                                    "[async] Chromaprint failed: %s", e
                                )

                        # === MFCC MATCH CHECK ===
                        if track.preview_url:
                            try:
                                from audio_verify import (
                                    extract_mfcc,
                                    fetch_spotify_preview,
                                    similarity,
                                )

                                def _mfcc_pair():
                                    cand_vec = extract_mfcc(result)
                                    ref_vec = fetch_spotify_preview(track.preview_url)
                                    return cand_vec, ref_vec

                                cand_vec, ref_vec = await loop.run_in_executor(
                                    None, _mfcc_pair
                                )
                                score = similarity(cand_vec, ref_vec)
                                logging.info(
                                    "[async] MFCC similarity: %.3f", score
                                )

                                if score < 0.55:
                                    logging.warning(
                                        "[async] Rejected by MFCC (%.3f): %s",
                                        score,
                                        title,
                                    )
                                    result.unlink(missing_ok=True)
                                    result = None
                                    continue
                            except Exception as e:
                                logging.warning(
                                    "[async] MFCC verification failed: %s", e
                                )

                        # === SUCCESSFUL CANDIDATE ===
                        break

                    except Exception as e:
                        logging.warning("[async] Candidate failed: %s", e)
                        result = None
                        continue

                if result and result.exists():
                    break

        # ====================================================================================
        # FALLBACK (yt-dlp normal search)
        # ====================================================================================
        if not result:
            for q in self._queries_for(track):
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.ytdlp.download_to_audio(
                            q, temp_template, self.bitrate, track=track
                        ),
                    )
                except Exception as e:
                    logging.debug("[async] yt-dlp error for '%s': %s", q, e)
                    result = None

                if result and result.exists():
                    # Optional: you could also run MFCC/Chromaprint here
                    break

        # ====================================================================================
        # FINALIZE OR FAIL
        # ====================================================================================
        if result and result.exists():
            try:
                if result.resolve() != media_path.resolve():
                    result.rename(media_path)

                await loop.run_in_executor(
                    None, self.tagger.embed_tags, media_path, track, index
                )

                return index, track, {"ok": True, "path": media_path}
            except Exception as e:
                logging.exception("[async] Finalize error: %s", e)
                return index, track, {"ok": False, "reason": f"finalize error: {e}"}

        return index, track, {"ok": False, "reason": "not found"}


    def write_reports(self, target_dir: Path, results: List[Tuple[int, Track, Dict[str, Any]]]) -> None:
        m3u_path = target_dir / "playlist.m3u"
        with m3u_path.open("w", encoding="utf-8") as m3u:
            for idx, _track, res in sorted(results, key=lambda r: r[0]):
                p = res.get("path")
                if p:
                    m3u.write(Path(p).name + "\n")
        logging.info("Wrote M3U: %s", m3u_path)

        report = {
            "success": [
                {
                    "index": idx,
                    "title": t.title,
                    "artists": t.artists,
                    "album": t.album,
                    "file": str(res.get("path")) if res.get("path") else None,
                    "year": t.release_year,
                    "genres": t.genres,
                }
                for idx, t, res in results
                if res.get("ok")
            ],
            "failed": [
                {
                    "index": idx,
                    "title": t.title,
                    "artists": t.artists,
                    "album": t.album,
                    "reason": res.get("reason"),
                }
                for idx, t, res in results
                if not res.get("ok")
            ],
        }
        report_path = target_dir / "download_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logging.info("Wrote report: %s", report_path)

    def download(self, playlist_ref: str) -> None:
        # unified Spotify / SoundCloud fetch (SoundCloud via yt-dlp only)
        tracks, playlist_name, source = get_playlist_tracks(playlist_ref, verbose=self.verbose)

        if not tracks:
            logging.error("No tracks found. Is the playlist public?")
            raise SystemExit(1)

        if self.dry_run:
            logging.info("Dry run enabled. Listing tracks only:")
            for i, t in enumerate(tracks, start=1):
                print(f"{i:02d}. {t.artist_str} - {t.title} [{t.album or 'Single'}]")
            logging.info("Dry run complete. No downloads performed.")
            return

        target_dir = self.out_dir / safe_filename(playlist_name or playlist_ref)
        target_dir.mkdir(parents=True, exist_ok=True)

        tasks = [(i, t, target_dir) for i, t in enumerate(tracks, start=1)]
        results: List[Tuple[int, Track, Dict[str, Any]]] = []

        if self.use_async and _ASYNC_OK:
            logging.info("Starting async downloads with up to %d concurrent tasks", self.workers)

            async def runner():
                sem = asyncio.Semaphore(self.workers)

                async def guarded(task):
                    async with sem:
                        return await self._process_one_async(task)

                if _RICH:
                    console = Console()
                    with Progress(
                        TextColumn("[bold]Tracks[/]"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TimeRemainingColumn(),
                        transient=True,
                        console=console,
                    ) as progress:
                        task_id = progress.add_task("Downloading", total=len(tasks))
                        coros = [guarded(t) for t in tasks]
                        for coro in asyncio.as_completed(coros):
                            res = await coro
                            results.append(res)
                            progress.advance(task_id)
                else:
                    coros = [guarded(t) for t in tasks]
                    for coro in asyncio.as_completed(coros):
                        res = await coro
                        results.append(res)

            try:
                asyncio.run(runner())
            except KeyboardInterrupt:
                logging.warning("Interrupted by user. Exiting.")
                raise SystemExit(130)
        else:
            logging.info("Starting downloads with %d workers", self.workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as ex:
                futures = [ex.submit(self._process_one, payload) for payload in tasks]
                if _RICH:
                    console = Console()
                    with Progress(
                        TextColumn("[bold]Tracks[/]"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TimeRemainingColumn(),
                        transient=True,
                        console=console,
                    ) as progress:
                        task_id = progress.add_task("Downloading", total=len(futures))
                        for fut in concurrent.futures.as_completed(futures):
                            try:
                                results.append(fut.result())
                            except KeyboardInterrupt:
                                raise
                            except Exception as e:
                                logging.exception("Worker crashed: %s", e)
                            progress.advance(task_id)
                else:
                    from tqdm import tqdm  

                    for fut in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="Tracks",
                        unit="trk",
                    ):
                        try:
                            results.append(fut.result())
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            logging.exception("Worker crashed: %s", e)

        # Per-track summary (separate messages per file)
        for idx, t, res in sorted(results, key=lambda r: r[0]):
            if res.get("ok"):
                logging.info(
                    "OK   [%02d] %s - %s -> %s",
                    idx,
                    t.artist_str,
                    t.title,
                    res.get("path"),
                )
            else:
                logging.warning(
                    "FAIL [%02d] %s - %s (%s)",
                    idx,
                    t.artist_str,
                    t.title,
                    res.get("reason") or "unknown error",
                )

        # Write playlist + JSON summary
        self.write_reports(target_dir, results)
        ok = sum(1 for _i, _t, r in results if r.get("ok"))
        fail = len(results) - ok
        logging.info(
            "Done. %d succeeded, %d failed. Source=%s. Files saved in: %s",
            ok,
            fail,
            source,
            target_dir,
        )
