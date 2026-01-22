from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from downloader import PlaylistDownloader


DEFAULT_CONFIG_PATH = Path("config.json")


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Spotify / SoundCloud playlist downloader using yt-dlp + tagging"
    )

    p.add_argument(
        "playlist",
        help="Spotify playlist URL/ID or SoundCloud set/playlist URL",
    )

    # Codec / quality
    p.add_argument(
        "-c",
        "--codec",
        choices=["mp3", "m4a", "flac", "alac", "wav", "aiff"],
        help="Output codec (default from config.json, fallback mp3)",
    )
    p.add_argument(
        "-b",
        "--bitrate",
        type=int,
        help="Bitrate in kbps (for lossy formats, default from config.json, e.g. 320)",
    )

    # Parallelism + paths
    p.add_argument(
        "-w",
        "--workers",
        type=int,
        help="Number of parallel download workers (default from config.json)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory root (default from config.json, fallback ./downloads)",
    )

    # Skip / overwrite
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tracks whose output file already exists",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if the target file already exists",
    )

    # Smart search toggle
    p.add_argument(
        "--smart",
        dest="smart_search",
        action="store_true",
        help="Force enable YouTube API smart search (overrides config)",
    )
    p.add_argument(
        "--no-smart",
        dest="smart_search",
        action="store_false",
        help="Force disable YouTube API smart search (overrides config)",
    )
    p.set_defaults(smart_search=None)

    # Logging / dry-run
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging (INFO)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Debug logging (very noisy)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List tracks but do not download anything",
    )

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(DEFAULT_CONFIG_PATH)

    # logging 
    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Optional log file from config
    log_file = cfg.get("log_file")
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)

    # merge config + CLI 

    def merged_opt(cli_value, *keys, default=None):
        if cli_value is not None:
            return cli_value
        for k in keys:
            if k in cfg and cfg[k] is not None:
                return cfg[k]
        return default

    codec = merged_opt(args.codec, "codec", default="mp3")
    bitrate = merged_opt(args.bitrate, "bitrate_kbps", "bitrate", default=320)
    workers = merged_opt(args.workers, "max_workers", default=4)

    out_root = merged_opt(args.output, "output_dir", "downloads_root", default="downloads")
    out_dir = Path(out_root)

    # skip_existing flag resolution
    if args.skip_existing and args.no_skip_existing:
        logging.warning("Both --skip-existing and --no-skip-existing given; using --skip-existing.")
        skip_existing = True
    elif args.skip_existing:
        skip_existing = True
    elif args.no_skip_existing:
        skip_existing = False
    else:
        skip_existing = bool(cfg.get("skip_existing", True))

    # smart search
    if args.smart_search is not None:
        smart_search = bool(args.smart_search)
    else:
        smart_search = bool(cfg.get("smart_search", True))

    youtube_api_key = cfg.get("youtube_api_key")  # still used by HybridSmartSearch
    ffmpeg_path = cfg.get("ffmpeg_path") or None
    use_async = bool(cfg.get("use_async", False))
    max_video_seconds = cfg.get("max_video_seconds", None)
    duration_tolerance_seconds = int(cfg.get("duration_tolerance_seconds", 45))

    if codec in ("flac", "alac", "wav", "aiff"):
        logging.info(
            "Using lossless container '%s' — note: YouTube/SoundCloud sources are still lossy; "
            "this is lossless transcoding, not true lossless audio.",
            codec,
        )

    logging.info("Using codec '%s' at %d kbps", codec, int(bitrate))

    downloader = PlaylistDownloader(
        out_dir=out_dir,
        bitrate=int(bitrate),
        workers=int(workers),
        skip_existing=skip_existing,
        verbose=bool(args.verbose or args.debug),
        ffmpeg_path=ffmpeg_path,
        log_file=Path(log_file) if log_file else None,
        use_async=use_async,
        smart_search=smart_search,
        youtube_api_key=youtube_api_key,
        dry_run=bool(args.dry_run),
        codec=codec,
        max_video_seconds=max_video_seconds,
        duration_tolerance_seconds=duration_tolerance_seconds,
    )

    try:
        downloader.download(args.playlist)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
