from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path("config.json")


DEFAULT_CONFIG: Dict[str, Any] = {
    "spotify": {
        "client_id": "",
        "client_secret": "",
        "redirect_uri": "http://localhost:8888/callback",
        "market": "US",
    },
    "youtube": {
        "api_key": "",
        "prefer_music_search": True,
        "safe_search": False,
    },
    "downloader": {
        "max_workers": 4,
        "bitrate": 320,
        "retry_count": 3,
        "retry_delay": 2,
        "timeout": 30,
        "prefer_source": "any",
        "allow_youtube_fallback": True,
        "lossless_preferred": True,
        "codec": "flac",
        "smartsearch_enabled": True,
        "ml_assist_enabled": True,
        "cover_embed_quality": "max",
        "tagging_threads": 4,
        "skip_duplicates": True,
        "dedupe_strategy": "filename",
        "filename_template": "{artist_str} - {title}",
        "max_video_seconds": 900,
        "duration_tolerance_seconds": 45,
    },
    "network": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "proxies": None,
    },
    "paths": {
        "download_root": "downloads",
        "temp_dir": "temp",
        "log_dir": "logs",
    },
    "ffmpeg_path": None,
}


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if (
            k in base
            and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


@lru_cache(maxsize=1)
def get_app_config() -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text("utf-8"))
            if isinstance(data, dict):
                _deep_update(cfg, data)
        except Exception as e:
            logging.error("Failed to read config.json: %s", e)
    else:
        logging.info("config.json not found, using defaults.")
    return cfg
