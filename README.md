
# songdownloader

> **A high-fidelity Spotify & SoundCloud downloader with audio Integrity Verification.**

songdownloader is a sophisticated audio retrieval pipeline designed to ensure the music you download actually matches the metadata you requested. 

It combines multiple techniques to automatically reject low-quality upscales, incorrect tracks, and music videos with dialogue.

## Design Goals

* Prioritize metadata correctness over download speed
* Avoid opaque heuristics; keep search and selection explainable
* Allow full user control via a single configuration file
* Support verification and post-download inspection
* Remain fully local and offline after metadata ingestion


## Key Features

### Intelligent Search & Retrieval
* Hybrid Search Engine: Utilizes the YouTube Data API v3 for precision and falls back to `yt-dlp` scraping when necessary.
* ML Candidate Scoring: Ranks potential audio matches based on title similarity, channel trustworthiness, and duration matching using a custom heuristic algorithm (`ml_search.py`).
* Smart Video Filtering: Automatically detects and skips "Music Videos" (which often contain dialogue, SFX, and intros) in favor of "Official Audio" or "Topic" tracks.

### The Audio Verification Pipeline
Unlike other downloaders that blindly grab the first result, this project verifies the audio data before finalizing the file:
1.  Spectrogram Quality Check: Uses `librosa` to analyze the frequency spectrum. It detects "fake" 320kbps files (low-bitrate audio upscaled) by checking the high-frequency energy ratio and spectral entropy.
2.  Chromaprint Fingerprinting: Generates an audio fingerprint (using `fpcalc`) of the downloaded file and compares it against the official Spotify 30-second preview to ensure the song is correct.
3.  MFCC Similarity: Calculates Mel-frequency cepstral coefficients to mathematically verify that the downloaded audio matches the sonic characteristics of the original source.

### Format & Metadata
* Multi-Codec Support: FLAC, ALAC, MP3, M4A, WAV, and AIFF.
* Rich Tagging: Embeds high-resolution cover art, lyrics references, album name, release year, track number, and Spotify IDs using `mutagen`.
* Playlist Generation: Automatically generates `.m3u` playlists for every batch.

---

## Prerequisites

To run this project, you need the following installed on your system:

1.  **Python 3.8+**
2.  **FFmpeg**: Required for audio conversion.
    * **Windows**: `winget install ffmpeg`
    * **Mac**: `brew install ffmpeg`
    * **Linux**: `sudo apt install ffmpeg`
3.  **fpcalc**: Required for audio fingerprinting (Chromaprint).
    * Download from [AcoustID](https://acoustid.org/chromaprint).
    * Ensure `fpcalc` is in your system PATH or specified in `config.json`.

---

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/samarpreetxd/songdownloader.git
    cd songdownloader
    ```

2.  **Install Python dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Dependencies include: `yt-dlp`, `spotipy`, `librosa`, `mutagen`, `numpy`, `requests`, `tqdm`)*

---

## Configuration

The project uses a `config.json` file to manage API keys and behavior.

1.  Create a `config.json` file in the root directory (or rename a template).
2.  Populate it with your keys:

```json
{
  "download": {
    "source_priority": ["youtube_api", "yt-dlp"],
    "fallback_search": "yt-dlp",
    "max_workers": 4,
    "output_directory": "downloads",
    "codec": "flac",
    "bitrate": 320,
    "save_metadata": true,
    "embed_cover": true,
    "write_m3u": true,
    "separate_logs_per_file": true
  },

  "spotify": {
    "client_id": "YOUR KEY HERE",
    "client_secret": "YOUR KEY HERE",
    "market": null
  },

  "search": {
    "youtube_api_enabled": true,
    "youtube_api_key": "YOUR KEY HERE",
    "soundcloud_enabled": false,
    "audiocistid_enabled": false,

    "results_per_query": 10,

    "ytapi_query_templates": [
      "{artist} - {title} official audio",
      "{artist} - {title} audio",
      "{artist} {title} audio",
      "{artist} {title}"
    ],

    "yt_dlp_query_templates": [
      "{artist} - {title} audio",
      "{artist} {title} audio",
      "{artist} - {title} lyrics"
    ]
  },

  "yt_dlp": {
    "binary": "yt-dlp",
    "format": "bestaudio/best",
    "ignore_errors": true,
    "postprocessors": [
      {
        "key": "FFmpegExtractAudio",
        "preferredcodec": "flac",
        "preferredquality": "0"
      }
    ],
    "extra_args": [
      "--no-warnings",
      "--no-call-home",
      "--restrict-filenames"
    ]
  },

  "logging": {
    "verbose": false,
    "debug": false,
    "color": true,
    "log_each_file_separately": true
  },
  "fingerprinting": {
    "fpcalc_path": "songdownloader/fpcalc.exe"
  }

}


```

### Obtaining Keys

* **Spotify**: Create an app at the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
* **YouTube**: Enable "YouTube Data API v3" at the [Google Cloud Console](https://console.cloud.google.com/).

---

## Usage

### Basic Download

Download a Spotify playlist or SoundCloud set:

```bash
python main.py https://open.spotify.com/playlist/XXXXXXXX
python main.py https://soundcloud.com/user-XXXXXXXX/sets/XXXXXXXX
```

### CLI Arguments

You can override `config.json` settings directly from the command line:

| Flag | Description | Example |
| --- | --- | --- |
| `-c`, `--codec` | Output format (mp3, flac, m4a, wav) | `-c flac` |
| `-b`, `--bitrate` | Audio bitrate (kbps) | `-b 320` |
| `-w`, `--workers` | Number of parallel downloads | `-w 8` |
| `-o`, `--output` | Custom output directory | `-o "C:/Music"` |
| `--smart` | Force enable Smart Search (API + ML) | `--smart` |
| `--skip-existing` | Skip tracks that already exist | `--skip-existing` |
| `--dry-run` | Fetch metadata but don't download | `--dry-run` |

### Examples

**Download in FLAC (Lossless) with 8 threads:**

```bash
python main.py "SPOTIFY_URL" -c flac -w 8

```

**Download from SoundCloud to a specific folder:**

```bash
python main.py "SOUNDCLOUD_URL" -o "./MySets"

```

---

## Project Structure

* **`main.py`**: CLI entry point. Handles argument parsing and initialization.
* **`downloader.py`**: The core engine. Manages the download loop, threading, and calls verification modules.
* **`source.py`**: Handles fetching metadata from Spotify API and SoundCloud (via yt-dlp).
* **`ml_search.py`**: Contains the heuristic algorithm for scoring search results.
* **`spectrogram_quality.py`**: Uses `librosa` to analyze audio files for quality defects (transcoding artifacts).
* **`fingerprint.py`**: Wrapper for `fpcalc` to handle Chromaprint generation and comparison.
* **`audio_verify.py`**: Handles MFCC (Mel-frequency cepstral coefficients) comparison.
* **`config_utils.py`**: Helper to load and merge configurations.

---

## Disclaimer

This tool is created for **educational purposes only**.

* Respect the copyright of artists and creators.
* Do not distribute copyrighted content.
* Adhere to the Terms of Service of Spotify, SoundCloud, and YouTube.

---

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/samarpreetxd/songdownloader/refs/heads/main/LICENSE) - see the [LICENSE](https://raw.githubusercontent.com/samarpreetxd/songdownloader/refs/heads/main/LICENSE) file for details.
