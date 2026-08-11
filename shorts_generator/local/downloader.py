"""Local YouTube download via yt-dlp.

Returns a local mp4 path so the rest of the local pipeline can read it
directly off disk.
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from typing import Optional

from ..config import LOCAL_OUTPUT_DIR


def _import_ytdlp():
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    return yt_dlp


def _format_for(fmt: str) -> str:
    """Map our '720' / '1080' shorthand to a yt-dlp format selector."""
    try:
        height = int(fmt)
    except ValueError:
        height = 720
    return (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"best[height<={height}][ext=mp4]/best"
    )


def _extract_youtube_video_id(source: str) -> Optional[str]:
    """Best-effort extraction of a YouTube video id from a URL."""
    parsed = urlparse(source)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/").split("/", 1)[0]
        return video_id or None

    if "youtube.com" in host:
        if parsed.path.startswith("/watch"):
            qs = parse_qs(parsed.query)
            video_id = qs.get("v", [""])[0]
            return video_id or None
        match = re.search(r"/(?:shorts|embed|live)/([^/?#&]+)", parsed.path)
        if match:
            return match.group(1)

    return None


def _resolve_local_path(source: str) -> Optional[str]:
    """Return a local filesystem path if the input already points at one."""
    parsed = urlparse(source)
    if parsed.scheme == "file":
        raw_path = unquote(parsed.path)
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            raw_path = f"//{parsed.netloc}{raw_path}"
        candidate = Path(raw_path).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
        raise RuntimeError(f"Local file URL does not exist: {source}")

    if parsed.scheme in ("http", "https"):
        return None

    candidate = Path(source).expanduser()
    if candidate.exists() and candidate.is_file():
        return str(candidate.resolve())

    if any(sep in source for sep in (os.sep, "/")) or source.startswith("~") or source.startswith("."):
        raise RuntimeError(f"Local file path does not exist: {source}")

    return None


def _existing_download(out_dir: str, video_id: str) -> Optional[str]:
    """Return a cached download path if we already have this YouTube id."""
    for ext in (".mp4", ".mkv", ".webm"):
        candidate = os.path.join(out_dir, f"source_{video_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def _try_extract_cookies_browser(youtube_url: str, out_dir: str) -> Optional[str]:
    """Try to extract YouTube cookies from browser and write to temp file."""
    cookie_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w')
    cookie_file.close()
    try:
        import browser_cookie3
        cj = browser_cookie3.chrome(domain_name='.youtube.com')
        if cj:
            for cookie in cj:
                cookie_file.write(
                    f"{cookie.domain}\t{'TRUE' if cookie.secure else 'FALSE'}\t{cookie.path}\t"
                    f"{'TRUE' if cookie.secure else 'FALSE'}\t{int(cookie.expires or 0)}\t"
                    f"{cookie.name}\t{cookie.value}\n"
                )
            cookie_file.close()
            if os.path.getsize(cookie_file.name) > 0:
                print(f"[download/local] browser cookies found: {len(cj)} cookies", flush=True)
                return cookie_file.name
    except Exception as e:
        print(f"[download/local] cookie extraction failed: {e}", flush=True)
    finally:
        try:
            os.unlink(cookie_file.name)
        except OSError:
            pass
    return None


def download_youtube_local(video_url: str, fmt: str = "720", out_dir: Optional[str] = None) -> str:
    """Download a remote URL or return a local file path unchanged."""
    local_path = _resolve_local_path(video_url)
    if local_path:
        print(f"[download/local] using local file: {local_path}", flush=True)
        return local_path

    yt_dlp = _import_ytdlp()
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    video_id = _extract_youtube_video_id(video_url)
    if video_id:
        cached = _existing_download(out_dir, video_id)
        if cached:
            print(f"[download/local] reusing cached download: {cached}", flush=True)
            return cached

    print(f"[download/local] {video_url} @ {fmt}p → {out_dir}/", flush=True)

    base_opts = {
        "format": _format_for(fmt),
        "outtmpl": os.path.join(out_dir, "source_%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "extra_user_agent": "",
        "socket_timeout": 30,
    }

    # First attempt: plain download with spoofed user-agent
    cookie_file = None
    errors = []
    for attempt, opts_extra in enumerate([
        {},
        {"extract_flat": False},
        {"force_generic_extractor": True},
    ]):
        opts = {**base_opts, **opts_extra}
        if cookie_file:
            opts["cookies"] = cookie_file
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                path = ydl.prepare_filename(info)
                if not os.path.exists(path):
                    stem, _ = os.path.splitext(path)
                    for ext in (".mp4", ".mkv", ".webm"):
                        if os.path.exists(stem + ext):
                            path = stem + ext
                            break
                if os.path.exists(path) and os.path.getsize(path) > 1000:
                    print(f"[download/local] ready: {path}", flush=True)
                    if cookie_file:
                        try:
                            os.unlink(cookie_file)
                        except OSError:
                            pass
                    return path
        except Exception as e:
            err_str = str(e)
            errors.append(f"attempt-{attempt}: {err_str[:100]}")
            if "Sign in to confirm you're not a bot" in err_str and not cookie_file:
                cookie_file = _try_extract_cookies_browser(video_url, out_dir)
                if cookie_file:
                    print(f"[download/local] retrying with browser cookies", flush=True)
                    continue
            continue

    # All attempts failed
    raise RuntimeError(
        f"yt-dlp download failed for {video_url}. Errors: {'; '.join(errors)}. "
        "YouTube may be blocking automated downloads from this IP. "
        "Try: (1) set FORCE_LOCAL_MODE=false and use API mode, or "
        "(2) upload cookies.txt to /content/cookies.txt and re-run."
    )
