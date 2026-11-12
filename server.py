#!/usr/bin/env python3
"""Нарезка: free local AI renderer. Run together with the Cloudflare control plane."""
import hashlib
import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import imageio_ffmpeg
import yt_dlp
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent
CLIPS = ROOT / "clips"
CLIPS.mkdir(exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SECRET = os.environ.get("PROCESSOR_SECRET", "local-development-secret")
CALLBACK = os.environ.get("APP_CALLBACK_URL", "")
PUBLIC_URL = os.environ.get("PROCESSOR_PUBLIC_URL", "http://127.0.0.1:8788")
MAX_SECONDS = int(os.environ.get("MAX_SOURCE_SECONDS", "43200"))
MODEL = None
MODEL_LOCK = threading.Lock()
JOBS = {}
ALLOWED = re.compile(r"(^|\.)((twitch\.tv)|(tiktok\.com)|(youtube\.com)|(youtu\.be)|(vkvideo\.ru))$")


def safe_url(value):
    try:
        parsed = urlparse(value)
        return parsed.scheme == "https" and bool(ALLOWED.search(parsed.hostname or ""))
    except ValueError:
        return False


def run(args, timeout=1800):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-1200:])
    return result.stdout


def resolve_source(source):
    """Turn a Twitch channel URL into its latest finished archive."""
    parsed = urlparse(source.strip())
    match = re.fullmatch(r"/([A-Za-z0-9_]+)(?:/videos)?/?", parsed.path) if (parsed.hostname or "").lower() in {"twitch.tv", "www.twitch.tv", "m.twitch.tv"} else None
    if not match or match.group(1).lower() in {"videos", "directory", "downloads", "settings"}:
        return source
    channel = match.group(1)
    playlist = f"https://www.twitch.tv/{channel}/videos?filter=archives&sort=time"
    opts = {"extract_flat": True, "playlistend": 10, "quiet": True, "no_warnings": True, "socket_timeout": 30}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(playlist, download=False)
        entries = info.get("entries") or []
        for entry in entries:
            if entry and entry.get("url") and not entry.get("is_live") and (entry.get("duration") or 0) <= MAX_SECONDS:
                return entry["url"]
        if not entries:
            raise ValueError("На канале нет публичных завершённых записей. Пришлите ссылку на конкретное видео.")
    raise ValueError("На канале нет доступных завершённых записей до 12 часов. Пришлите ссылку на конкретное видео.")


def download(source, folder):
    source = resolve_source(source)
    output = str(folder / "source.%(ext)s")
    opts = {
        "outtmpl": output,
        "format": "worst[height<=480][ext=mp4]/worst[height<=480]/worst",
        "noplaylist": True,
        "max_filesize": 2_000_000_000,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 20,
        "concurrent_fragment_downloads": 4,
        "download_ranges": yt_dlp.utils.download_range_func(None, [(0, 300)]),
        "retry_sleep": {"http": lambda n: min(2 ** n, 30), "fragment": lambda n: min(2 ** n, 30)},
        "continuedl": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=True)
        if info.get("duration", 0) > MAX_SECONDS:
            raise ValueError("Видео длиннее 12 часов. Выберите запись покороче.")
        if info.get("is_live"):
            raise ValueError("Прямой эфир ещё идёт. Отправьте ссылку после его завершения.")
        return Path(ydl.prepare_filename(info)), info


def audio_peaks(video):
    raw = run([FFMPEG, "-i", str(video), "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level", "-f", "null", "-"], 900)
    values = []
    for match in re.finditer(r"pts_time:([\d.]+)[\s\S]{0,250}?RMS_level=(-?[\d.]+)", raw):
        values.append((float(match.group(1)), max(0.0, float(match.group(2)) + 45)))
    return values


def highlight(segments, peaks, duration, wishes=""):
    from bisect import bisect_right
    candidates = []
    funny = re.search(r"смеш|юмор|смех", wishes, re.I)
    reaction = re.search(r"реакц|эмоц", wishes, re.I)
    laughter = re.compile(r"(?<![а-я])(?:ха(?:-?ха)+|аха(?:-?аха)+|лол)(?![а-я])|смех|смеюсь|угар|ору", re.I)
    emotion = re.compile(r"что\?|невозможно|смотри|офиг|обалд|невероят|круто|класс|давай|погнали|nice|wow|omg", re.I)
    wish_words = [w for w in re.findall(r"[а-яa-z]{5,}", wishes.lower()) if w not in {"смешная", "смешное", "момент", "клипа", "видео", "реакция", "эмоции"}]
    ends = [s.end for s in segments]
    peak_times = [t for t, _ in peaks]
    baseline = sorted(v for _, v in peaks)[len(peaks) // 2] if peaks else 0
    final = segments[-1].end if segments else 60
    for start in range(0, max(1, int(final)), 2):
        end = start + duration
        first = bisect_right(ends, start)
        selected = [s for s in segments[first:] if s.start <= end]
        text = " ".join(s.text.strip() for s in selected).strip()
        if len(text) < 15:
            continue
        peak_start = bisect_right(peak_times, start)
        peak_end = bisect_right(peak_times, end)
        energy = [max(0, v - baseline) for _, v in peaks[peak_start:peak_end]]
        peak = max(energy, default=0)
        sustained = sum(energy) / len(energy) if energy else 0
        words = re.findall(r"[а-яa-z]+", text.lower())
        laughs = len(laughter.findall(text))
        score = laughs * 10 + min(4, len(emotion.findall(text))) * 3
        score += min(text.count("!"), 4) * 1.5 + min(peak, 25) * 0.7 + min(sustained, 15) * 0.5
        score += min(len(text) / duration, 15) / 3 + min(12, sum(text.lower().count(w) for w in wish_words) * 4)
        if funny:
            score += laughs * 10
        if reaction:
            score += min(peak, 25) * 0.6 + min(text.count("!"), 4) * 2
        if words and len(set(words)) / len(words) < 0.4:
            score -= 6
        if re.search(r"промокод|спонсор|рекламная интеграция", text, re.I):
            score -= 8
        event_ends = [s.end for s in selected if laughter.search(s.text) or emotion.search(s.text) or any(w in s.text.lower() for w in wish_words)]
        if event_ends:
            score -= max(0, max(event_ends) - (start + duration * 0.8)) * 3
        nearby = [s.start for s in selected if 0 < s.start - start <= 1.2]
        aligned = max(0, min(nearby) - 0.35) if nearby else float(start)
        candidates.append((score, aligned, text))
    if not candidates:
        return 0.0, "Лучший момент"
    _, start, text = max(candidates, key=lambda item: item[0])
    phrases = re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"\s+", " ", text))
    clean = max(phrases, key=lambda p: (len(laughter.findall(p)) * 3 + len(emotion.findall(p)), len(p))).strip()
    if len(clean) > 52:
        clean = clean[:53].rsplit(" ", 1)[0]
    return start, (clean or "ЛУЧШИЙ МОМЕНТ").upper()


def escape_text(text):
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")


def render(video, job, start, title, segments, output):
    duration = job["duration"]
    font = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    if not Path(font).exists():
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    color = {"yellow": "0xFFE34D", "white": "white", "neon": "0x66FFC2"}.get(job["color"], "0xFFE34D")
    crop = "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280" if job["layout"] == "crop" else "split[bg][fg];[bg]scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,boxblur=22:8[blur];[fg]scale=720:1280:force_original_aspect_ratio=decrease[main];[blur][main]overlay=(W-w)/2:(H-h)/2"
    vf = f"{crop},drawtext=fontfile='{font}':text='{escape_text(job['title'] or title)}':fontsize=42:fontcolor={color}:borderw=3:bordercolor=black:x=(w-text_w)/2:y=100"
    if job["captions"]:
        subs = []
        for s in segments:
            if s.end >= start and s.start <= start + duration:
                text = escape_text(s.text.strip().upper()[:70])
                a, b = max(0, s.start-start), min(duration, s.end-start)
                subs.append(f"drawtext=fontfile='{font}':text='{text}':fontsize=30:fontcolor=white:borderw=2:bordercolor=black:x=(w-text_w)/2:y=h-180:enable='between(t,{a:.2f},{b:.2f})'")
        if subs:
            vf += "," + ",".join(subs)
    run([FFMPEG, "-y", "-ss", str(start), "-i", str(video), "-t", str(duration), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(output)], 1800)


def upload_clip(job_id, output):
    if not CALLBACK:
        return
    if os.environ.get("CLIP_STORAGE") == "http":
        request = urllib.request.Request(f"{CALLBACK}/api/internal/clips/{job_id}", data=output.read_bytes(), method="PUT", headers={"Content-Type": "video/mp4", "X-Processor-Secret": SECRET, "User-Agent": "Mozilla/5.0"})
        urllib.request.urlopen(request, timeout=120).read()
        return
    run(["npx", "wrangler", "kv", "key", "put", "--namespace-id", "6cdfb10f7b114189b0c0acbd7fb3482a", "--remote", "--path", str(output), "--ttl", "2592000", job_id], 180)


def callback(job_id, payload):
    if not CALLBACK:
        return
    body = json.dumps(payload).encode()
    request = urllib.request.Request(f"{CALLBACK}/api/internal/jobs/{job_id}", data=body, headers={"Content-Type": "application/json", "X-Processor-Secret": SECRET, "User-Agent": "Mozilla/5.0"})
    try:
        urllib.request.urlopen(request, timeout=30).read()
    except Exception as error:
        print("Callback failed:", error)


def friendly_error(error):
    text = str(error)
    if "not currently live" in text:
        return "Канал сейчас не в эфире. Выберите завершённую запись или повторите позже."
    if "нет публичных" in text:
        return "На канале нет публичных завершённых записей. Пришлите прямую ссылку на видео."
    if "No available" in text or "нет доступных" in text:
        return "На канале нет подходящих записей до 12 часов."
    if "private" in text.lower() or "403" in text or "404" in text:
        return "Видео недоступно: оно приватное, удалённое или ограничено в вашем регионе."
    if "Unsupported URL" in text:
        return "Эта ссылка не поддерживается. Отправьте канал Twitch или прямую ссылку на видео."
    if "UNEXPECTED_EOF" in text or "Connection reset" in text or "Connection aborted" in text:
        return "Источник оборвал соединение во время загрузки. Попробуйте ещё раз — попытка уже возвращена."
    if "timed out" in text.lower() or "timeout" in text.lower():
        return "Источник отвечает слишком долго. Попробуйте ещё раз через несколько минут."
    return "Не удалось обработать видео. Проверьте ссылку или попробуйте другую запись."


def process(job_id, job):
    for old_clip in CLIPS.glob("*.mp4"):
        if old_clip.stat().st_mtime < time.time() - 30 * 86400:
            old_clip.unlink(missing_ok=True)
    JOBS[job_id] = {"status": "processing", "progress": 5}
    folder = Path(tempfile.mkdtemp(prefix="narezka-"))
    try:
        video, info = download(job["source"], folder)
        JOBS[job_id]["progress"] = 25
        audio = folder / "speech.wav"
        run([FFMPEG, "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)], 300)
        JOBS[job_id]["progress"] = 35
        with MODEL_LOCK:
            global MODEL
            if MODEL is None:
                accelerated = os.name == "posix" and os.uname().sysname == "Darwin"
                MODEL = WhisperModel("tiny", device="mps" if accelerated else "cpu", compute_type="float16" if accelerated else "int8")
            segments, _ = MODEL.transcribe(str(audio), language="ru", vad_filter=True, beam_size=1, condition_on_previous_text=False)
            segments = list(segments)
        JOBS[job_id]["progress"] = 65
        start, auto_title = highlight(segments, audio_peaks(video), job["duration"], job.get("wishes", ""))
        output = CLIPS / f"{job_id}.mp4"
        render(video, job, start, auto_title, segments, output)
        upload_clip(job_id, output)
        clip_base = CALLBACK if os.environ.get("CLIP_STORAGE") == "http" else PUBLIC_URL
        result = {"status": "done", "progress": 100, "clip_url": f"{clip_base}/clips/{job_id}.mp4", "title": job["title"] or auto_title, "source_title": info.get("title", "Видео"), "start": start}
        JOBS[job_id] = result
        callback(job_id, result)
    except Exception as error:
        print("Render failed:", str(error)[:1000])
        result = {"status": "error", "error": friendly_error(error)}
        JOBS[job_id] = result
        callback(job_id, result)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self.json(200, {"ok": True, "model": "faster-whisper-tiny", "cost": 0})
        if self.path.startswith("/clips/"):
            name = Path(self.path).name
            file = CLIPS / name
            if not file.exists() or not re.fullmatch(r"[a-f0-9-]+\.mp4", name):
                return self.json(404, {"error": "not found"})
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(file.stat().st_size))
            self.end_headers()
            self.wfile.write(file.read_bytes())
            return
        self.json(404, {"error": "not found"})

    def do_POST(self):
        if not hmac.compare_digest(self.headers.get("X-Processor-Secret", ""), SECRET):
            return self.json(401, {"error": "unauthorized"})
        if self.path == "/jobs":
            job = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if not safe_url(job.get("source", "")):
                return self.json(400, {"error": "unsupported source"})
            job["duration"] = int(job.get("duration", 30))
            if job["duration"] not in (15, 30, 45, 60) or len(job.get("title", "")) > 70 or len(job.get("wishes", "")) > 500:
                return self.json(400, {"error": "invalid options"})
            requested_id = str(job.get("id", ""))
            job_id = requested_id if re.fullmatch(r"[a-f0-9-]{36}", requested_id) else str(uuid.uuid4())
            if job_id in JOBS:
                return self.json(202, {"id": job_id, "status": JOBS[job_id].get("status", "processing")})
            threading.Thread(target=process, args=(job_id, job), daemon=True).start()
            return self.json(202, {"id": job_id})
        match = re.fullmatch(r"/jobs/([a-f0-9-]+)", self.path)
        if match:
            return self.json(200, JOBS.get(match.group(1), {"status": "unknown"}))
        self.json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        print(fmt % args)


if __name__ == "__main__":
    print(f"Нарезка renderer: http://127.0.0.1:{int(os.environ.get('PORT', '8788'))}")
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8788"))), Handler).serve_forever()
