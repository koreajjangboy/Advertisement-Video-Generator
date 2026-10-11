"""BGM 선택·기본 비트 합성, 그리고 오디오 음량 곡선(호스트 캐릭터 반응용).

BGM 우선순위: BGM_PATH가 파일이면 그 파일 → 폴더면 그 안의 첫 번째 음원(이름순, 생성 파일 제외)
→ 없고 BGM_GENERATE=true면 기본 붐뱁 비트를 순수 Python으로 합성해 media/bgm/_default_beat.wav에 캐시.
기본 비트는 이음매 없이 반복되는 2마디 루프(88BPM, Am7-Fmaj7)로, ffmpeg에서 -stream_loop로 영상 길이만큼 반복한다.
"""

import logging
import math
import random
import subprocess
import wave
from array import array
from pathlib import Path

from config import BGM_DIR, BGM_ENABLED, BGM_GENERATE, BGM_PATH

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
DEFAULT_BEAT = BGM_DIR / "_default_beat.wav"
BEAT_VERSION = "1"  # 합성 방식이 바뀌면 올려서 캐시를 다시 만든다

SR = 44100
BPM = 88


def resolve_bgm() -> Path | None:
    """렌더링에 쓸 BGM 파일 경로 (없으면 None)."""
    if not BGM_ENABLED:
        return None
    if BGM_PATH.is_file():
        return BGM_PATH
    if BGM_PATH.is_dir():
        files = sorted(p for p in BGM_PATH.iterdir()
                       if p.suffix.lower() in AUDIO_EXTENSIONS and not p.name.startswith("_"))
        if files:
            return files[0]
    if not BGM_GENERATE:
        return None
    try:
        return ensure_default_beat()
    except Exception:
        logger.exception("기본 BGM 합성 실패 — BGM 없이 렌더링합니다.")
        return None


def ensure_default_beat() -> Path:
    marker = DEFAULT_BEAT.with_suffix(".version")
    if DEFAULT_BEAT.exists() and marker.exists() and marker.read_text() == BEAT_VERSION:
        return DEFAULT_BEAT
    DEFAULT_BEAT.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEFAULT_BEAT.with_name("_default_beat.tmp.wav")
    _write_wav(tmp, _synthesize_beat())
    tmp.replace(DEFAULT_BEAT)
    marker.write_text(BEAT_VERSION)
    logger.info("기본 BGM 비트 생성: %s", DEFAULT_BEAT)
    return DEFAULT_BEAT


# ---------- 기본 비트 합성 (lo-fi 붐뱁) ----------
def _synthesize_beat() -> array:
    step = 60 / BPM / 4                      # 16분음표 길이(초)
    length = round(step * 32 * SR)          # 2마디
    buf = array("f", [0.0]) * length
    rng = random.Random(7)

    def add(start_sec: float, samples):
        """루프 끝을 넘는 꼬리는 처음으로 감아 넣어 이음매 없이 반복되게 한다."""
        i0 = round(start_sec * SR)
        for k, v in enumerate(samples):
            buf[(i0 + k) % length] += v

    def kick(n=int(0.38 * SR)):
        phase, out = 0.0, []
        for k in range(n):
            t = k / SR
            phase += 2 * math.pi * (45 + 75 * math.exp(-t * 28)) / SR
            out.append(math.sin(phase) * math.exp(-t * 7) * 0.9)
        return out

    def snare(n=int(0.28 * SR)):
        out = []
        for k in range(n):
            t = k / SR
            noise = rng.uniform(-1, 1) * math.exp(-t * 16) * 0.45
            tone = math.sin(2 * math.pi * 185 * t) * math.exp(-t * 30) * 0.35
            out.append(noise + tone)
        return out

    def hat(n=int(0.05 * SR), vol=0.12):
        out, prev = [], 0.0
        for k in range(n):
            x = rng.uniform(-1, 1)
            out.append((x - prev) * 0.5 * math.exp(-k / SR * 70) * vol)  # 1차 차분 = 간이 고역 통과
            prev = x
        return out

    def bass(freq, n=int(0.45 * SR)):
        return [math.sin(2 * math.pi * freq * k / SR) * min(k / 300, 1) * math.exp(-k / SR * 3.5) * 0.35
                for k in range(n)]

    def keys(freqs, n):
        out = []
        for k in range(n):
            t = k / SR
            env = min(t / 0.03, 1) * math.exp(-t * 0.9) * 0.05
            out.append(env * sum(math.sin(2 * math.pi * f * t) + 0.25 * math.sin(4 * math.pi * f * t) for f in freqs))
        return out

    k = kick()
    s = snare()
    roots = [55.0, 43.65]                                            # A1, F1
    chords = [[220.0, 261.63, 329.63, 392.0], [174.61, 220.0, 261.63, 329.63]]  # Am7, Fmaj7
    for bar in range(2):
        base = bar * 16
        add((base) * step, keys(chords[bar], int(step * 16 * SR)))
        for st in (0, 7, 10):
            add((base + st) * step, k)
            add((base + st) * step, bass(roots[bar]))
        for st in (4, 12):
            add((base + st) * step, s)
        for st in range(0, 16, 2):
            swing = step * 0.33 if (st // 2) % 2 else 0               # 엇박 스윙
            add((base + st) * step + swing, hat(vol=0.14 if st % 4 == 0 else 0.09))

    peak = max(abs(v) for v in buf) or 1.0
    gain = 0.8 / peak
    return array("f", (v * gain for v in buf))


def _write_wav(path: Path, samples: array) -> None:
    pcm = array("h", (int(max(-1.0, min(1.0, v)) * 32767) for v in samples))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


# ---------- 음량 곡선 ----------
def audio_envelope(ffmpeg: str, path: Path, fps: int, rate: int = 8000) -> list[float]:
    """오디오를 영상 프레임 단위 RMS 음량 목록으로 (호스트 캐릭터의 말하기 반응용)."""
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
         "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-500:])
    samples = array("h")
    samples.frombytes(result.stdout[: len(result.stdout) // 2 * 2])
    per_frame = rate / fps
    env, n = [], 0
    while round(n * per_frame) < len(samples):
        chunk = samples[round(n * per_frame): round((n + 1) * per_frame)]
        env.append(math.sqrt(sum(v * v for v in chunk) / len(chunk)) / 32768 if chunk else 0.0)
        n += 1
    return env
