"""WAV 컨테이너 처리 — Gemini TTS 응답 해석, 끝단 페이드, 손상 파일 복구.

Gemini 3.x TTS는 오디오를 '완성된 WAV 파일'(RIFF 헤더 + PCM + C2PA 출처 서명 청크)로 돌려준다.
예전 코드는 이것을 raw PCM으로 보고 WAV 헤더를 한 번 더 씌웠기 때문에
  - 앞: 안쪽 RIFF 헤더 44바이트가 샘플로 재생 → 짧은 '틱'
  - 끝: C2PA 서명(약 6KB)이 샘플로 재생 → 약 0.1초의 큰 잡음 ('툭' 끊기는 잡음의 원인)
이 모듈은 청크를 제대로 나눠 PCM만 오디오로 쓰고, 출처 서명 등 나머지 청크는 그대로 보존한다.
"""

import array
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

FADE_IN_SEC = 0.008    # 시작 '틱' 방지 (말소리에는 영향 없는 길이)
FADE_OUT_SEC = 0.06    # 끝단 여음을 부드럽게 마감


@dataclass
class WavData:
    pcm: bytes
    rate: int
    channels: int = 1
    bits: int = 16
    extra_chunks: list[tuple[bytes, bytes]] = field(default_factory=list)  # (청크 ID, 내용) — 예: C2PA 서명

    @property
    def duration(self) -> float:
        return len(self.pcm) / (self.rate * self.channels * self.bits // 8)


def parse_wav(data: bytes) -> WavData | None:
    """RIFF/WAVE 바이트 → fmt·data·기타 청크. WAV가 아니면 None."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos, fmt, pcm, extra = 12, None, None, []
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        body = data[pos + 8:pos + 8 + size]  # 크기가 실제보다 크게 적힌 경우(스트리밍) 남은 만큼만
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data" and pcm is None:
            pcm = body
        else:
            extra.append((cid, body))
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None or len(fmt) < 16:
        return None
    audio_format = int.from_bytes(fmt[0:2], "little")
    if audio_format != 1:  # PCM이 아니면 처리하지 않음
        return None
    channels = int.from_bytes(fmt[2:4], "little")
    rate = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")
    frame = channels * bits // 8
    return WavData(pcm[: len(pcm) - len(pcm) % frame], rate, channels, bits, extra)


def raw_pcm(data: bytes, mime_type: str | None, default_rate: int) -> WavData:
    """WAV가 아닌 raw PCM 응답 (예: audio/L16;codec=pcm;rate=24000)."""
    m = re.search(r"rate=(\d+)", mime_type or "")
    return WavData(data[: len(data) - len(data) % 2], int(m.group(1)) if m else default_rate)


def apply_fades(wav: WavData, fade_in: float = FADE_IN_SEC, fade_out: float = FADE_OUT_SEC) -> WavData:
    """16bit PCM 앞뒤에 짧은 페이드 (코사인 곡선) — 마지막 샘플이 0이 아닌 채 끊겨 생기는 팝 노이즈 제거."""
    if wav.bits != 16 or not wav.pcm:
        return wav
    samples = array.array("h", wav.pcm)
    ch = wav.channels
    frames = len(samples) // ch
    n_in = min(int(wav.rate * fade_in), frames // 4)
    n_out = min(int(wav.rate * fade_out), frames // 3)
    for i in range(n_in):
        g = 0.5 - 0.5 * math.cos(math.pi * i / n_in)
        for c in range(ch):
            samples[i * ch + c] = int(samples[i * ch + c] * g)
    for i in range(n_out):
        g = 0.5 + 0.5 * math.cos(math.pi * (i + 1) / n_out)  # 1 → 0, 마지막 샘플은 정확히 0
        k = frames - n_out + i
        for c in range(ch):
            samples[k * ch + c] = int(samples[k * ch + c] * g)
    return WavData(samples.tobytes(), wav.rate, wav.channels, wav.bits, wav.extra_chunks)


def write_wav(path: Path, wav: WavData) -> None:
    """표준 WAV로 저장 (fmt → data → 보존할 기타 청크 순서)."""
    block = wav.channels * wav.bits // 8
    fmt = (b"\x01\x00" + wav.channels.to_bytes(2, "little") + wav.rate.to_bytes(4, "little")
           + (wav.rate * block).to_bytes(4, "little") + block.to_bytes(2, "little") + wav.bits.to_bytes(2, "little"))
    chunks = [(b"fmt ", fmt), (b"data", wav.pcm), *wav.extra_chunks]
    body = b"WAVE" + b"".join(cid + len(c).to_bytes(4, "little") + c + (b"\x00" if len(c) & 1 else b"")
                              for cid, c in chunks)
    path.write_bytes(b"RIFF" + len(body).to_bytes(4, "little") + body)


def repair_wrapped_wav(path: Path) -> float | None:
    """예전 코드가 만든 '이중 WAV'(헤더 안에 또 WAV + 서명이 샘플로 들어간 파일)를 복구.
    복구했으면 새 길이(초), 정상 파일이면 None."""
    try:
        outer = parse_wav(path.read_bytes())
    except OSError:
        return None
    if outer is None:
        return None
    inner = parse_wav(outer.pcm)
    if inner is None:
        return None
    fixed = apply_fades(inner)
    tmp = path.with_name(path.stem + ".repair.tmp" + path.suffix)
    try:
        write_wav(tmp, fixed)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return fixed.duration
