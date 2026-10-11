"""내레이션 → 음성 파일(TTS).

- gemini: Gemini TTS 모델. 기존 GEMINI_API_KEY 사용, 24kHz 16bit mono WAV 저장.
  (무료 등급은 분당 요청 수가 매우 적어 429가 자주 발생)
- gtts: Google 번역 TTS(무료, 키 불필요). MP3 저장. 품질·톤 조절은 제한적.

TTS_PROVIDER가 기본 엔진, TTS_FALLBACK이 폴백 엔진이다. 기본 엔진이 사용량 한도(429)나
서버 오류로 실패하면 즉시 폴백 엔진으로 생성하고, 안내된 대기 시간 동안은 기본 엔진을 건너뛴다.

읽을 텍스트와 연출 지시의 분리
- 음성으로 바꾸는 것은 내레이션 대본뿐이다. 엔진에 넘기기 전에 services.narration.sanitize_narration으로
  괄호·대괄호 지시문, 화자 표기, 읽기 지시 문장, 이모지 등을 걷어낸다 (gTTS 포함 모든 엔진 공통).
- Gemini의 말투·템포 지시(TTS_STYLE)는 텍스트에 붙이지 않고 Part.speech_metadata.style(별도 필드)로 보낸다.
  Gemini 3.8 TTS는 text를 '그대로 읽을 대본'으로 취급하므로, 예전처럼 "지시:\n대본"으로 합치면 지시까지 읽힌다.
  모델이 style 필드를 지원하지 않으면(400) 지시 없이 대본만 보낸다 — 어떤 경우에도 지시문을 대본에 섞지 않는다.

오디오 후가공 (services/wav_utils.py)
- Gemini 응답은 완성된 WAV(RIFF 헤더 + PCM + C2PA 출처 서명)다. 청크를 나눠 PCM만 오디오로 쓰고 서명 청크는 보존한다
  (예전처럼 통째로 PCM 취급하면 앞에 '틱', 끝에 약 0.1초 큰 잡음).
- 앞 8ms 페이드 인 / 끝 60ms 페이드 아웃으로 팝 노이즈 없이 마감. 렌더링 합성 때도 afade로 한 번 더 다듬는다.

씬 오디오: 파일명에 엔진·내레이션 해시 + 생성 시각을 넣어 재생성할 때마다 새 URL이 되도록 한다
(브라우저 캐시 문제 없음, 재생 중 잠긴 파일 덮어쓰기 방지). 이전 파일은 호출 측에서 삭제.
미리듣기 오디오: media/audio/preview/에 해시 이름으로 저장·재사용하고, 오래된 파일은 자동 정리.
"""

import hashlib
import logging
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from google.genai import errors, types

from config import (
    AUDIO_DIR, MEDIA_DIR, TTS_FALLBACK, TTS_MODEL, TTS_PROVIDER, TTS_STYLE, TTS_VOICE,
)
from services.llm_service import is_rate_limit, retry_after
from services.narration import sanitize_narration
from services.wav_utils import apply_fades, parse_wav, raw_pcm, repair_wrapped_wav, write_wav

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # Gemini TTS가 raw PCM을 줄 때의 기본 샘플레이트
PREVIEW_DIR = AUDIO_DIR / "preview"
PREVIEW_TTL_SEC = 60 * 60  # 미리듣기 파일 보관 시간
MAX_RATE_LIMIT_RETRIES = 5  # 폴백이 없을 때 백그라운드 작업의 재시도 횟수
GTTS_RATE_LIMIT_SEC = 60.0  # gTTS 429는 대기 시간을 알려주지 않으므로 고정값
EXTENSIONS = {"gemini": ".wav", "gtts": ".mp3"}

for _name in filter(None, (TTS_PROVIDER, TTS_FALLBACK)):
    if _name not in EXTENSIONS:
        raise RuntimeError(f"지원하지 않는 TTS 엔진: {_name} (gemini | gtts)")

# 엔진별 '이 시각까지 사용하지 않음' (429 수신 시 설정, 프로세스 메모리)
_blocked_until: dict[str, float] = {}


class TTSRateLimited(RuntimeError):
    """TTS API 사용량 한도 초과."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"TTS 사용량 한도 초과 — 약 {retry_after:.0f}초 후 다시 시도하세요.")


@dataclass
class TTSResult:
    url: str            # /media/audio/... (DB 저장 및 브라우저 재생용)
    path: Path          # 실제 파일 경로
    duration_sec: float
    provider: str       # 실제로 사용된 엔진 (폴백 여부 확인용)


def media_url_to_path(url: str | None) -> Path | None:
    """'/media/audio/x.wav' → 실제 파일 경로."""
    if not url or not url.startswith("/media/"):
        return None
    return MEDIA_DIR / url.removeprefix("/media/")


def audio_exists(url: str | None) -> bool:
    path = media_url_to_path(url)
    return bool(path and path.exists())


def delete_media(url: str | None) -> None:
    """/media/... URL의 파일 삭제 (오디오·이미지·영상 공용)."""
    path = media_url_to_path(url)
    if path and path.exists():
        try:
            path.unlink(missing_ok=True)
        except OSError:  # Windows에서 재생 중 등으로 잠긴 파일
            logger.warning("미디어 파일 삭제 실패: %s", path)


def synthesize(text: str, basename: str, wait_on_limit: bool = False) -> TTSResult:
    """text를 음성으로 변환해 media/audio/{basename}_{엔진+hash}_{time}.{ext}에 저장.
    폴백 엔진이 없고 wait_on_limit=True(백그라운드 작업)면 한도 초과 시 기다렸다 재시도."""
    text = _clean(text)
    stamp = int(time.time() * 1000)

    def make_path(provider: str) -> Path:
        return AUDIO_DIR / f"{basename}_{_digest(provider, text)}_{stamp}{EXTENSIONS[provider]}"

    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return _synthesize_with_fallback(text, make_path)
        except TTSRateLimited as e:
            if not wait_on_limit or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            logger.info("TTS 한도 초과, %.0f초 대기 후 재시도 (%d/%d)", e.retry_after, attempt + 1, MAX_RATE_LIMIT_RETRIES)
            time.sleep(e.retry_after + 1)


def preview(text: str) -> TTSResult:
    """저장하지 않은 내레이션 미리듣기. 같은 텍스트·엔진 설정이면 기존 파일 재사용."""
    text = _clean(text)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_previews()

    def make_path(provider: str) -> Path:
        return PREVIEW_DIR / f"preview_{_digest(provider, text)}{EXTENSIONS[provider]}"

    for provider in _provider_chain():
        path = make_path(provider)
        if path.exists():
            return TTSResult(media_path_to_url(path), path, round(read_duration(path), 2), provider)
    return _synthesize_with_fallback(text, make_path)


def active_provider() -> str:
    """현재 실제로 사용될 엔진 (기본 엔진이 한도 초과로 쉬는 중이면 폴백)."""
    for provider in _provider_chain():
        if _blocked_until.get(provider, 0) <= time.time():
            return provider
    return TTS_PROVIDER


# ---------- 내부 유틸 ----------
def _provider_chain() -> list[str]:
    chain = [TTS_PROVIDER]
    if TTS_FALLBACK and TTS_FALLBACK != TTS_PROVIDER:
        chain.append(TTS_FALLBACK)
    return chain


def _synthesize_with_fallback(text: str, make_path) -> TTSResult:
    """기본 엔진이 어떤 이유로든(한도·서버·네트워크·키 오류, 빈 응답) 실패하면 폴백 엔진으로 생성.
    마지막 엔진의 한도 초과는 TTSRateLimited로, 그 외 실패는 원인을 담은 RuntimeError로 올린다."""
    chain = _provider_chain()
    failures: list[str] = []
    for i, provider in enumerate(chain):
        is_last = i == len(chain) - 1
        if _blocked_until.get(provider, 0) > time.time() and not is_last:
            continue  # 한도 대기 중인 엔진은 바로 건너뜀
        try:
            return _synthesize_to(provider, text, make_path(provider))
        except TTSRateLimited as e:
            _blocked_until[provider] = time.time() + e.retry_after
            if is_last:
                raise  # 마지막 엔진까지 한도 초과 → 호출 측에서 대기·재시도 가능
            error: Exception = e
        except Exception as e:
            if is_last and not failures:
                raise  # 폴백 없이 단일 엔진만 시도한 경우 원래 예외 그대로
            error = e
        failures.append(f"{provider}: {error}")
        if not is_last:
            logger.warning("TTS '%s' 실패 → '%s'로 폴백: %s", provider, chain[i + 1], error)
    raise RuntimeError("모든 TTS 엔진 실패 — " + " / ".join(failures))


def _clean(text: str) -> str:
    """엔진에 넘길 순수 대사만 남긴다 (연출 지시·화자 표기·스타일 문구가 섞여 있으면 제거)."""
    cleaned = sanitize_narration(text, (TTS_STYLE,))
    if not cleaned:
        raise ValueError("TTS 대상 내레이션이 비어 있습니다 (연출 지시를 제외하면 읽을 문장이 없음).")
    if cleaned != (text or "").strip():
        logger.info("TTS 텍스트에서 연출 지시·표기를 제거했습니다: %r → %r", (text or "")[:80], cleaned[:80])
    return cleaned


def _digest(provider: str, text: str) -> str:
    settings = f"{TTS_MODEL}|{TTS_VOICE}|{TTS_STYLE}" if provider == "gemini" else "ko"
    return provider[0] + hashlib.sha1(f"{provider}|{settings}|{text}".encode()).hexdigest()[:10]


def media_path_to_url(path: Path) -> str:
    return "/media/" + path.relative_to(MEDIA_DIR).as_posix()


def _synthesize_to(provider: str, text: str, path: Path) -> TTSResult:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)  # 생성 도중 실패 시 불완전 파일 방지
    try:
        duration = _gemini_tts(text, tmp) if provider == "gemini" else _gtts(text, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return TTSResult(media_path_to_url(path), path, round(duration, 2), provider)


def read_duration(path: Path) -> float:
    if path.suffix == ".wav":
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    from mutagen import File as MutagenFile

    audio = MutagenFile(str(path))
    # 주의: mutagen 객체는 태그가 없으면 bool()이 False (gTTS mp3는 ID3 태그가 없음) → None 비교 필수
    return audio.info.length if audio is not None else 0.0


def repair_audio_files() -> dict[Path, float]:
    """예전 코드가 저장한 손상 WAV(이중 헤더 + 끝단 서명 잡음)를 복구. {경로: 새 길이(초)}."""
    repaired = {}
    for folder in (AUDIO_DIR, PREVIEW_DIR):
        for path in folder.glob("*.wav") if folder.exists() else []:
            if ".tmp" in path.name:
                continue
            try:
                duration = repair_wrapped_wav(path)
            except Exception as e:  # 재생 중 잠긴 파일 등 — 다음 기회에
                logger.warning("오디오 복구 실패 (%s): %s", path.name, e)
                continue
            if duration is not None:
                repaired[path] = duration
    if repaired:
        logger.info("손상된 TTS 오디오 %d개 복구 (끝단 잡음 제거)", len(repaired))
    return repaired


def cleanup_previews() -> None:
    _cleanup_previews()


def _cleanup_previews() -> None:
    cutoff = time.time() - PREVIEW_TTL_SEC
    for f in PREVIEW_DIR.glob("preview_*"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


# ---------- Providers ----------
def _gemini_tts(text: str, out_path: Path) -> float:
    from services.llm_service import get_client  # 같은 Gemini 클라이언트 재사용

    style = TTS_STYLE.strip() if TTS_STYLE and TTS_MODEL not in _style_unsupported else ""
    try:
        try:
            response = _gemini_tts_request(get_client(), text, style)
        except errors.ClientError as e:
            if not style or is_rate_limit(e) or e.code != 400:
                raise
            # 스타일 필드를 모르는 모델 — 지시를 대본에 섞지 않고 대본만 다시 요청
            logger.warning("TTS 모델 %s가 스타일 지시(speech_metadata.style)를 지원하지 않아 대본만 보냅니다: %s",
                           TTS_MODEL, e.message or e)
            _style_unsupported.add(TTS_MODEL)
            response = _gemini_tts_request(get_client(), text, "")
    except errors.ClientError as e:
        if is_rate_limit(e):
            raise TTSRateLimited(retry_after(e)) from e
        raise RuntimeError(f"Gemini TTS 요청 오류({e.code}): {e.message or e}") from e
    try:
        blob = response.candidates[0].content.parts[0].inline_data
        data, mime_type = blob.data, blob.mime_type
    except (IndexError, AttributeError, TypeError):
        raise RuntimeError(f"Gemini TTS 응답에 오디오가 없습니다: {response!r}"[:500])
    if not data:
        raise RuntimeError("Gemini TTS 응답 오디오가 비어 있습니다.")

    # 완성된 WAV(헤더·서명 청크 포함)면 청크를 나눠 PCM만, 아니면 raw PCM으로 해석
    wav = parse_wav(data) or raw_pcm(data, mime_type, SAMPLE_RATE)
    wav = apply_fades(wav)
    write_wav(out_path, wav)
    return wav.duration


_style_unsupported: set[str] = set()  # 스타일 필드를 거부한 TTS 모델 (프로세스 동안 기억)


def _gemini_tts_request(client, text: str, style: str):
    """text: 그대로 읽을 대본 / style: 말투·템포 지시 — 서로 다른 필드로 전달."""
    part = types.Part(text=text, speech_metadata=types.SpeechMetadata(style=style) if style else None)
    return client.models.generate_content(
        model=TTS_MODEL,
        contents=[types.Content(role="user", parts=[part])],
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
                )
            ),
        ),
    )


def _gtts(text: str, out_path: Path) -> float:
    from gtts import gTTS, gTTSError
    from mutagen.mp3 import MP3

    try:
        gTTS(text=text, lang="ko", timeout=30).save(str(out_path))
    except gTTSError as e:
        if getattr(e.rsp, "status_code", None) == 429:  # Google 번역 TTS도 과다 호출 시 차단
            raise TTSRateLimited(GTTS_RATE_LIMIT_SEC) from e
        raise RuntimeError(f"gTTS 실패: {e}") from e
    return MP3(str(out_path)).info.length
