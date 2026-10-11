"""Gemini(google-genai)를 이용한 도서 숏폼 광고 대본 생성.

『래퍼의 노트』 원문 발췌를 입력받아 훅(Hook) → 바디(Body) → CTA 구조의
장면(scene) 리스트를 Structured Outputs(Pydantic 스키마)로 반환한다.

한도 초과(429)·서버 오류 대응:
1. GEMINI_MODEL → GEMINI_FALLBACK_MODELS 순으로 시도 (한도 초과 모델은 안내된 시간 동안 건너뜀)
2. 모두 막혔고 대기 시간이 LLM_MAX_WAIT_SEC 이하면 기다렸다 재시도 (백그라운드 작업만)
3. 그래도 실패하면 LLM_FALLBACK=mock 일 때 템플릿 대본, 아니면 LLMRateLimited(안내 메시지)
LLM_PROVIDER=mock(또는 API_MODE=offline)이면 Gemini를 전혀 호출하지 않는다.
"""

import logging
import re
import time
from typing import Literal

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, PrivateAttr

from config import (
    GEMINI_API_KEY, GEMINI_FALLBACK_MODELS, GEMINI_MODEL, GEMINI_TIMEOUT_SEC, LLM_FALLBACK,
    LLM_MAX_WAIT_SEC, LLM_PROVIDER, SOURCE_TEXT_MAX_CHARS, HEROINE_CHARACTER,
)

logger = logging.getLogger(__name__)

Section = Literal["hook", "body", "cta"]
MAX_RATE_LIMIT_RETRIES = 3
DAILY_QUOTA_BLOCK_SEC = 60 * 60  # 일일 한도 소진 시 해당 모델을 쉬게 할 시간


# ---------- Structured Output 스키마 ----------
class Scene(BaseModel):
    narration: str = Field(
        description="TTS가 그대로 소리 내어 읽을 한국어 구어체 내레이션. 한 호흡에 읽히는 1~2문장. "
                    "실제로 말할 문장만 쓸 것 — (밝게), [효과음], '내레이션:' 같은 연출 지시·화자 표기·이모지 금지.",
    )
    on_screen_text: str = Field(description="화면에 크게 띄울 자막. 15자 이내, 임팩트 있게.")
    visual_prompt: str = Field(
        description="이 장면의 배경 이미지/영상 생성용 영어 프롬프트. 세로형 9:16, 분위기·조명·구도 포함."
    )
    duration_sec: float = Field(description="장면 길이(초).")
    emoji: str = Field(
        default="",
        description="이 장면의 내용·감정을 상징하는 이모지 1~2개 (화면 중앙 스티커 일러스트로 사용). "
                    "단일 이모지만, 사람·피부색·ZWJ 조합 이모지는 피할 것. 예: 📚✍️, 🎤🔥, 🌙💭",
    )
    heroine: bool = Field(
        default=False,
        description="이 장면에 여자주인공이 함께 등장하면 true. 여자주인공 등장이 요청된 경우에만 사용 (아니면 항상 false).",
    )
    host_pose: str = Field(
        default="",
        description="이 장면에서 진행자 캐릭터가 취할 포즈. 대사·감정선에 맞게 다음 중 하나: "
                    "surprised(놀람·질문·반전), thinking(회상·고민·감성), confident(추천·강조·확신), "
                    "pointing(시청자를 가리키며 행동 유도), waving(인사·팔로우·마무리). "
                    "연속된 장면에 같은 포즈를 반복하지 말 것.",
    )


class ShortFormScript(BaseModel):
    hook: Scene = Field(description="첫 3초 안에 스크롤을 멈추게 하는 훅 장면. 질문·반전·강렬한 원문 한 줄 활용.")
    body: list[Scene] = Field(
        description="책의 핵심 감성과 에피소드를 전달하는 장면들(2~5개). 원문의 문장·이미지를 살릴 것."
    )
    cta: Scene = Field(description="도서 구매와 채널 구독/팔로우를 유도하는 마지막 장면.")
    caption: str = Field(description="업로드 시 게시물 본문으로 쓸 1~2문장 캡션.")
    hashtags: list[str] = Field(description="'#'을 포함한 해시태그 5~8개.")

    # 실제 생성 경로 (Gemini 모델명 | mock | mock-fallback) — Gemini 스키마에는 포함되지 않음
    _provider: str = PrivateAttr(default="gemini")

    @property
    def provider(self) -> str:
        return self._provider

    def to_scenes(self) -> list[tuple[Section, Scene]]:
        return [("hook", self.hook), *(("body", s) for s in self.body), ("cta", self.cta)]


# ---------- 프롬프트 ----------
SYSTEM_INSTRUCTION = """당신은 도서 마케팅 전문 숏폼(릴스·쇼츠·틱톡) 크리에이티브 디렉터이자 카피라이터입니다.
책의 원문에서 가장 감정이 응축된 문장과 장면을 골라, 책을 읽지 않은 사람도 끝까지 보게 만드는 세로형 광고 대본을 씁니다.

원칙:
- 훅(Hook): 첫 3초 안에 스크롤을 멈추게 할 것. 설명으로 시작하지 말고 질문, 반전, 원문 속 강렬한 한 줄로 시작.
- 바디(Body): 책의 핵심 감성과 에피소드를 짧은 컷으로 리듬감 있게 전달. 원문의 표현을 인용하거나 살리되, 원문에 없는 사실을 지어내지 말 것.
- CTA: 도서 구매(예: '프로필 링크', '지금 서점에서')와 채널 구독/팔로우를 자연스럽게 함께 유도.
- 래퍼의 노트라는 정체성에 맞게 라임과 플로우가 느껴지는 리듬감 있는 문장을 선호.
- 내레이션은 한국어 구어체, 과장된 광고 멘트나 클리셰는 피할 것.
- narration에는 음성으로 읽힐 대사만 쓸 것. 톤·속도·효과음 같은 연출 지시는 넣지 말 것 (그대로 읽힘).
"""

USER_TEMPLATE = """다음 정보를 바탕으로 숏폼 광고 대본을 작성하세요.

[도서 정보]
- 제목: {book_title}
- 소개: {book_description}
- 타깃 시청자: {target_audience}
- 목표 영상 길이: 총 {duration}초 (hook은 3초 이내, 모든 장면 duration_sec 합계 = {duration}초)

[원문 발췌]
<<<
{source_text}
>>>
"""


HEROINE_TEMPLATE = """
[등장인물]
- 남자 주인공: 영상을 이끄는 진행자 JB (모든 장면에 등장)
- 여자주인공: {heroine} — 이야기 흐름상 어울리는 1~3개 장면(주로 body)에만 함께 등장시키고, 그 장면은 heroine=true.
  여자주인공 장면의 visual_prompt에는 두 사람이 함께 있는 장면(위치·거리·시선·분위기)을 묘사할 것.
"""
NO_HEROINE_NOTE = "\n(여자주인공은 등장하지 않음 — 모든 장면 heroine=false)\n"


# ---------- 클라이언트 / 오류 분류 (tts_service와 공유) ----------
_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다 (.env 확인).")
        _client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=int(GEMINI_TIMEOUT_SEC * 1000)),  # ms
        )
    return _client


class LLMRateLimited(RuntimeError):
    """대본 생성용 Gemini 모델이 모두 사용량 한도 초과."""

    def __init__(self, retry_after: float, daily: bool = False):
        self.retry_after = retry_after
        if daily:
            msg = ("Gemini 일일 사용량 한도를 모두 소진했습니다. 내일 다시 시도하거나 .env에서 "
                   "GEMINI_FALLBACK_MODELS(다른 모델), LLM_FALLBACK=mock 또는 API_MODE=offline을 설정하세요.")
        else:
            msg = f"Gemini 사용량 한도 초과 — 약 {retry_after:.0f}초 후 다시 시도하세요."
        super().__init__(msg)


def is_rate_limit(e: Exception) -> bool:
    return isinstance(e, errors.ClientError) and e.code == 429


def is_transient(e: Exception) -> bool:
    """잠시 후 다시 하면 성공할 수 있는 오류 (5xx, 네트워크/타임아웃)."""
    return isinstance(e, (errors.ServerError, httpx.TransportError, TimeoutError, ConnectionError))


def is_daily_quota(e: Exception) -> bool:
    return "PerDay" in str(e)


def retry_after(e: Exception, default: float = 20.0) -> float:
    """429 응답에 안내된 대기 시간(초). 일일 한도면 길게 잡는다."""
    if is_daily_quota(e):
        return DAILY_QUOTA_BLOCK_SEC
    m = re.search(r"retry in ([\d.]+)s", str(e)) or re.search(r"retryDelay\W+(\d+)s", str(e))
    return float(m.group(1)) if m else default


# 모델별 '이 시각까지 사용하지 않음' (429 수신 시 설정, 프로세스 메모리)
_blocked_until: dict[str, float] = {}


def active_model() -> str:
    """현재 대본 생성에 실제로 쓰일 경로 (대시보드 표시용)."""
    if LLM_PROVIDER == "mock":
        return "mock"
    for model in dict.fromkeys([GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]):
        if _blocked_until.get(model, 0) <= time.time():
            return model
    return "mock" if LLM_FALLBACK == "mock" else "rate-limited"


# ---------- 대본 생성 ----------
def generate_script(
    book_title: str,
    source_text: str,
    book_description: str = "",
    target_audience: str = "",
    duration_sec: int = 30,
    wait_on_limit: bool = False,
    heroine: bool = False,
) -> ShortFormScript:
    """wait_on_limit=True(백그라운드 작업)면 짧은 한도 대기는 기다렸다 재시도한다."""
    if not source_text.strip():
        raise ValueError("원문(source_text)이 비어 있습니다.")

    if LLM_PROVIDER == "mock":
        script = mock_script(book_title, source_text, duration_sec)
        if heroine:  # 템플릿 대본: 본문 장면에 등장
            for scene in script.body:
                scene.heroine = True
    else:
        prompt = USER_TEMPLATE.format(
            book_title=book_title,
            book_description=book_description or "(없음)",
            target_audience=target_audience or "책과 힙합에 관심 있는 20~30대",
            duration=duration_sec,
            source_text=source_text.strip()[:SOURCE_TEXT_MAX_CHARS],
        ) + (HEROINE_TEMPLATE.format(heroine=HEROINE_CHARACTER) if heroine else NO_HEROINE_NOTE)
        try:
            script = _generate_with_fallback(prompt, wait_on_limit)
        except Exception as e:
            if not (isinstance(e, LLMRateLimited) or is_transient(e)):
                raise
            if LLM_FALLBACK != "mock":
                if isinstance(e, LLMRateLimited):
                    raise
                raise RuntimeError(f"Gemini 서버/네트워크 오류로 대본 생성 실패: {e}") from e
            logger.warning("Gemini 대본 생성 불가 → 템플릿 대본으로 대체: %s", e)
            script = mock_script(book_title, source_text, duration_sec)
            script._provider = "mock-fallback"

    _fit_duration(script, duration_sec)
    return script


def _generate_with_fallback(prompt: str, wait_on_limit: bool) -> ShortFormScript:
    models = list(dict.fromkeys([GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]))

    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        last_error: Exception | None = None
        for model in models:
            if _blocked_until.get(model, 0) > time.time():
                continue
            try:
                script = _request(model, prompt)
                script._provider = model
                return script
            except errors.ClientError as e:
                if not is_rate_limit(e):
                    raise RuntimeError(_client_error_message(e)) from e
                _blocked_until[model] = time.time() + retry_after(e)
                last_error = e
            except Exception as e:
                if not is_transient(e):
                    raise
                last_error = e
            logger.warning("Gemini '%s' 대본 생성 실패: %s", model, last_error)

        if last_error is not None and not is_rate_limit(last_error):
            raise last_error  # 서버/네트워크 오류 — 호출 측에서 처리
        wait = max(min(_blocked_until.get(m, 0) for m in models) - time.time(), 0)
        daily = last_error is not None and is_daily_quota(last_error)
        if not wait_on_limit or daily or wait > LLM_MAX_WAIT_SEC or attempt == MAX_RATE_LIMIT_RETRIES:
            raise LLMRateLimited(wait, daily=daily)
        logger.info("Gemini 한도 초과, %.0f초 대기 후 재시도 (%d/%d)", wait, attempt + 1, MAX_RATE_LIMIT_RETRIES)
        time.sleep(wait + 1)
    raise AssertionError("unreachable")


def _client_error_message(e: errors.ClientError) -> str:
    if e.code in (401, 403):
        return f"Gemini API 키 또는 권한 오류({e.code}) — .env의 GEMINI_API_KEY를 확인하세요."
    if e.code == 404:
        return f"Gemini 모델을 찾을 수 없습니다({e.code}) — GEMINI_MODEL/GEMINI_FALLBACK_MODELS를 확인하세요."
    return f"Gemini 요청 오류({e.code}): {e.message or e}"


def _request(model: str, prompt: str) -> ShortFormScript:
    response = get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=ShortFormScript,
        ),
    )

    script = response.parsed
    if not isinstance(script, ShortFormScript):
        # SDK가 파싱하지 못한 경우 원본 JSON으로 한 번 더 검증
        if not response.text:
            raise RuntimeError("Gemini 응답이 비어 있습니다 (안전 필터 차단 가능성).")
        script = ShortFormScript.model_validate_json(response.text)
    return script


def _fit_duration(script: ShortFormScript, target: int) -> None:
    """장면 길이 합계를 목표 길이에 맞게 비례 보정한다 (훅은 3초 상한)."""
    script.hook.duration_sec = min(max(script.hook.duration_sec, 1.0), 3.0)
    rest = [s for _, s in script.to_scenes()[1:]]
    remaining = max(target - script.hook.duration_sec, len(rest))
    current = sum(max(s.duration_sec, 0.5) for s in rest)
    for s in rest:
        s.duration_sec = round(max(s.duration_sec, 0.5) * remaining / current, 1)


# ---------- 오프라인 템플릿 대본 (API 호출 없음) ----------
def mock_script(book_title: str, source_text: str, duration_sec: int = 30) -> ShortFormScript:
    """원문 문장을 그대로 잘라 훅/바디/CTA를 구성한다. 반복 테스트·한도 소진 시 대체용."""
    sentences = [s for s in _split_sentences(source_text[:SOURCE_TEXT_MAX_CHARS]) if len(s) >= 4]
    if not sentences:
        sentences = [source_text.strip()[:80]]
    body_count = max(2, min(4, (duration_sec - 8) // 6))
    hook_line = next((s for s in sentences if s.endswith("?")), sentences[0])
    body_lines = [s for s in sentences if s != hook_line][:body_count] or [hook_line]

    def scene(narration: str, title: str, prompt: str, sec: float) -> Scene:
        return Scene(narration=narration, on_screen_text=title, visual_prompt=prompt, duration_sec=sec)

    style = "vertical 9:16, cinematic, moody lighting, urban night"
    script = ShortFormScript(
        hook=scene(hook_line, _headline(hook_line), f"close-up of handwritten lyrics, {style}", 3),
        body=[scene(line, _headline(line), f"rapper writing in a notebook, {style}", 6) for line in body_lines],
        cta=scene(
            f"『{book_title}』, 지금 프로필 링크에서 만나보세요. 팔로우하면 다음 이야기도 들려드릴게요.",
            f"『{book_title}』", f"book on a desk under a warm spotlight, {style}", 5,
        ),
        caption=f"『{book_title}』 속 한 줄이 당신의 하루를 바꿀지도.",
        hashtags=["#래퍼의노트", "#책추천", "#힙합", "#북스타그램", "#숏폼"],
    )
    script._provider = "mock"
    return script


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。…])\s+|\n+", text.strip())
    return [p.strip().strip("\"'“”‘’") for p in parts if p.strip()]


def _headline(sentence: str, limit: int = 15) -> str:
    s = re.sub(r"[.!。…]+$", "", sentence)
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"
