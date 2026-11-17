"""Luma (Ray / uni-1) 어댑터 — 공통 흐름은 services/video_ai_service.py.

두 가지 API를 지원하며 키 형식으로 자동 선택한다 (config.LUMA_API_FLAVOR)
- agents (Luma API Platform 키 'luma-api-…'): https://agents.lumalabs.ai/v1, 모델 ray-3.2
    요청 {"model", "type": "video", "prompt", "aspect_ratio", "video": {"resolution", "duration", "start_frame": ImageRef}}
    ImageRef는 공개 URL {"url"} 또는 이미지 직접 첨부 {"data": base64, "media_type"} → 공개 URL 없이도 동작
    키프레임: {"model": uni-1, "type": "image", "prompt", "aspect_ratio", "image_ref": [ImageRef…]}
    완료 시 결과 주소: output[].url (1시간 유효한 서명 URL → 바로 다운로드)
- dream-machine (예전 키): https://api.lumalabs.ai/dream-machine/v1, 모델 ray-2
    요청 {"model", "prompt", "aspect_ratio", "resolution", "duration", "keyframes": {"frame0": {"type": "image", "url"}}}
    이미지는 공개 URL만 가능, 키프레임 생성·여자주인공 불가, 완료 시 영상 주소: assets.video
두 API 모두 인증은 'Authorization: Bearer <키>' 헤더. 키와 API 서버가 맞지 않으면 401/403 'Not authenticated'.
작업: POST /generations → GET /generations/{id} (state: queued | dreaming | completed | failed)

이미지 전달: LUMA_IMAGE_BASE_URL(http/https)이 있으면 {base}/media/... 공개 URL, 없으면 agents API에 한해 JPEG로 줄여
base64로 직접 첨부한다 (dream-machine API는 공개 URL이 꼭 필요).
"""

import base64
from pathlib import Path
from urllib.parse import quote

import httpx

from config import (
    LUMA_API_BASE, LUMA_API_FLAVOR, LUMA_API_KEY, LUMA_CONCURRENCY, LUMA_DURATION, LUMA_ENABLED, LUMA_IMAGE_BASE_URL,
    LUMA_IMAGE_MODEL, LUMA_MODEL, LUMA_POLL_SEC, LUMA_RESOLUTION, LUMA_SCENE_FRAME, LUMA_TIMEOUT_SEC, MEDIA_DIR,
)
from services import video_ai_service as ai

NAME = "luma"
LABEL = "Luma"
API_BASE = LUMA_API_BASE
MODEL = LUMA_MODEL
IMAGE_MODEL = LUMA_IMAGE_MODEL
CONCURRENCY = LUMA_CONCURRENCY
POLL_SEC = LUMA_POLL_SEC          # 문서 권장 2~5초 (조회는 생성 RPM 한도에 포함되지 않음)
TIMEOUT_SEC = LUMA_TIMEOUT_SEC
CACHE_PREFIX = "luma_v2_"
PROMPT_MAX_CHARS = 4000           # Agents API 한도 6000자 — 사용자 프롬프트를 맨 앞에 두므로 잘려도 덧붙인 지시부터 잘림
USER_PROMPT_MAX_CHARS = 2500
MAX_HEROINE_REFS = 8
DESCRIBE_REFS = True              # 키프레임 프롬프트에 캐릭터 외형 설명도 넣음
ASPECT_RATIO = ai.ASPECT_RATIO
# 생성 실패(state=failed) failure_code 분류 (docs.agents.lumalabs.ai/guides/error-handling)
RETRYABLE_FAILURES = {"generation_failed", "output_not_found", "rate_limited"}  # 같은 요청으로 다시 제출
FATAL_FAILURES = {"budget_exhausted"}                                            # 잔액 부족 → 남은 씬 중단
PUBLIC_IMAGE_BASE = LUMA_IMAGE_BASE_URL if LUMA_IMAGE_BASE_URL.startswith(("http://", "https://")) else ""


# ---------- 설정 ----------
def config_problem() -> str | None:
    if not LUMA_ENABLED:
        return "LUMA_ENABLED=false"
    if not LUMA_API_KEY:
        return "LUMA_API_KEY가 설정되지 않음"
    if LUMA_API_FLAVOR == "dream-machine" and not PUBLIC_IMAGE_BASE:
        return ("LUMA_IMAGE_BASE_URL(http/https 공개 주소)이 필요함 — 예전 Dream Machine 키는 이미지를 공개 URL로만 받음. "
                "platform.lumalabs.ai의 'luma-api-' 키를 쓰면 필요 없음")
    return None


def frame_mode() -> str:
    """설정상 첫 프레임 방식 (generate | pose). 키 유무·API 종류와 무관 — 클립 최신 여부 판단에 쓰므로 고정값이어야 함."""
    return "pose" if LUMA_SCENE_FRAME == "pose" else "generate"


def frame_supported() -> bool:
    """키프레임 생성(uni-1)은 Agents API에서만 가능."""
    return LUMA_API_FLAVOR == "agents"


def heroine_supported() -> bool:
    return LUMA_API_FLAVOR == "agents"


def _duration() -> str:
    # agents API: 첫 프레임을 지정하면 10s 불가 → 5s
    return "5s" if LUMA_API_FLAVOR == "agents" and LUMA_DURATION != "5s" else LUMA_DURATION


def status_extra() -> dict:
    return {"api": LUMA_API_FLAVOR, "resolution": LUMA_RESOLUTION, "duration": _duration(),
            "image": "public-url" if PUBLIC_IMAGE_BASE else "inline"}


def job_settings(mode: str) -> list:
    """캐시 파일 이름에 들어가는 생성 설정."""
    return [LUMA_API_FLAVOR, LUMA_MODEL, LUMA_RESOLUTION, _duration(), ASPECT_RATIO, mode,
            LUMA_IMAGE_MODEL if mode == "generate" else ""]


# ---------- 요청 만들기 ----------
def image_ref(path: Path) -> dict | None:
    """이미지 파일 → Luma ImageRef (공개 URL 또는 base64 JPEG)."""
    if PUBLIC_IMAGE_BASE and path.is_relative_to(MEDIA_DIR):
        return {"url": f"{PUBLIC_IMAGE_BASE}/media/{quote(path.relative_to(MEDIA_DIR).as_posix())}"}
    if LUMA_API_FLAVOR != "agents":
        return None
    return {"data": base64.b64encode(ai.inline_jpeg(path)).decode("ascii"), "media_type": "image/jpeg"}


def keyframe_ref(data: bytes, media_type: str) -> dict:
    return {"data": base64.b64encode(data).decode("ascii"), "media_type": media_type}


def ref_names(heroine_count: int) -> tuple[str, str, str]:
    """프롬프트에서 레퍼런스 이미지를 부르는 이름: (JB, 여자주인공 대표, 여자주인공 전체). image_ref 순서와 일치."""
    last = 1 + heroine_count
    return ("reference image 1", "reference image 2",
            "reference image 2" if heroine_count <= 1 else f"reference images 2-{last}")


def headers() -> dict:
    return {"Authorization": f"Bearer {LUMA_API_KEY}", "Accept": "application/json",
            "Content-Type": "application/json"}


def frame_request(job: ai.ClipJob) -> tuple[str, dict]:
    """키프레임 이미지 요청 (uni-1): 장면 프롬프트 + 캐릭터 포즈 이미지를 레퍼런스(image_ref)로."""
    return "/generations", {
        "model": LUMA_IMAGE_MODEL,
        "type": "image",
        "prompt": job.frame_prompt,
        "aspect_ratio": ASPECT_RATIO,
        "image_ref": [job.image, *job.heroine_refs],  # 1 = JB 포즈, 2~ = 여자주인공 (프롬프트의 번호와 일치)
    }


def video_request(job: ai.ClipJob, start_frame: dict | None = None) -> tuple[str, dict]:
    """영상 요청. start_frame: 생성한 키프레임 (없으면 캐릭터 포즈 이미지)."""
    if LUMA_API_FLAVOR == "agents":
        return "/generations", {
            "model": LUMA_MODEL,
            "type": "video",
            "prompt": job.prompt,
            "aspect_ratio": ASPECT_RATIO,
            "video": {"resolution": LUMA_RESOLUTION, "duration": _duration(), "start_frame": start_frame or job.image},
        }
    return "/generations", {
        "model": LUMA_MODEL,
        "prompt": job.prompt,
        "aspect_ratio": ASPECT_RATIO,
        "resolution": LUMA_RESOLUTION,
        "duration": LUMA_DURATION,
        "loop": False,
        "keyframes": {"frame0": {"type": "image", "url": job.image["url"]}},
    }


# ---------- 응답 해석 ----------
def task_path(task_id: str) -> str:
    return f"/generations/{task_id}"


def task_id(response: dict) -> str | None:
    return response.get("id")


def parse_task(generation: dict) -> ai.TaskState:
    state = generation.get("state")
    if state == "completed":
        # agents: output[].url (영상이면 type=video 우선) / dream-machine: assets.video
        outputs = [o for o in generation.get("output") or [] if isinstance(o, dict) and o.get("url")]
        videos = [o["url"] for o in outputs if o.get("type", "video") == "video"]
        urls = videos or [o["url"] for o in outputs]
        legacy = (generation.get("assets") or {}).get("video")
        return ai.TaskState("completed", tuple(urls + ([legacy] if legacy else [])))
    if state == "failed":
        return ai.TaskState("failed", failure=generation.get("failure_reason") or "",
                            code=generation.get("failure_code") or "")
    return ai.TaskState("running")


def failure_action(code: str) -> str:
    """생성 실패 코드 → retry(같은 요청 재제출) | fatal(남은 씬 중단) | give_up(그 씬만 포기)."""
    if code in FATAL_FAILURES:
        return "fatal"
    return "retry" if code in RETRYABLE_FAILURES else "give_up"


def http_error(r: httpx.Response) -> Exception:
    detail = ai.error_detail(r)
    request_id = r.headers.get("x-request-id")
    if request_id:  # Luma 지원팀 문의용
        detail += f" [request-id {request_id}]"
    if r.status_code in (401, 403):
        return ai.VideoAIFatal(f"Luma 인증 실패 ({r.status_code}): {detail} — {_auth_hint()}")
    if r.status_code == 402:
        return ai.VideoAIFatal(f"Luma 크레딧(잔액) 부족 (402). {detail}", kind="credits")
    return ai.transient_error(r, "Luma", detail) or ai.VideoAIError(f"Luma 요청 거부 ({r.status_code}) {detail}")


def _auth_hint() -> str:
    """인증 실패 원인 안내: 키 형식과 실제로 호출한 API 서버를 함께 보여준다 (키 값은 노출하지 않음)."""
    kind = "Luma API Platform 키(luma-api-…)" if LUMA_API_FLAVOR == "agents" else "예전 Dream Machine 키 형식"
    return (f"{kind}로 {LUMA_API_BASE} 를 호출했습니다. 키가 만료·삭제되지 않았는지, 이 서버용 키가 맞는지 확인하세요 "
            "(platform.lumalabs.ai에서 발급한 'luma-api-' 키는 agents.lumalabs.ai, 예전 키는 api.lumalabs.ai/dream-machine 전용"
            ", LUMA_API_BASE를 직접 지정했다면 비워 두세요).")
