"""Runway (Gen-4.5 / gen4_image) 어댑터 — 공통 흐름은 services/video_ai_service.py.

API: https://api.dev.runwayml.com/v1 (docs.dev.runwayml.com)
- 인증: 'Authorization: Bearer <키>' + 'X-Runway-Version: 2024-11-06' 헤더 (버전 헤더가 없으면 요청 거부)
- 영상: POST /image_to_video {"model": gen4.5, "promptText", "promptImage", "ratio": "720:1280", "duration": 2~10}
- 키프레임: POST /text_to_image {"model": gen4_image, "promptText", "ratio", "referenceImages": [{"uri", "tag"}]}
    레퍼런스는 최대 3장이고, 프롬프트에서 @태그로 부른다 (@hero = JB, @heroine… = 여자주인공)
- 작업 조회: GET /tasks/{id} → status PENDING | THROTTLED | RUNNING | SUCCEEDED | FAILED | CANCELLED
    완료 시 output: [URL] (24~48시간 유효 — 바로 다운로드, 만료되면 작업을 다시 조회해 새 URL)
    THROTTLED는 계정 동시 실행 한도로 대기 중인 상태 (오류 아님)
- 이미지는 data URI(base64, 5MB 이하)로 요청에 직접 담는다 → 공개 URL 불필요
- promptText는 1000자(UTF-16 코드 단위) 이하 — 사용자 프롬프트를 짧게 자르고 덧붙인 지시는 뒤부터 잘린다

오류 (docs.dev.runwayml.com/errors)
- 400: 잘못된 입력 (크레딧 부족도 400 + error 메시지로 옴) / 401: 키가 잘못됨
- 429·502·503·504: 재시도 / 작업 실패 failureCode: SAFETY.*·ASSET.INVALID는 재시도 무의미,
  INTERNAL·INTERNAL.BAD_OUTPUT.*·INPUT_PREPROCESSING.INTERNAL·THIRD_PARTY.UNAVAILABLE은 잠시 뒤 재시도
"""

import base64
from pathlib import Path

import httpx

from config import (
    RUNWAY_API_BASE, RUNWAY_API_KEY, RUNWAY_API_VERSION, RUNWAY_CONCURRENCY, RUNWAY_DURATION, RUNWAY_ENABLED,
    RUNWAY_IMAGE_MODEL, RUNWAY_MODEL, RUNWAY_POLL_SEC, RUNWAY_RATIO, RUNWAY_SCENE_FRAME, RUNWAY_TIMEOUT_SEC,
)
from services import video_ai_service as ai

NAME = "runway"
LABEL = "Runway"
API_BASE = RUNWAY_API_BASE
MODEL = RUNWAY_MODEL
IMAGE_MODEL = RUNWAY_IMAGE_MODEL
CONCURRENCY = RUNWAY_CONCURRENCY
POLL_SEC = RUNWAY_POLL_SEC
TIMEOUT_SEC = RUNWAY_TIMEOUT_SEC  # THROTTLED(동시 실행 한도 대기) 시간 포함
CACHE_PREFIX = "runway_v1_"
PROMPT_MAX_CHARS = 1000
USER_PROMPT_MAX_CHARS = 1000      # 넘치면 공통 흐름이 장면 묘사를 줄여 연기·일관성 지시가 들어갈 자리를 남긴다
DESCRIBE_REFS = False             # 키프레임 프롬프트의 외형 설명 생략 (@태그 레퍼런스 이미지로 충분, 1000자 절약)
MAX_HEROINE_REFS = 2              # gen4_image 레퍼런스 최대 3장 = JB 1 + 여자주인공 2
LEAD_TAG = "hero"                 # 태그: 3~16자, 소문자로 시작, 소문자·숫자·밑줄만
HEROINE_TAGS = ("heroine", "heroine2")
RUNNING = {"PENDING", "THROTTLED", "RUNNING"}
# 작업 실패 failureCode 중 같은 요청을 다시 제출해 볼 만한 것 (그 외는 그 씬만 포기)
RETRYABLE_FAILURES = {"INTERNAL", "INPUT_PREPROCESSING.INTERNAL", "THIRD_PARTY.UNAVAILABLE", ""}
RETRYABLE_PREFIXES = ("INTERNAL.BAD_OUTPUT",)


# ---------- 설정 ----------
def config_problem() -> str | None:
    if not RUNWAY_ENABLED:
        return "RUNWAY_ENABLED=false"
    if not RUNWAY_API_KEY:
        return "RUNWAY_API_KEY가 설정되지 않음"
    return None


def frame_mode() -> str:
    """설정상 첫 프레임 방식 (generate | pose). 클립 최신 여부 판단에 쓰므로 키 유무와 무관한 고정값."""
    return "generate" if RUNWAY_SCENE_FRAME == "generate" else "pose"


def frame_supported() -> bool:
    return True


def heroine_supported() -> bool:
    return True


def status_extra() -> dict:
    return {"ratio": RUNWAY_RATIO, "duration": f"{RUNWAY_DURATION}s", "image": "inline"}


def job_settings(mode: str) -> list:
    """캐시 파일 이름에 들어가는 생성 설정."""
    return [NAME, RUNWAY_MODEL, RUNWAY_RATIO, RUNWAY_DURATION, mode,
            RUNWAY_IMAGE_MODEL if mode == "generate" else ""]


# ---------- 요청 만들기 ----------
def image_ref(path: Path) -> str:
    """이미지 파일 → data URI (JPEG로 줄여서 — 5MB 한도 안쪽)."""
    return keyframe_ref(ai.inline_jpeg(path), "image/jpeg")


def keyframe_ref(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def ref_names(heroine_count: int) -> tuple[str, str, str]:
    """프롬프트에서 레퍼런스 이미지를 부르는 이름: (JB, 여자주인공 대표, 여자주인공 전체) — @태그."""
    tags = [f"@{t}" for t in HEROINE_TAGS[:max(heroine_count, 1)]]
    return f"@{LEAD_TAG}", tags[0], " and ".join(tags)


def headers() -> dict:
    return {"Authorization": f"Bearer {RUNWAY_API_KEY}", "X-Runway-Version": RUNWAY_API_VERSION,
            "Accept": "application/json", "Content-Type": "application/json"}


def frame_request(job: ai.ClipJob) -> tuple[str, dict]:
    """키프레임 이미지 요청 (gen4_image): 장면 프롬프트 + JB(·여자주인공) 이미지를 태그 붙은 레퍼런스로."""
    refs = [{"uri": job.image, "tag": LEAD_TAG}]
    refs += [{"uri": uri, "tag": tag} for uri, tag in zip(job.heroine_refs, HEROINE_TAGS)]
    return "/text_to_image", {
        "model": RUNWAY_IMAGE_MODEL,
        "promptText": job.frame_prompt,
        "ratio": RUNWAY_RATIO,
        "referenceImages": refs,
    }


def video_request(job: ai.ClipJob, start_frame: str | None = None) -> tuple[str, dict]:
    """영상 요청. start_frame: 생성한 키프레임 (없으면 캐릭터 포즈 이미지)."""
    return "/image_to_video", {
        "model": RUNWAY_MODEL,
        "promptText": job.prompt,
        "promptImage": start_frame or job.image,
        "ratio": RUNWAY_RATIO,
        "duration": RUNWAY_DURATION,
    }


# ---------- 응답 해석 ----------
def task_path(task_id: str) -> str:
    return f"/tasks/{task_id}"


def task_id(response: dict) -> str | None:
    return response.get("id")


def parse_task(task: dict) -> ai.TaskState:
    status = task.get("status")
    if status == "SUCCEEDED":
        return ai.TaskState("completed", tuple(u for u in task.get("output") or [] if isinstance(u, str)))
    if status in RUNNING:
        return ai.TaskState("running")
    code = "CANCELLED" if status == "CANCELLED" else (task.get("failureCode") or "")
    return ai.TaskState("failed", failure=task.get("failure") or (status or ""), code=code)


def failure_action(code: str) -> str:
    """작업 실패 코드 → retry(같은 요청 재제출) | give_up(그 씬만 포기). 크레딧 부족은 제출 단계(400)에서 걸러진다."""
    if code in RETRYABLE_FAILURES or code.startswith(RETRYABLE_PREFIXES):
        return "retry"
    return "give_up"


def http_error(r: httpx.Response) -> Exception:
    detail = ai.error_detail(r)
    try:
        reason = str(r.json().get("reason") or "")
    except Exception:
        reason = ""
    if r.status_code in (401, 403):
        return ai.VideoAIFatal(f"Runway 인증 실패 ({r.status_code}): {detail} — RUNWAY_API_KEY가 dev.runwayml.com에서 "
                               "발급한 API 키인지, 삭제·만료되지 않았는지 확인하세요.")
    if r.status_code == 400 and ("credit" in detail.lower() or reason == "QUOTA"):
        return ai.VideoAIFatal(f"Runway 크레딧 부족 또는 사용 한도 초과 (400): {detail}", kind="credits")
    return (ai.transient_error(r, "Runway", detail)
            or ai.VideoAIError(f"Runway 요청 거부 ({r.status_code}) {detail}" + (f" [{reason}]" if reason else "")))
