"""Luma (Ray) — 씬별 캐릭터 연기 영상 클립 생성 (Image-to-Video).

두 가지 API를 지원하며 키 형식으로 자동 선택한다 (config.LUMA_API_FLAVOR)
- agents (Luma API Platform 키 'luma-api-…'): https://agents.lumalabs.ai/v1, 모델 ray-3.2
    요청 {"model", "type": "video", "prompt", "aspect_ratio", "video": {"resolution", "duration", "start_frame": ImageRef}}
    ImageRef는 공개 URL {"url"} 또는 이미지 직접 첨부 {"data": base64, "media_type"} → 공개 URL 없이도 동작
    완료 시 영상 주소: output[].url (type=video, 1시간 유효한 서명 URL → 바로 다운로드)
- dream-machine (예전 키): https://api.lumalabs.ai/dream-machine/v1, 모델 ray-2
    요청 {"model", "prompt", "aspect_ratio", "resolution", "duration", "keyframes": {"frame0": {"type": "image", "url"}}}
    이미지는 공개 URL만 가능, 완료 시 영상 주소: assets.video
두 API 모두 인증은 'Authorization: Bearer <키>' 헤더. 키와 API 서버가 맞지 않으면 401/403 'Not authenticated'.

흐름
1. 씬마다 연기 계획(host_service.plan_acting: 감정 비트·포즈)을 세우고, 첫 포즈의 JB 이미지를 첫 프레임(frame0)으로,
   캐릭터 외형 + 감정 연기 + 씬 분위기(visual_prompt) + 섹션별 카메라 무빙을 묶은 영어 프롬프트를 만든다.
2. POST /generations → GET /generations/{id} 폴링 → 완성 영상 다운로드. 씬들은 asyncio로 동시에 요청한다
   (LUMA_CONCURRENCY). 결과는 media/clips/{요청 내용 해시}.mp4 로 캐시해 같은 씬을 다시 렌더링해도 재생성하지 않는다.
3. 실패 처리
   - 인증 실패(401/403)·크레딧 부족(402): 남은 씬 요청을 모두 중단 (오류 메시지에 키 형식·API 서버 안내)
   - 한도 초과(429)·일시 오류(5xx, 네트워크): Retry-After(없으면 지수 대기)만큼 기다렸다 재시도
   - 생성 실패(state=failed)·시간 초과·잘못된 요청(400): 그 씬만 포기
   포기한 씬은 렌더링에서 기존 방식(캐릭터 카드 + 켄 번즈 배경)으로 자동 대체된다.

이미지 전달: LUMA_IMAGE_BASE_URL(http/https)이 있으면 {base}/media/... 공개 URL, 없으면 agents API에 한해 JPEG로 줄여
base64로 직접 첨부한다 (dream-machine API는 공개 URL이 꼭 필요).
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx

from config import (
    API_MODE, CLIP_DIR, LUMA_API_BASE, LUMA_API_FLAVOR, LUMA_API_KEY, LUMA_CHARACTER, LUMA_CONCURRENCY, LUMA_DURATION,
    LUMA_ENABLED, LUMA_IMAGE_BASE_URL, LUMA_IMAGE_MODEL, LUMA_MODEL, LUMA_POLL_SEC, LUMA_RESOLUTION,
    LUMA_SCENE_FRAME, LUMA_TIMEOUT_SEC, MEDIA_DIR, HEROINE_CHARACTER,
)
from services import host_service

logger = logging.getLogger(__name__)

ASPECT_RATIO = "9:16"
# 생성 실패(state=failed) failure_code 분류 (docs.agents.lumalabs.ai/guides/error-handling)
RETRYABLE_FAILURES = {"generation_failed", "output_not_found", "rate_limited"}  # 같은 요청으로 다시 제출
FATAL_FAILURES = {"budget_exhausted"}                                            # 잔액 부족 → 남은 씬 중단
MAX_RESUBMITS = 2
INLINE_IMAGE_MAX_SIDE = 1280     # base64로 첨부할 포즈 이미지 최대 변 (요청 크기 줄이기)
PUBLIC_IMAGE_BASE = LUMA_IMAGE_BASE_URL if LUMA_IMAGE_BASE_URL.startswith(("http://", "https://")) else ""
MAX_RETRIES = 4
PROMPT_MAX_CHARS = 4000          # Agents API 한도 6000자 — 사용자 프롬프트를 맨 앞에 두므로 잘려도 덧붙인 지시부터 잘림
USER_PROMPT_MAX_CHARS = 2500
# 사용자 프롬프트에 이미 카메라 지시가 있으면 기본 카메라 무빙을 덧붙이지 않는다 (충돌 방지)
CAMERA_WORDS = ("camera", "shot", "dolly", "push-in", "push in", "pull-back", "pull back", "pan ", "panning", "tilt",
                "zoom", "tracking", "close-up", "close up", "closeup", "wide angle", "aerial", "handheld", "crane",
                "orbit", "static", "drone", "pov", "macro", "lens", "angle")
# '영상 프롬프트'가 비어 있을 때의 섹션별 기본 장면
DEFAULT_SCENES = {
    "hook": "An intimate, moody moment in a dim room at night, soft neon glow and gentle haze",
    "body": "A quiet cinematic moment on an urban street at dusk, soft bokeh city lights",
    "cta": "A cozy bookstore corner with warm light and books on display",
}

# 감정별 연기 지시 (host_service.EMOTIONS 키)
ACTING = {
    "greeting": "smiles warmly at the camera and gives a friendly wave hello",
    "question": "tilts his head with a curious, questioning look, eyebrows slightly raised, as if asking the viewer",
    "surprise": "gasps in genuine surprise, eyes widening, hand rising to his chest",
    "emphasis": "nods with conviction and points toward the camera with a bright, confident smile",
    "reflection": "looks away thoughtfully with his chin resting on his hand, lost in a quiet memory",
    "conclusion": "leans slightly toward the camera and points forward, warmly inviting the viewer",
}
# 섹션별 카메라 (영화적 리듬: 훅은 다가가고, 본문은 부드럽게 흐르고, 마무리는 물러나며 여운)
CAMERA = {
    "hook": "slow dramatic push-in toward his face",
    "body": "gentle handheld dolly with soft parallax",
    "cta": "slow pull-back revealing the scene",
}


class LumaError(Exception):
    pass


class LumaFatal(LumaError):
    """더 요청해도 소용없는 오류 — 남은 씬 요청을 모두 중단. kind: auth(인증) | credits(크레딧·잔액 부족)."""

    def __init__(self, message: str, kind: str = "auth"):
        super().__init__(message)
        self.kind = kind


class LumaRetryable(LumaError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class ClipJob:
    index: int                   # 씬 순서 (결과 목록 위치)
    prompt: str                  # 영상(ray) 프롬프트
    image: dict                  # 캐릭터 포즈 ImageRef: {"url": ...} 또는 {"data": base64, "media_type": ...}
    cache_path: Path
    frame_prompt: str | None = None  # 키프레임(uni-1) 프롬프트 — generate 모드에서만
    heroine_refs: tuple = ()         # 여자주인공 레퍼런스 ImageRef들 (대표 이미지 먼저) — 등장 씬에서만


@dataclass
class ClipResults:
    paths: list[Path | None]
    errors: dict[int, str] = field(default_factory=dict)  # 씬 순서 → 실패 사유
    fatal_kind: str | None = None  # 치명적 오류로 중단됐으면 auth | credits

    def reasons(self) -> str:
        return " / ".join(sorted(set(self.errors.values())))[:600]


# ---------- 사용 가능 여부 ----------
def unavailable_reason() -> str | None:
    """Luma를 쓸 수 없는 이유 (쓸 수 있으면 None)."""
    if not LUMA_ENABLED:
        return "LUMA_ENABLED=false"
    if API_MODE != "live":
        return f"API_MODE={API_MODE} — 유료 API 호출 안 함"
    if not LUMA_API_KEY:
        return "LUMA_API_KEY가 설정되지 않음"
    if LUMA_API_FLAVOR == "dream-machine" and not PUBLIC_IMAGE_BASE:
        return ("LUMA_IMAGE_BASE_URL(http/https 공개 주소)이 필요함 — 예전 Dream Machine 키는 이미지를 공개 URL로만 받음. "
                "platform.lumalabs.ai의 'luma-api-' 키를 쓰면 필요 없음")
    if host_service.mode() != "character":
        return "캐릭터 포즈 이미지(HOST_POSE_DIR)가 없음"
    return None


def enabled() -> bool:
    return unavailable_reason() is None


def status() -> dict:
    reason = unavailable_reason()
    return {"enabled": reason is None, "reason": reason, "api": LUMA_API_FLAVOR, "model": LUMA_MODEL,
            "resolution": LUMA_RESOLUTION, "duration": _duration(),
            "image": "public-url" if PUBLIC_IMAGE_BASE else "inline",
            "frame": _effective_frame_mode(), "image_model": LUMA_IMAGE_MODEL}


def frame_mode() -> str:
    """설정상 첫 프레임 방식 (generate | pose). 키 유무·API 종류와 무관 — 클립 최신 여부 판단에 쓰므로 고정값이어야 함."""
    return "pose" if LUMA_SCENE_FRAME == "pose" else "generate"


def _effective_frame_mode(heroine: bool = False) -> str:
    """실제로 쓸 방식 — 키프레임 생성은 Agents API(uni-1)에서만 가능.
    여자주인공 등장 씬은 두 사람이 함께 있는 첫 프레임이 필요하므로 pose 설정이어도 키프레임을 만든다."""
    if LUMA_API_FLAVOR != "agents":
        return "pose"
    return "generate" if heroine else frame_mode()


def _duration() -> str:
    # agents API: 첫 프레임을 지정하면 10s 불가 → 5s
    return "5s" if LUMA_API_FLAVOR == "agents" and LUMA_DURATION != "5s" else LUMA_DURATION


# ---------- 요청 만들기 ----------
def pose_image_ref(pose: str) -> dict | None:
    """포즈 이미지 ImageRef — 공개 URL이 있으면 URL, 없으면(agents API) JPEG base64 첨부."""
    path = host_service.pose_files().get(pose)
    return image_ref(path) if path is not None else None


def image_ref(path: Path) -> dict | None:
    """이미지 파일 → Luma ImageRef (공개 URL 또는 base64 JPEG)."""
    if PUBLIC_IMAGE_BASE and path.is_relative_to(MEDIA_DIR):
        return {"url": f"{PUBLIC_IMAGE_BASE}/media/{quote(path.relative_to(MEDIA_DIR).as_posix())}"}
    if LUMA_API_FLAVOR != "agents":
        return None
    return {"data": _inline_jpeg(path), "media_type": "image/jpeg"}


def _inline_jpeg(path: Path) -> str:
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((INLINE_IMAGE_MAX_SIDE, INLINE_IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def user_prompt(visual_prompt: str) -> str:
    """씬 편집 화면의 '영상 프롬프트' (공백 정리, 끝 문장부호 정리). 비어 있으면 ""."""
    text = " ".join((visual_prompt or "").split())[:USER_PROMPT_MAX_CHARS]
    return text.rstrip(" ,;.")


def _scene_text(section: str, visual_prompt: str) -> str:
    return user_prompt(visual_prompt) or DEFAULT_SCENES.get(section, DEFAULT_SCENES["body"])


def _has(text: str, words) -> bool:
    low = f" {text.lower()} "
    return any(w in low for w in words)


def _style_hints(scene: str) -> list[str]:
    """사용자 프롬프트에 없는 스타일 키워드만 덧붙인다 (중복·충돌 방지)."""
    hints = []
    if "9:16" not in scene and "vertical" not in scene.lower():
        hints.append("Vertical 9:16 framing with the character's face in the upper-middle third.")
    if not _has(scene, ("cinematic", "cinematography", "film", "movie")):
        hints.append("Cinematic lighting with natural depth of field.")
    return hints


def _character() -> str:
    return LUMA_CHARACTER[:1].upper() + LUMA_CHARACTER[1:]


def build_frame_prompt(section: str, visual_prompt: str, acting: host_service.SceneActing,
                       heroine_count: int = 0) -> str:
    """키프레임(uni-1) 프롬프트: 사용자의 장면 묘사가 먼저, 그 다음 레퍼런스 이미지 1 = 캐릭터라는 지시와 연기·스타일.
    heroine_count: 여자주인공 레퍼런스 수 (레퍼런스 이미지 2번부터) — 0이면 JB 혼자."""
    scene = _scene_text(section, visual_prompt)
    first = ACTING.get(acting.beats[0].emotion, ACTING["emphasis"])
    if heroine_count:
        last = 1 + heroine_count
        refs = "reference image 2" if heroine_count == 1 else f"reference images 2-{last}"
        parts = [
            f"{scene}.",
            f"Two characters share this scene. Male lead: the character from reference image 1 ({LUMA_CHARACTER}), "
            f"who {first}.",
            f"Female lead: the woman shown in {refs} ({HEROINE_CHARACTER}), close to him and reacting naturally "
            "with a matching emotion"
            + (". Use reference image 2 for her hairstyle and outfit; use the other references only to keep her face "
               "identical." if heroine_count > 1 else "."),
            "Keep both faces exactly consistent with their references. Draw both characters in the same illustration "
            "style as reference image 1 (do not mix photographic and illustrated styles), and do not copy the "
            "reference backgrounds, decorations or watermarks.",
            *_style_hints(scene),
            "No text, captions or logos.",
        ]
        return " ".join(parts)[:PROMPT_MAX_CHARS]
    parts = [
        f"{scene}.",
        f"Main subject: the character from reference image 1 ({LUMA_CHARACTER}), who {first}, placed naturally "
        "inside this scene.",
        "Keep exactly the same face, hairstyle, outfit and illustration style as reference image 1, but do not "
        "copy its background, pose framing or decorations.",
        *_style_hints(scene),
        "No text, captions or logos.",
    ]
    return " ".join(parts)[:PROMPT_MAX_CHARS]


def build_prompt(section: str, visual_prompt: str, acting: host_service.SceneActing, heroine: bool = False) -> str:
    """영상(ray) 프롬프트: 사용자의 '영상 프롬프트'를 맨 앞에 그대로 두고, 캐릭터 연기(감정 비트) → (프롬프트에 카메라
    지시가 없을 때만) 섹션별 카메라 → 빠진 스타일 키워드 → 캐릭터 일관성·텍스트 금지 순으로 덧붙인다."""
    scene = _scene_text(section, visual_prompt)
    actions = [ACTING.get(b.emotion, ACTING["emphasis"]) for b in acting.beats]
    action = actions[0] + (f", then {actions[1]}" if len(actions) > 1 else "")
    who = (f"{_character()} and the woman beside him in the first frame: he {action}, while she reacts "
           "naturally with subtle, matching emotion." if heroine else f"{_character()} in the first frame {action}.")
    parts = [
        f"{scene}.",
        who,
        "" if _has(scene, CAMERA_WORDS) else f"Camera: {CAMERA.get(section, CAMERA['body'])}.",
        *_style_hints(scene),
        "Natural, expressive facial acting and subtle body motion; keep "
        + ("both characters' faces, hairstyles and outfits" if heroine else "the same face, hairstyle and outfit")
        + " as the first frame throughout. No text, no captions, no logos.",
    ]
    return " ".join(p for p in parts if p)[:PROMPT_MAX_CHARS]


def prompt_preview(section: str, visual_prompt: str, acting: host_service.SceneActing,
                   heroine_refs: tuple = ()) -> dict:
    """대시보드 표시용: 실제로 보낼 프롬프트. heroine_refs: 여자주인공 레퍼런스 이미지 경로들."""
    heroine = bool(heroine_refs) and LUMA_API_FLAVOR == "agents"
    mode = _effective_frame_mode(heroine)
    return {"mode": mode, "user_prompt": user_prompt(visual_prompt),
            "video": build_prompt(section, visual_prompt, acting, heroine),
            "frame": build_frame_prompt(section, visual_prompt, acting, len(heroine_refs) if heroine else 0)
            if mode == "generate" else None,
            "heroine_refs": [p.name for p in heroine_refs] if heroine else [],
            "heroine_unsupported": bool(heroine_refs) and not heroine}


def _frame_request_body(job: ClipJob) -> dict:
    """키프레임 이미지 요청 (uni-1): 장면 프롬프트 + 캐릭터 포즈 이미지를 레퍼런스(image_ref)로."""
    return {
        "model": LUMA_IMAGE_MODEL,
        "type": "image",
        "prompt": job.frame_prompt,
        "aspect_ratio": ASPECT_RATIO,
        "image_ref": [job.image, *job.heroine_refs],  # 1 = JB 포즈, 2~ = 여자주인공 (프롬프트의 번호와 일치)
    }


def _request_body(job: ClipJob, start_frame: dict | None = None) -> dict:
    """영상 요청. start_frame: 생성한 키프레임 (없으면 캐릭터 포즈 이미지)."""
    if LUMA_API_FLAVOR == "agents":
        return {
            "model": LUMA_MODEL,
            "type": "video",
            "prompt": job.prompt,
            "aspect_ratio": ASPECT_RATIO,
            "video": {"resolution": LUMA_RESOLUTION, "duration": _duration(), "start_frame": start_frame or job.image},
        }
    return {
        "model": LUMA_MODEL,
        "prompt": job.prompt,
        "aspect_ratio": ASPECT_RATIO,
        "resolution": LUMA_RESOLUTION,
        "duration": LUMA_DURATION,
        "loop": False,
        "keyframes": {"frame0": {"type": "image", "url": job.image["url"]}},
    }


def _video_url(generation: dict) -> str | None:
    """완료된 generation의 영상 주소 (agents: output[].url / dream-machine: assets.video)."""
    for item in generation.get("output") or []:
        if isinstance(item, dict) and item.get("url") and item.get("type", "video") == "video":
            return item["url"]
    return (generation.get("assets") or {}).get("video")


def _pose_signature(acting: host_service.SceneActing) -> list:
    pose_file = host_service.pose_files()[acting.pose]
    stat = pose_file.stat()
    return [pose_file.name, stat.st_size, stat.st_mtime_ns]


def content_key(section: str, visual_prompt: str, acting: host_service.SceneActing, mode: str | None = None,
                heroine_refs: tuple = ()) -> str:
    """씬 클립 내용 키 — 클립을 결정하는 '입력'(영상 프롬프트, 감정 비트·포즈, 캐릭터 설명, 첫 프레임 방식)의 해시.
    API 종류·모델·키 유무나 프롬프트 조립 문구 수정과는 무관 → 설정을 바꾸거나 키를 비워도 만든 클립은 최신으로 인정.
    영상 프롬프트를 고치면 키가 바뀌어 '다시 생성 필요'가 된다."""
    from services import heroine_service

    beats = [(b.emotion, b.pose, round(b.at, 3)) for b in acting.beats]
    parts = [mode or frame_mode(), section, user_prompt(visual_prompt), LUMA_CHARACTER, beats, *_pose_signature(acting)]
    if heroine_refs:  # 여자주인공이 없는 씬은 예전과 같은 키 (이미 만든 클립 유지)
        parts.append(["heroine", HEROINE_CHARACTER, heroine_service.signature(list(heroine_refs))])
    raw = json.dumps(parts, ensure_ascii=False)
    return "luma_k_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def job_key(section: str, visual_prompt: str, acting: host_service.SceneActing, heroine_refs: tuple = ()) -> str:
    """캐시 파일 이름 — 내용 + 생성 설정(API·모델·해상도·길이·첫 프레임 방식). 같은 설정으로 같은 씬을 다시 요청하면
    API를 부르지 않는다."""
    from services import heroine_service

    heroine = bool(heroine_refs) and LUMA_API_FLAVOR == "agents"
    mode = _effective_frame_mode(heroine)
    parts = [LUMA_API_FLAVOR, LUMA_MODEL, LUMA_RESOLUTION, _duration(), ASPECT_RATIO, mode,
             LUMA_IMAGE_MODEL if mode == "generate" else "",
             build_prompt(section, visual_prompt, acting, heroine),
             build_frame_prompt(section, visual_prompt, acting, len(heroine_refs) if heroine else 0)
             if mode == "generate" else "",
             *_pose_signature(acting)]
    if heroine:
        parts.append(heroine_service.signature(list(heroine_refs)))
    raw = json.dumps(parts, ensure_ascii=False)
    return "luma_v2_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def plan_jobs(scenes: list[tuple], acting: list[host_service.SceneActing | None]) -> list[ClipJob | None]:
    """[(section, visual_prompt[, 여자주인공 레퍼런스 경로들])] + 연기 계획 → 씬별 생성 작업 (만들 수 없는 씬은 None)."""
    jobs: list[ClipJob | None] = []
    for index, (scene, act) in enumerate(zip(scenes, acting)):
        section, visual_prompt = scene[0], scene[1]
        refs = tuple(scene[2]) if len(scene) > 2 and LUMA_API_FLAVOR == "agents" else ()
        image = pose_image_ref(act.pose) if act else None
        if act is None or image is None:
            jobs.append(None)
            continue
        heroine_refs = tuple(r for r in (image_ref(p) for p in refs) if r)
        key = job_key(section, visual_prompt, act, refs)
        frame_prompt = build_frame_prompt(section, visual_prompt, act, len(heroine_refs)) \
            if _effective_frame_mode(bool(heroine_refs)) == "generate" else None
        jobs.append(ClipJob(index, build_prompt(section, visual_prompt, act, bool(heroine_refs)), image,
                            CLIP_DIR / f"{key}.mp4", frame_prompt, heroine_refs))
    return jobs


# ---------- 생성 ----------
def generate_clips(jobs: list[ClipJob | None], on_update=None) -> ClipResults:
    """씬별 클립 생성 (동기 진입점 — 백그라운드 작업 스레드에서 호출). 캐시가 있으면 API를 부르지 않는다.
    on_update(index, state, path, error): 씬 진행 상황 알림 — state는 generating | completed | failed."""
    notify = on_update or (lambda *args: None)
    results = ClipResults(paths=[None] * len(jobs))
    pending = []
    for job in jobs:
        if job is None:
            continue
        if job.cache_path.exists() and job.cache_path.stat().st_size > 0:
            results.paths[job.index] = job.cache_path
            notify(job.index, "completed", job.cache_path, None)
        else:
            pending.append(job)
    if pending:
        logger.info("Luma 클립 생성 시작: %d개 (캐시 %d개)", len(pending),
                    sum(p is not None for p in results.paths))
        asyncio.run(_generate_all(pending, results, notify))
    return results


async def _generate_all(jobs: list[ClipJob], results: ClipResults, notify) -> None:
    limit = asyncio.Semaphore(max(LUMA_CONCURRENCY, 1))
    stop = asyncio.Event()  # 치명적 오류가 나면 아직 시작하지 않은 요청은 건너뜀
    headers = {"Authorization": f"Bearer {LUMA_API_KEY}", "Accept": "application/json",
               "Content-Type": "application/json"}
    async with httpx.AsyncClient(base_url=LUMA_API_BASE, headers=headers,
                                 timeout=httpx.Timeout(60, connect=15)) as client:
        async def run(job: ClipJob) -> None:
            async with limit:
                if stop.is_set():
                    results.errors[job.index] = "앞선 치명적 오류로 요청 생략"
                    notify(job.index, "failed", None, results.errors[job.index])
                    return
                notify(job.index, "generating", None, None)
                try:
                    await _generate_one(client, job)
                    results.paths[job.index] = job.cache_path
                    notify(job.index, "completed", job.cache_path, None)
                    return
                except LumaFatal as e:
                    stop.set()
                    results.fatal_kind = results.fatal_kind or e.kind
                    results.errors[job.index] = str(e)
                    logger.error("Luma 치명적 오류 — 남은 씬 생성 중단: %s", e)
                except Exception as e:
                    results.errors[job.index] = str(e) or type(e).__name__
                    logger.warning("Luma 씬 %d 생성 실패: %s", job.index, e)
                notify(job.index, "failed", None, results.errors[job.index])

        await asyncio.gather(*(run(job) for job in jobs))


async def _generate_one(client: httpx.AsyncClient, job: ClipJob) -> None:
    """(generate 모드) 장면 키프레임 생성 → 영상 생성 → 다운로드.
    키프레임이 실패하면(치명적 오류 제외) 캐릭터 포즈 이미지를 첫 프레임으로 영상만 만든다."""
    start_frame = None
    if job.frame_prompt:
        try:
            start_frame = await _generate_keyframe(client, job)
        except LumaFatal:
            raise
        except Exception as e:
            logger.warning("Luma 씬 %d 키프레임 생성 실패 → 포즈 이미지를 첫 프레임으로 사용: %s", job.index, e)
    gen_id, generation = await _run_generation(client, _request_body(job, start_frame), f"씬 {job.index} 영상")
    await _download_with_refresh(client, gen_id, generation, job.cache_path)
    logger.info("Luma 씬 %d 완료 → %s", job.index, job.cache_path.name)


async def _generate_keyframe(client: httpx.AsyncClient, job: ClipJob) -> dict:
    """장면 키프레임(uni-1, 캐릭터 레퍼런스) 생성 → 받아서 저장(대시보드 미리보기용) → 영상 첫 프레임용 ImageRef.
    서명 URL을 그대로 넘기지 않고 내려받아 base64로 첨부 (URL 가져오기 장애·만료 영향 없음)."""
    gen_id, generation = await _run_generation(client, _frame_request_body(job), f"씬 {job.index} 키프레임")
    url = next((o.get("url") for o in generation.get("output") or [] if isinstance(o, dict) and o.get("url")), None)
    if not url:
        raise LumaError(f"키프레임 완료됐지만 이미지 URL이 없음 (id={gen_id})")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15), follow_redirects=True) as cdn:
        r = await cdn.get(url)
        r.raise_for_status()
    media_type = r.headers.get("content-type", "").split(";")[0] or "image/jpeg"
    ext = {"image/png": ".png", "image/webp": ".webp"}.get(media_type, ".jpg")
    frame_path = job.cache_path.with_name(job.cache_path.stem + "_frame" + ext)
    frame_path.write_bytes(r.content)
    logger.info("Luma 씬 %d 키프레임 → %s", job.index, frame_path.name)
    return {"data": base64.b64encode(r.content).decode("ascii"), "media_type": media_type}


async def _run_generation(client: httpx.AsyncClient, body: dict, label: str) -> tuple[str, dict]:
    """제출 → 폴링(queued/processing → completed|failed). 완료된 (id, generation)을 반환.
    다시 시도해도 되는 생성 실패(failure_code)는 같은 요청을 MAX_RESUBMITS번까지 다시 제출한다."""
    for submit in range(MAX_RESUBMITS + 1):
        generation = await _with_retries(lambda: _call(client, "POST", "/generations", json=body))
        gen_id = generation.get("id")
        if not gen_id:
            raise LumaError("Luma 응답에 generation id가 없음")
        logger.info("Luma %s 요청됨 (id=%s%s)", label, gen_id, f", 재제출 {submit}" if submit else "")
        generation = await _poll(client, gen_id)
        if generation.get("state") == "completed":
            return gen_id, generation
        code = generation.get("failure_code") or ""
        message = f"{label} 생성 실패: {generation.get('failure_reason') or '사유 없음'}" + (f" ({code})" if code else "")
        if code in FATAL_FAILURES:
            raise LumaFatal(f"{message} — Luma 계정 잔액을 확인하세요.", kind="credits")
        if code not in RETRYABLE_FAILURES or submit == MAX_RESUBMITS:
            raise LumaError(message)
        logger.warning("Luma %s — 다시 제출합니다.", message)
        await asyncio.sleep(5 * (submit + 1))
    raise LumaError(f"{label} 생성 실패")


async def _poll(client: httpx.AsyncClient, gen_id: str) -> dict:
    """완료(completed) 또는 실패(failed)가 될 때까지 LUMA_POLL_SEC 간격으로 조회 (문서 권장 2~5초).
    조회는 생성 RPM 한도에 포함되지 않는다."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + LUMA_TIMEOUT_SEC
    state = None
    while loop.time() <= deadline:
        await asyncio.sleep(LUMA_POLL_SEC)
        generation = await _with_retries(lambda: _call(client, "GET", f"/generations/{gen_id}"))
        state = generation.get("state")
        if state in ("completed", "failed"):
            return generation
    raise LumaError(f"생성 시간 초과 ({LUMA_TIMEOUT_SEC}초, 마지막 상태 {state}, id={gen_id})")


async def _download_with_refresh(client: httpx.AsyncClient, gen_id: str, generation: dict, dest: Path) -> None:
    """완성 영상 URL은 1시간 유효한 서명 URL — 받기에 실패하면 generation을 다시 조회해 새 URL로 재시도."""
    for attempt in range(2):
        video_url = _video_url(generation)
        if not video_url:
            raise LumaError(f"완료됐지만 영상 URL이 없음 (id={gen_id})")
        try:
            await _download(video_url, dest)
            return
        except LumaError:
            if attempt == 1:
                raise
            logger.warning("Luma 영상 다운로드 실패 — 새 URL을 받아 다시 시도 (id=%s)", gen_id)
            generation = await _with_retries(lambda: _call(client, "GET", f"/generations/{gen_id}"))


async def _with_retries(make_call):
    """한도 초과·일시 오류는 Retry-After(없으면 지수 대기 + 지터)만큼 기다렸다 재시도."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await make_call()
        except LumaRetryable as e:
            if attempt == MAX_RETRIES:
                raise LumaError(f"재시도 {MAX_RETRIES}회 후에도 실패: {e}") from e
            wait = e.retry_after if e.retry_after is not None else min(2 ** attempt * 3, 60)
            wait += random.uniform(0, 1.5)
            logger.warning("Luma 일시 오류 (%s) — %.1f초 후 재시도", e, wait)
            await asyncio.sleep(wait)


async def _call(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> dict:
    try:
        r = await client.request(method, url, **kwargs)
    except httpx.TransportError as e:
        raise LumaRetryable(f"네트워크 오류: {type(e).__name__}") from e
    if r.status_code < 300:
        return r.json()
    detail = _error_detail(r)
    request_id = r.headers.get("x-request-id")
    if request_id:  # Luma 지원팀 문의용
        detail += f" [request-id {request_id}]"
    if r.status_code in (401, 403):
        raise LumaFatal(f"Luma 인증 실패 ({r.status_code}): {detail} — {_auth_hint()}")
    if r.status_code == 402:
        raise LumaFatal(f"Luma 크레딧(잔액) 부족 (402). {detail}", kind="credits")
    if r.status_code == 429 or r.status_code >= 500:
        retry_after = r.headers.get("retry-after")
        try:
            seconds = float(retry_after) if retry_after else None
        except ValueError:
            seconds = None
        raise LumaRetryable(f"{'한도 초과' if r.status_code == 429 else '서버 오류'} ({r.status_code}) {detail}",
                            seconds)
    raise LumaError(f"Luma 요청 거부 ({r.status_code}) {detail}")


def _auth_hint() -> str:
    """인증 실패 원인 안내: 키 형식과 실제로 호출한 API 서버를 함께 보여준다 (키 값은 노출하지 않음)."""
    kind = "Luma API Platform 키(luma-api-…)" if LUMA_API_FLAVOR == "agents" else "예전 Dream Machine 키 형식"
    return (f"{kind}로 {LUMA_API_BASE} 를 호출했습니다. 키가 만료·삭제되지 않았는지, 이 서버용 키가 맞는지 확인하세요 "
            "(platform.lumalabs.ai에서 발급한 'luma-api-' 키는 agents.lumalabs.ai, 예전 키는 api.lumalabs.ai/dream-machine 전용"
            ", LUMA_API_BASE를 직접 지정했다면 비워 두세요).")


def _error_detail(r: httpx.Response) -> str:
    try:
        body = r.json()
        return str(body.get("detail") or body.get("error") or body)[:300]
    except Exception:
        return r.text[:300]


async def _download(url: str, dest: Path) -> None:
    """생성된 영상 다운로드 (CDN 주소 — 인증 헤더 없이). 임시 파일에 받은 뒤 이름을 바꿔 반쯤 받은 파일이 캐시로 남지 않게."""
    tmp = dest.with_suffix(".part")
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15), follow_redirects=True) as cdn:
                async with cdn.stream("GET", url) as r:
                    r.raise_for_status()
                    with tmp.open("wb") as f:
                        async for chunk in r.aiter_bytes(1 << 16):
                            f.write(chunk)
            if tmp.stat().st_size == 0:
                raise LumaError("빈 영상 파일")
            tmp.replace(dest)
            return
        except (httpx.HTTPError, LumaError) as e:
            tmp.unlink(missing_ok=True)
            expired = isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500
            if expired or attempt == 2:  # 4xx(만료된 서명 URL 등)는 같은 주소로 다시 받아도 소용없음 → 새 URL로
                raise LumaError(f"영상 다운로드 실패: {e}") from e
            await asyncio.sleep(2 ** attempt)
