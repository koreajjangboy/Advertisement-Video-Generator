"""AI 씬 영상 — 씬별 캐릭터 연기 영상 클립 생성 (Image-to-Video). 업체와 무관한 공통 흐름.

업체는 .env의 VIDEO_PROVIDER로 고른다 (luma | runway). 업체별 요청 형식·오류 해석은 각 어댑터 모듈이 맡는다.
- services/luma_service.py   : Luma (Ray / uni-1)
- services/runway_service.py : Runway (Gen-4.5 / gen4_image)

흐름
1. 씬 분위기(visual_prompt) + 캐릭터 외형 + 섹션별 카메라 무빙을 묶은 영어 프롬프트를 만든다.
   씬 포즈·감정 연기는 넣지 않는다 — 표정·동작은 영상 프롬프트를 따른다. 이미지 모델이 고정 레퍼런스
   (VIDEO_REF_POSE의 JB 이미지, 얼굴·옷 일관성용)로 '그 장면 속의 JB' 키프레임을 먼저 만들고 그 키프레임에서 영상을 시작
   (키프레임을 못 만드는 예전 Luma API만 레퍼런스 이미지를 그대로 첫 프레임으로 씀)
2. 작업 제출 → 상태 폴링 → 완성 영상 다운로드. 씬들은 asyncio로 동시에 요청한다 (업체별 CONCURRENCY).
   결과는 media/clips/{요청 내용 해시}.mp4 로 캐시해 같은 씬을 다시 렌더링해도 재생성하지 않는다.
3. 실패 처리
   - 인증 실패·크레딧 부족: 남은 씬 요청을 모두 중단
   - 한도 초과(429)·일시 오류(5xx, 네트워크): Retry-After(없으면 지수 대기)만큼 기다렸다 재시도
   - 생성 실패·시간 초과·잘못된 요청(400): 그 씬만 포기 (다시 시도해도 되는 실패 코드는 같은 요청을 재제출)
   포기한 씬은 대체 클립으로 자동 대체된다 (pipeline.generate_clips_for_episode).

업체 어댑터가 제공하는 것 (모듈 수준)
  NAME, LABEL, API_BASE, MODEL, IMAGE_MODEL, CONCURRENCY, POLL_SEC, TIMEOUT_SEC, CACHE_PREFIX,
  PROMPT_MAX_CHARS, USER_PROMPT_MAX_CHARS, MAX_HEROINE_REFS, DESCRIBE_REFS
  config_problem() · frame_mode() · frame_supported() · heroine_supported() · status_extra() · job_settings(mode)
  image_ref(path) · keyframe_ref(data, media_type) · ref_names(heroine_count) · headers()
  frame_request(job) · video_request(job, start_frame) · task_path(task_id) · task_id(response) · parse_task(task)
  failure_action(code) · http_error(response)
"""

import asyncio
import hashlib
import importlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import httpx

from config import API_MODE, CLIP_DIR, HEROINE_CHARACTER, VIDEO_CHARACTER, VIDEO_PROVIDER, VIDEO_REF_POSE
from services import host_service

logger = logging.getLogger(__name__)

PROVIDERS = ("luma", "runway")   # Script.clip_source 값이기도 함 (AI로 만든 클립)
ASPECT_RATIO = "9:16"
MAX_RESUBMITS = 2
MAX_RETRIES = 4
INLINE_IMAGE_MAX_SIDE = 1280     # 요청에 직접 담는 포즈 이미지 최대 변 (요청 크기 줄이기)
CONTENT_PROMPT_MAX_CHARS = 2500  # 내용 키 계산용 사용자 프롬프트 길이 — 바꾸면 기존 클립이 모두 '다시 생성 필요'가 됨
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

# 표정·동작 — 씬 포즈·감정 대신 영상 프롬프트의 분위기를 따르게 하는 공통 지시
MOOD_ACTING = ("his facial expression, gaze, gestures and body language follow the mood and action described "
               "in this scene")
# 섹션별 카메라 (영화적 리듬: 훅은 다가가고, 본문은 부드럽게 흐르고, 마무리는 물러나며 여운)
CAMERA = {
    "hook": "slow dramatic push-in toward his face",
    "body": "gentle handheld dolly with soft parallax",
    "cta": "slow pull-back revealing the scene",
}


class VideoAIError(Exception):
    pass


class VideoAIFatal(VideoAIError):
    """더 요청해도 소용없는 오류 — 남은 씬 요청을 모두 중단. kind: auth(인증) | credits(크레딧·잔액·사용 한도)."""

    def __init__(self, message: str, kind: str = "auth"):
        super().__init__(message)
        self.kind = kind


class VideoAIRetryable(VideoAIError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class ClipJob:
    index: int                   # 씬 순서 (결과 목록 위치)
    prompt: str                  # 영상 프롬프트
    image: object                # 캐릭터 포즈 이미지 참조 (형식은 업체별 — image_ref)
    cache_path: Path
    frame_prompt: str | None = None  # 키프레임 프롬프트 — generate 방식에서만
    heroine_refs: tuple = ()         # 여자주인공 레퍼런스 이미지 참조들 (대표 이미지 먼저) — 등장 씬에서만


@dataclass
class ClipResults:
    paths: list[Path | None]
    errors: dict[int, str] = field(default_factory=dict)  # 씬 순서 → 실패 사유
    fatal_kind: str | None = None  # 치명적 오류로 중단됐으면 auth | credits

    def reasons(self) -> str:
        return " / ".join(sorted(set(self.errors.values())))[:600]


@dataclass(frozen=True)
class TaskState:
    """업체별 작업 응답을 공통 형태로 — state: running | completed | failed."""
    state: str
    urls: tuple = ()            # 완료 시 결과 파일 주소 (영상 또는 키프레임 이미지)
    failure: str = ""           # 실패 사유 (사람이 읽는 문장)
    code: str = ""              # 실패 코드 (failure_action으로 재시도·중단 판단)


# ---------- 업체 선택 ----------
def provider() -> ModuleType:
    """VIDEO_PROVIDER에 맞는 업체 어댑터 모듈."""
    return importlib.import_module(f"services.{VIDEO_PROVIDER}_service")


def label() -> str:
    return provider().LABEL


def is_ai_source(source: str | None) -> bool:
    """clip_source가 AI 업체로 만든(비용을 들인) 클립인지."""
    return source in PROVIDERS


# ---------- 사용 가능 여부 ----------
def unavailable_reason() -> str | None:
    """AI 씬 영상을 쓸 수 없는 이유 (쓸 수 있으면 None)."""
    p = provider()
    if API_MODE != "live":
        return f"API_MODE={API_MODE} — 유료 API 호출 안 함"
    problem = p.config_problem()
    if problem:
        return problem
    if host_service.mode() != "character":
        return "캐릭터 포즈 이미지(HOST_POSE_DIR)가 없음"
    return None


def enabled() -> bool:
    return unavailable_reason() is None


def status() -> dict:
    p = provider()
    reason = unavailable_reason()
    return {"enabled": reason is None, "reason": reason, "provider": p.NAME, "label": p.LABEL, "model": p.MODEL,
            "frame": _effective_frame_mode(), "image_model": p.IMAGE_MODEL, **p.status_extra()}


def _effective_frame_mode(heroine: bool = False) -> str:
    """실제로 쓸 첫 프레임 방식. 포즈 이미지가 첫 프레임이 되면 포즈의 표정·동작·배경이 영상을 지배하므로
    키프레임을 만들 수 있으면 항상 generate (*_SCENE_FRAME 설정 무시). 키프레임 생성을 못 하는 API만 pose."""
    return "generate" if provider().frame_supported() else "pose"


def reference_pose_file() -> Path | None:
    """AI 영상의 캐릭터 레퍼런스 이미지 — 씬 포즈와 무관한 고정 이미지 (VIDEO_REF_POSE, 없으면 다른 포즈)."""
    files = host_service.pose_files()
    if not files:
        return None
    return files.get(VIDEO_REF_POSE) or files.get("thinking") or files[sorted(files)[0]]


# ---------- 프롬프트 ----------
def user_prompt(visual_prompt: str, limit: int = CONTENT_PROMPT_MAX_CHARS) -> str:
    """씬 편집 화면의 '영상 프롬프트' (공백 정리, 끝 문장부호 정리). 비어 있으면 ""."""
    text = " ".join((visual_prompt or "").split())[:limit]
    return text.rstrip(" ,;.")


def scene_prompt_budget(heroine: bool = False) -> int:
    """1단계 '영상 프롬프트' 권장 길이 — 키프레임·영상 지시를 덧붙여도 _fit이 장면 묘사를
    자르지 않는 길이 (프롬프트에 카메라·9:16·cinematic이 들어 있다고 가정, 최대 1200자)."""
    p = provider()
    act = host_service.SceneActing((host_service.Beat(0, "emphasis", ""),), False)
    scene = "Vertical 9:16 cinematic shot, slow push-in"
    refs = p.MAX_HEROINE_REFS if heroine else 0
    tail = max(_len16(build_prompt("hook", scene, act, heroine)), _len16(build_frame_prompt("hook", scene, act, refs)))
    return max(250, min(p.USER_PROMPT_MAX_CHARS, p.PROMPT_MAX_CHARS - (tail - _len16(scene)), 1200))


def _scene_text(section: str, visual_prompt: str) -> str:
    return (user_prompt(visual_prompt, provider().USER_PROMPT_MAX_CHARS)
            or DEFAULT_SCENES.get(section, DEFAULT_SCENES["body"]))


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
    return VIDEO_CHARACTER[:1].upper() + VIDEO_CHARACTER[1:]


def _len16(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _truncate(text: str, limit: int) -> str:
    """UTF-16 코드 단위 기준으로 자른다 (Runway 한도 기준 — 이모지 등은 2단위)."""
    if _len16(text) <= limit:
        return text
    return text.encode("utf-16-le")[:max(limit, 0) * 2].decode("utf-16-le", errors="ignore")


def _fit(scene: str, rest: list[str]) -> str:
    """'장면. 덧붙일 지시들'을 업체 프롬프트 한도에 맞춘다 — 넘치면 장면 묘사를 먼저 줄여 연기·일관성 지시를 살린다
    (지시만으로도 넘치면 뒤부터 잘림)."""
    limit = provider().PROMPT_MAX_CHARS
    tail = " ".join(p for p in rest if p)
    room = limit - _len16(tail) - 2  # ". "
    if _len16(scene) > room:
        scene = _truncate(scene, max(room, 80)).rstrip(" ,;.")
    return _truncate(f"{scene}. {tail}", limit)


def build_frame_prompt(section: str, visual_prompt: str, acting: host_service.SceneActing | None = None,
                       heroine_count: int = 0) -> str:
    """키프레임 프롬프트: 사용자의 장면 묘사가 먼저, 그 다음 레퍼런스 이미지(캐릭터 외형) 지시와 스타일.
    씬 포즈·감정 연기는 넣지 않는다 (표정·동작은 장면 묘사를 따름). acting은 호출 호환용으로만 받는다.
    heroine_count: 여자주인공 레퍼런스 수 — 0이면 JB 혼자. 레퍼런스를 부르는 이름은 업체별 (ref_names)."""
    p = provider()
    lead, heroine_main, heroine_all = p.ref_names(heroine_count)
    # 외형 설명 — 프롬프트가 짧은 업체(DESCRIBE_REFS=False)는 레퍼런스 이미지로 충분하므로 생략
    him, her = (f" ({VIDEO_CHARACTER})", f" ({HEROINE_CHARACTER})") if p.DESCRIBE_REFS else ("", "")
    scene = _scene_text(section, visual_prompt)
    no_copy = ("do not copy the reference images' pose, hand gesture, facial expression, background, "
               "decorations or watermarks")
    if heroine_count:
        return _fit(scene, [
            f"Two characters share this scene. Male lead: the character from {lead}{him}; {MOOD_ACTING}.",
            f"Female lead: the woman shown in {heroine_all}{her}, close to him, her expression also matching the scene"
            + (f". Use {heroine_main} for her hairstyle and outfit; use the other references only to keep her face "
               "identical." if heroine_count > 1 else "."),
            "Keep both faces exactly consistent with their references. Draw both characters in the same illustration "
            f"style as {lead} (do not mix photographic and illustrated styles), but {no_copy}.",
            *_style_hints(scene),
            "No text, captions or logos.",
        ])
    return _fit(scene, [
        f"Main subject: the character from {lead}{him}, placed naturally inside this scene; {MOOD_ACTING}.",
        f"Use {lead} only for his face, hairstyle, outfit and illustration style; {no_copy}.",
        *_style_hints(scene),
        "No text, captions or logos.",
    ])


def build_prompt(section: str, visual_prompt: str, acting: host_service.SceneActing | None = None,
                 heroine: bool = False) -> str:
    """영상 프롬프트: 사용자의 '영상 프롬프트'를 맨 앞에 그대로 두고, 캐릭터(표정·동작은 장면 분위기를 따름) →
    (프롬프트에 카메라 지시가 없을 때만) 섹션별 카메라 → 빠진 스타일 키워드 → 캐릭터 일관성·텍스트 금지 순으로 덧붙인다.
    씬 포즈·감정 연기는 넣지 않는다. 길이 한도를 넘으면 장면 묘사부터 줄인다 (_fit)."""
    scene = _scene_text(section, visual_prompt)
    who = (f"{_character()} and the woman beside him in the first frame; both act naturally, with expressions "
           "and movements that match the mood of this scene." if heroine
           else f"{_character()} in the first frame; {MOOD_ACTING}.")
    return _fit(scene, [
        who,
        "" if _has(scene, CAMERA_WORDS) else f"Camera: {CAMERA.get(section, CAMERA['body'])}.",
        *_style_hints(scene),
        "Subtle, natural facial acting and body motion; keep "
        + ("both characters' faces, hairstyles and outfits" if heroine else "the same face, hairstyle and outfit")
        + " as the first frame throughout. No text, no captions, no logos.",
    ])


def _heroine_paths(heroine_refs: tuple) -> tuple:
    """이 업체로 보낼 여자주인공 레퍼런스 경로 (지원하지 않으면 (), 업체 한도만큼만)."""
    p = provider()
    if not heroine_refs or not p.heroine_supported():
        return ()
    return tuple(heroine_refs)[:p.MAX_HEROINE_REFS]


def prompt_preview(section: str, visual_prompt: str, acting: host_service.SceneActing,
                   heroine_refs: tuple = ()) -> dict:
    """대시보드 표시용: 실제로 보낼 프롬프트. heroine_refs: 여자주인공 레퍼런스 이미지 경로들."""
    refs = _heroine_paths(heroine_refs)
    mode = _effective_frame_mode(bool(refs))
    mine = user_prompt(visual_prompt)
    video = build_prompt(section, visual_prompt, acting, bool(refs))
    frame = build_frame_prompt(section, visual_prompt, acting, len(refs)) if mode == "generate" else None
    return {"provider": label(), "mode": mode, "user_prompt": mine,
            # 업체 프롬프트 길이 한도 때문에 사용자의 영상 프롬프트가 잘려서 들어가는지
            "user_prompt_cut": bool(mine) and not all(t.startswith(mine) for t in (video, frame) if t),
            "video": video, "frame": frame,
            "heroine_refs": [r.name for r in refs],
            "heroine_unsupported": bool(heroine_refs) and not refs}


# ---------- 캐시 키 ----------
def pose_signature(acting: host_service.SceneActing) -> list:
    """씬 포즈 이미지 서명 — 예전(포즈 연기) 방식 클립 식별용 (clip_service.adopt_legacy_clips)."""
    return _file_signature(host_service.pose_files()[acting.pose])


def _file_signature(path: Path) -> list:
    stat = path.stat()
    return [path.name, stat.st_size, stat.st_mtime_ns]


def reference_signature() -> list:
    path = reference_pose_file()
    return _file_signature(path) if path else []


def content_key(section: str, visual_prompt: str, acting: host_service.SceneActing, mode: str | None = None,
                heroine_refs: tuple = ()) -> str:
    """씬 클립 내용 키 — 클립을 결정하는 '입력'(영상 프롬프트, 캐릭터 설명·레퍼런스, 첫 프레임 방식)의 해시.
    업체·모델·키 유무나 프롬프트 조립 문구 수정과는 무관 → 업체를 바꾸거나 키를 비워도 만든 클립은 최신으로 인정.
    영상 프롬프트를 고치면 키가 바뀌어 '다시 생성 필요'가 된다. 씬 포즈·감정은 AI 영상에 쓰지 않으므로 키에도 없다.
    mode="pose"는 예전(포즈 연기) 방식 클립 등록용 — 감정 비트·씬 포즈가 들어간 예전 키 형식.
    접두어 luma_k_는 Luma만 쓰던 시기의 이름 — 기존 클립 키와 맞추기 위해 그대로 둔다."""
    from services import heroine_service

    if mode == "pose":
        beats = [(b.emotion, b.pose, round(b.at, 3)) for b in acting.beats]
        parts = [mode, section, user_prompt(visual_prompt), VIDEO_CHARACTER, beats, *pose_signature(acting)]
    else:
        parts = ["mood", mode or _effective_frame_mode(), section, user_prompt(visual_prompt), VIDEO_CHARACTER,
                 *reference_signature()]
    if heroine_refs:
        parts.append(["heroine", HEROINE_CHARACTER, heroine_service.signature(list(heroine_refs))])
    raw = json.dumps(parts, ensure_ascii=False)
    return "luma_k_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def job_key(section: str, visual_prompt: str, acting: host_service.SceneActing, heroine_refs: tuple = ()) -> str:
    """캐시 파일 이름 — 내용 + 생성 설정(업체·모델·해상도·길이·첫 프레임 방식). 같은 설정으로 같은 씬을 다시 요청하면
    API를 부르지 않는다."""
    from services import heroine_service

    p = provider()
    refs = _heroine_paths(heroine_refs)
    mode = _effective_frame_mode(bool(refs))
    parts = [*p.job_settings(mode),
             build_prompt(section, visual_prompt, acting, bool(refs)),
             build_frame_prompt(section, visual_prompt, acting, len(refs)) if mode == "generate" else "",
             *reference_signature()]
    if refs:
        parts.append(heroine_service.signature(list(refs)))
    raw = json.dumps(parts, ensure_ascii=False)
    return p.CACHE_PREFIX + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def plan_jobs(scenes: list[tuple], acting: list[host_service.SceneActing | None]) -> list[ClipJob | None]:
    """[(section, visual_prompt[, 여자주인공 레퍼런스 경로들])] + 연기 계획 → 씬별 생성 작업 (만들 수 없는 씬은 None)."""
    p = provider()
    jobs: list[ClipJob | None] = []
    for index, (scene, act) in enumerate(zip(scenes, acting)):
        section, visual_prompt = scene[0], scene[1]
        refs = _heroine_paths(tuple(scene[2]) if len(scene) > 2 else ())
        ref_file = reference_pose_file() if act else None   # 씬 포즈와 무관한 고정 캐릭터 레퍼런스
        image = p.image_ref(ref_file) if ref_file is not None else None
        if act is None or image is None:
            jobs.append(None)
            continue
        heroine_refs = tuple(r for r in (p.image_ref(ref) for ref in refs) if r)
        key = job_key(section, visual_prompt, act, refs)
        frame_prompt = build_frame_prompt(section, visual_prompt, act, len(heroine_refs)) \
            if _effective_frame_mode(bool(heroine_refs)) == "generate" else None
        jobs.append(ClipJob(index, build_prompt(section, visual_prompt, act, bool(heroine_refs)), image,
                            CLIP_DIR / f"{key}.mp4", frame_prompt, heroine_refs))
    return jobs


def inline_jpeg(path: Path) -> bytes:
    """이미지 파일 → 요청에 직접 담을 JPEG (긴 변 INLINE_IMAGE_MAX_SIDE 이하)."""
    import io

    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((INLINE_IMAGE_MAX_SIDE, INLINE_IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


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
        logger.info("%s 클립 생성 시작: %d개 (캐시 %d개)", label(), len(pending),
                    sum(p is not None for p in results.paths))
        asyncio.run(_generate_all(provider(), pending, results, notify))
    return results


async def _generate_all(p: ModuleType, jobs: list[ClipJob], results: ClipResults, notify) -> None:
    limit = asyncio.Semaphore(max(p.CONCURRENCY, 1))
    stop = asyncio.Event()  # 치명적 오류가 나면 아직 시작하지 않은 요청은 건너뜀
    async with httpx.AsyncClient(base_url=p.API_BASE, headers=p.headers(),
                                 timeout=httpx.Timeout(60, connect=15)) as client:
        runner = _Runner(p, client)

        async def run(job: ClipJob) -> None:
            async with limit:
                if stop.is_set():
                    results.errors[job.index] = "앞선 치명적 오류로 요청 생략"
                    notify(job.index, "failed", None, results.errors[job.index])
                    return
                notify(job.index, "generating", None, None)
                try:
                    await runner.generate_one(job)
                    results.paths[job.index] = job.cache_path
                    notify(job.index, "completed", job.cache_path, None)
                    return
                except VideoAIFatal as e:
                    stop.set()
                    results.fatal_kind = results.fatal_kind or e.kind
                    results.errors[job.index] = str(e)
                    logger.error("%s 치명적 오류 — 남은 씬 생성 중단: %s", p.LABEL, e)
                except Exception as e:
                    results.errors[job.index] = str(e) or type(e).__name__
                    logger.warning("%s 씬 %d 생성 실패: %s", p.LABEL, job.index, e)
                notify(job.index, "failed", None, results.errors[job.index])

        await asyncio.gather(*(run(job) for job in jobs))


class _Runner:
    """업체 어댑터 + HTTP 클라이언트로 작업 제출·폴링·다운로드."""

    def __init__(self, p: ModuleType, client: httpx.AsyncClient):
        self.p, self.client = p, client

    async def generate_one(self, job: ClipJob) -> None:
        """(generate 방식) 장면 키프레임 생성 → 영상 생성 → 다운로드.
        키프레임이 실패하면(치명적 오류 제외) 캐릭터 포즈 이미지를 첫 프레임으로 영상만 만든다."""
        start_frame = None
        if job.frame_prompt:
            try:
                start_frame = await self._keyframe(job)
            except VideoAIFatal:
                raise
            except Exception as e:
                logger.warning("%s 씬 %d 키프레임 생성 실패 → 포즈 이미지를 첫 프레임으로 사용: %s",
                               self.p.LABEL, job.index, e)
        task_id, task = await self._run_task(self.p.video_request(job, start_frame), f"씬 {job.index} 영상")
        await self._download_with_refresh(task_id, task, job.cache_path)
        logger.info("%s 씬 %d 완료 → %s", self.p.LABEL, job.index, job.cache_path.name)

    async def _keyframe(self, job: ClipJob):
        """장면 키프레임 생성 → 받아서 저장(대시보드 미리보기용) → 영상 첫 프레임용 이미지 참조.
        서명 URL을 그대로 넘기지 않고 내려받아 요청에 직접 담는다 (URL 가져오기 장애·만료 영향 없음)."""
        task_id, task = await self._run_task(self.p.frame_request(job), f"씬 {job.index} 키프레임")
        state = self.p.parse_task(task)
        if not state.urls:
            raise VideoAIError(f"키프레임 완료됐지만 이미지 URL이 없음 (id={task_id})")
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15), follow_redirects=True) as cdn:
            r = await cdn.get(state.urls[0])
            r.raise_for_status()
        media_type = r.headers.get("content-type", "").split(";")[0] or "image/jpeg"
        ext = {"image/png": ".png", "image/webp": ".webp"}.get(media_type, ".jpg")
        frame_path = job.cache_path.with_name(job.cache_path.stem + "_frame" + ext)
        frame_path.write_bytes(r.content)
        logger.info("%s 씬 %d 키프레임 → %s", self.p.LABEL, job.index, frame_path.name)
        return self.p.keyframe_ref(r.content, media_type)

    async def _run_task(self, request: tuple[str, dict], what: str) -> tuple[str, dict]:
        """제출 → 폴링(진행 중 → 완료|실패). 완료된 (id, 작업 응답)을 반환.
        다시 시도해도 되는 생성 실패 코드는 같은 요청을 MAX_RESUBMITS번까지 다시 제출한다."""
        path, body = request
        for submit in range(MAX_RESUBMITS + 1):
            response = await self._with_retries(lambda: self._call("POST", path, json=body))
            task_id = self.p.task_id(response)
            if not task_id:
                raise VideoAIError(f"{self.p.LABEL} 응답에 작업 id가 없음")
            logger.info("%s %s 요청됨 (id=%s%s)", self.p.LABEL, what, task_id, f", 재제출 {submit}" if submit else "")
            task = await self._poll(task_id)
            state = self.p.parse_task(task)
            if state.state == "completed":
                return task_id, task
            message = f"{what} 생성 실패: {state.failure or '사유 없음'}" + (f" ({state.code})" if state.code else "")
            action = self.p.failure_action(state.code)
            if action == "fatal":
                raise VideoAIFatal(f"{message} — {self.p.LABEL} 계정 잔액을 확인하세요.", kind="credits")
            if action != "retry" or submit == MAX_RESUBMITS:
                raise VideoAIError(message)
            logger.warning("%s %s — 다시 제출합니다.", self.p.LABEL, message)
            await asyncio.sleep(5 * (submit + 1))
        raise VideoAIError(f"{what} 생성 실패")

    async def _poll(self, task_id: str) -> dict:
        """완료 또는 실패가 될 때까지 POLL_SEC 간격으로 조회."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.p.TIMEOUT_SEC
        state = None
        while loop.time() <= deadline:
            await asyncio.sleep(self.p.POLL_SEC)
            task = await self._with_retries(lambda: self._call("GET", self.p.task_path(task_id)))
            state = self.p.parse_task(task).state
            if state in ("completed", "failed"):
                return task
        raise VideoAIError(f"생성 시간 초과 ({self.p.TIMEOUT_SEC}초, 마지막 상태 {state}, id={task_id})")

    async def _download_with_refresh(self, task_id: str, task: dict, dest: Path) -> None:
        """완성 영상 URL은 만료되는 서명 URL — 받기에 실패하면 작업을 다시 조회해 새 URL로 재시도."""
        for attempt in range(2):
            urls = self.p.parse_task(task).urls
            if not urls:
                raise VideoAIError(f"완료됐지만 영상 URL이 없음 (id={task_id})")
            try:
                await _download(urls[0], dest)
                return
            except VideoAIError:
                if attempt == 1:
                    raise
                logger.warning("%s 영상 다운로드 실패 — 새 URL을 받아 다시 시도 (id=%s)", self.p.LABEL, task_id)
                task = await self._with_retries(lambda: self._call("GET", self.p.task_path(task_id)))

    async def _with_retries(self, make_call):
        """한도 초과·일시 오류는 Retry-After(없으면 지수 대기 + 지터)만큼 기다렸다 재시도."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await make_call()
            except VideoAIRetryable as e:
                if attempt == MAX_RETRIES:
                    raise VideoAIError(f"재시도 {MAX_RETRIES}회 후에도 실패: {e}") from e
                wait = e.retry_after if e.retry_after is not None else min(2 ** attempt * 3, 60)
                wait += random.uniform(0, 1.5)
                logger.warning("%s 일시 오류 (%s) — %.1f초 후 재시도", self.p.LABEL, e, wait)
                await asyncio.sleep(wait)

    async def _call(self, method: str, url: str, **kwargs) -> dict:
        try:
            r = await self.client.request(method, url, **kwargs)
        except httpx.TransportError as e:
            raise VideoAIRetryable(f"네트워크 오류: {type(e).__name__}") from e
        if r.status_code < 300:
            return r.json()
        raise self.p.http_error(r)


# ---------- 업체 어댑터가 쓰는 도우미 ----------
def error_detail(r: httpx.Response) -> str:
    try:
        body = r.json()
        return str(body.get("detail") or body.get("error") or body)[:300]
    except Exception:
        return r.text[:300]


def retry_after(r: httpx.Response) -> float | None:
    value = r.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def transient_error(r: httpx.Response, name: str, detail: str) -> VideoAIRetryable | None:
    """한도 초과(429)·서버 오류(5xx)면 재시도용 예외, 아니면 None."""
    if r.status_code == 429 or r.status_code >= 500:
        return VideoAIRetryable(f"{name} {'한도 초과' if r.status_code == 429 else '서버 오류'} ({r.status_code}) {detail}",
                                retry_after(r))
    return None


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
                raise VideoAIError("빈 영상 파일")
            tmp.replace(dest)
            return
        except (httpx.HTTPError, VideoAIError) as e:
            tmp.unlink(missing_ok=True)
            expired = isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500
            if expired or attempt == 2:  # 4xx(만료된 서명 URL 등)는 같은 주소로 다시 받아도 소용없음 → 새 URL로
                raise VideoAIError(f"영상 다운로드 실패: {e}") from e
            await asyncio.sleep(2 ** attempt)
