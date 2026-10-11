import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import API_MODE, BASE_DIR, HEROINE_DIR, LLM_FALLBACK, LLM_PROVIDER, MEDIA_DIR, TTS_FALLBACK, TTS_PROVIDER
from database import get_db, init_db
from models import ClipStatus, Episode, EpisodeStatus, SceneRenderStatus, Script
from routers import production as production_router
from routers import video as video_router
from routers.common import BUSY_STATUSES
from routers.common import ensure_idle as _ensure_idle
from routers.common import get_episode_or_404 as _get_or_404
from schemas import (
    EpisodeCreate, EpisodeOut, EpisodeUpdate, RenderFinalRequest, ScriptGenerateRequest, ScriptOut, ScriptUpdate, TTSGenerateRequest,
    TTSPreviewOut, TTSPreviewRequest,
)
from services import (
    clip_service, heroine_service, host_service, llm_service, luma_service, pipeline, render_service, tts_service,
)

logging.basicConfig(level=logging.INFO)

MAX_IMAGE_BYTES = 15 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _recover_interrupted_jobs()
    _repair_audio()
    pipeline.cleanup_stale_media()
    yield


def _recover_interrupted_jobs():
    """서버가 작업 도중 종료되면 BackgroundTasks가 사라져 상태가 '진행 중'에 고정되므로 복구."""
    from database import SessionLocal

    with SessionLocal() as db:
        for ep in db.scalars(select(Episode).where(Episode.status.in_(BUSY_STATUSES))):
            interrupted = ep.status.value
            ep.status = EpisodeStatus.SCRIPTED if ep.scripts else EpisodeStatus.FAILED
            ep.error_message = f"서버 재시작으로 '{interrupted}' 작업이 중단되었습니다. 다시 실행하세요."
            for s in ep.scripts:
                if s.render_status in (SceneRenderStatus.RENDERING.value, SceneRenderStatus.GENERATING.value):
                    s.mark_render_stale()
                if s.clip_status in (ClipStatus.QUEUED.value, ClipStatus.GENERATING.value):
                    s.clip_status = ClipStatus.FAILED.value
                    s.clip_error = "서버 재시작으로 생성이 중단되었습니다."
        # 2단계 분리 전(한 번에 렌더링하던 때) 만들어진 씬 클립 → 다시 생성(과금)하지 않도록 지금 대본 기준으로 등록
        legacy = {s.episode for s in db.scalars(select(Script).where(Script.clip_path.is_not(None)))
                  if clip_service.needs_adoption(s)}
        for ep in legacy:
            clip_service.adopt_legacy_clips(list(ep.scripts))
        # render_status 컬럼 추가 전에 만들어진 씬: 음성 유무·완성 영상 여부로 초기값 보정
        for s in db.scalars(select(Script).where(
            Script.render_status == SceneRenderStatus.PENDING.value, Script.audio_path.is_not(None)
        )):
            done = s.episode.status == EpisodeStatus.DONE
            s.render_status = (SceneRenderStatus.COMPLETED if done else SceneRenderStatus.READY).value
        db.commit()


def _repair_audio():
    """예전 TTS 코드가 저장한 손상 WAV(끝단 잡음)를 복구하고, 그 오디오를 쓰는 씬의 길이·렌더링 상태를 갱신."""
    from database import SessionLocal

    repaired = tts_service.repair_audio_files()
    if not repaired:
        return
    urls = {tts_service.media_path_to_url(path): duration for path, duration in repaired.items()
            if path.is_relative_to(MEDIA_DIR)}
    with SessionLocal() as db:
        for s in db.scalars(select(Script).where(Script.audio_path.in_(list(urls)))):
            s.audio_duration_sec = round(urls[s.audio_path], 2)
            s.mark_render_stale()  # 기존 영상에는 잡음이 들어가 있으므로 다시 렌더링 필요 표시
            if s.episode.status == EpisodeStatus.DONE:
                s.episode.status = EpisodeStatus.CLIPPED if any(x.clip_path for x in s.episode.scripts) \
                    else EpisodeStatus.SCRIPTED
                s.episode.error_message = "TTS 음성 끝 잡음을 복구했습니다. 4단계 렌더링을 다시 실행하면 깨끗한 음성으로 반영됩니다."
        db.commit()


app = FastAPI(title="Short-form Ad Video Generator", lifespan=lifespan)
app.include_router(production_router.router)  # 1단계 대본 / 2단계 음성
app.include_router(video_router.router)  # 3단계 씬 영상 / 4단계 최종 렌더링

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ---------- 관리자 대시보드 ----------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ---------- API ----------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/episodes", response_model=list[EpisodeOut])
def list_episodes(db: Session = Depends(get_db)):
    return db.scalars(select(Episode).order_by(Episode.id.desc())).all()


@app.post("/api/episodes", response_model=EpisodeOut, status_code=201)
def create_episode(
    payload: EpisodeCreate,
    bg: BackgroundTasks,
    auto_script: bool = True,
    db: Session = Depends(get_db),
):
    """에피소드 생성 후 (기본값) 백그라운드에서 LLM 대본 생성 → Script 테이블 저장."""
    data = payload.model_dump()
    data.update(_heroine_settings(payload.heroine_enabled, payload.heroine_dir, payload.heroine_ref))
    episode = Episode(**data)
    if auto_script:
        episode.status = EpisodeStatus.SCRIPTING
    db.add(episode)
    db.commit()
    db.refresh(episode)
    if auto_script:
        bg.add_task(pipeline.generate_script_for_episode, episode.id)
    return episode


def _heroine_settings(enabled: bool, dir_value: str, ref: str) -> dict:
    """여자주인공 설정 검증·정규화. 켜져 있으면 폴더·이미지가 있어야 하고, 대표 이미지는 비면 첫 이미지."""
    try:
        folder = heroine_service.resolve_dir(dir_value)
        names = [p.name for p in heroine_service.list_images(folder)]
    except ValueError as e:
        if enabled:
            raise HTTPException(400, str(e))
        return {"heroine_enabled": False, "heroine_dir": (dir_value or "").strip(), "heroine_ref": ref or ""}
    if enabled and not names:
        raise HTTPException(400, f"여자주인공 이미지 폴더에 이미지(png·jpg·webp)가 없습니다: {heroine_service.to_setting(folder)}")
    if ref and ref not in names:
        raise HTTPException(400, f"대표 이미지가 폴더에 없습니다: {ref}")
    return {"heroine_enabled": enabled, "heroine_dir": heroine_service.to_setting(folder),
            "heroine_ref": ref or (names[0] if names else "")}


@app.patch("/api/episodes/{episode_id}", response_model=EpisodeOut)
def update_episode(episode_id: int, payload: EpisodeUpdate, db: Session = Depends(get_db)):
    """여자주인공 설정 변경. 켜면 등장 씬이 하나도 없을 때 본문(body) 씬에 등장시킨다.
    바뀐 설정은 해당 씬의 씬 영상(3단계)을 '다시 생성 필요'로 만든다."""
    episode = _get_or_404(db, episode_id)
    _ensure_idle(episode)
    changes = payload.model_dump(exclude_unset=True)
    settings = _heroine_settings(
        changes.get("heroine_enabled", episode.heroine_enabled),
        changes.get("heroine_dir", episode.heroine_dir),
        changes.get("heroine_ref", episode.heroine_ref if "heroine_dir" not in changes else ""),
    )
    changed = any(getattr(episode, k) != v for k, v in settings.items())
    for key, value in settings.items():
        setattr(episode, key, value)
    if changed:
        pipeline.ensure_heroine_scenes(episode)
        if episode.status == EpisodeStatus.DONE:
            episode.status = EpisodeStatus.CLIPPED if any(s.clip_path for s in episode.scripts) else EpisodeStatus.SCRIPTED
    db.commit()
    db.refresh(episode)
    return episode


@app.get("/api/heroine/folders")
def heroine_folders():
    """여자주인공 캐릭터 폴더 목록 (media/images/ 아래 이미지가 있는 폴더, JB 포즈 폴더 제외) + 기본 폴더."""
    return {"folders": heroine_service.list_folders(), "default_dir": HEROINE_DIR}


@app.get("/api/heroine/images")
def heroine_images(dir: str = ""):
    """여자주인공 이미지 폴더 확인 + 이미지 목록 (비우면 기본 폴더). media/ 안쪽 폴더만 허용."""
    try:
        return heroine_service.catalog(dir)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/episodes/{episode_id}", response_model=EpisodeOut)
def get_episode(episode_id: int, db: Session = Depends(get_db)):
    return _get_or_404(db, episode_id)


@app.delete("/api/episodes/{episode_id}", status_code=204)
def delete_episode(episode_id: int, db: Session = Depends(get_db)):
    episode = _get_or_404(db, episode_id)
    pipeline.delete_episode_media(episode)
    db.delete(episode)
    db.commit()


@app.post("/api/episodes/{episode_id}/script", status_code=202)
def generate_script(episode_id: int, bg: BackgroundTasks, db: Session = Depends(get_db)):
    """(호환용) 1단계 대본 (재)생성 — POST /api/generate-script 와 같음."""
    return production_router.generate_script(ScriptGenerateRequest(episode_id=episode_id), bg, db)


@app.post("/api/episodes/{episode_id}/render", status_code=202)
def render_video(episode_id: int, bg: BackgroundTasks, db: Session = Depends(get_db)):
    """(호환용) 4단계 최종 렌더링 — POST /api/render-final-video 와 같음."""
    return video_router.render_final_video(RenderFinalRequest(episode_id=episode_id), bg, db)


@app.get("/api/episodes/{episode_id}/video")
def download_video(episode_id: int, db: Session = Depends(get_db)):
    """완성 영상 다운로드 (에피소드 제목을 파일명으로)."""
    episode = _get_or_404(db, episode_id)
    path = tts_service.media_url_to_path(episode.video_path)
    if path is None or not path.exists():
        raise HTTPException(404, "렌더링된 영상이 없습니다.")
    safe_title = "".join(c for c in episode.title if c not in '\\/:*?"<>|').strip() or "video"
    return FileResponse(path, media_type="video/mp4", filename=f"{episode.id}_{safe_title}.mp4")


@app.post("/api/episodes/{episode_id}/tts", status_code=202)
def generate_audio(
    episode_id: int, bg: BackgroundTasks, force: bool = False, db: Session = Depends(get_db)
):
    """(호환용) 2단계 음성 생성 — POST /api/generate-tts 와 같음."""
    return production_router.generate_tts(TTSGenerateRequest(episode_id=episode_id, force=force), bg, db)


@app.patch("/api/scripts/{script_id}", response_model=ScriptOut)
def update_script(script_id: int, payload: ScriptUpdate, db: Session = Depends(get_db)):
    """씬 대본 수정. 내레이션이 바뀌면 기존 TTS 오디오를 무효화하고,
    완료된 영상이 있으면 상태를 scripted로 되돌려 재렌더링이 필요함을 표시한다."""
    script = db.get(Script, script_id)
    if script is None:
        raise HTTPException(404, "씬을 찾을 수 없습니다.")
    episode = script.episode
    _ensure_idle(episode)

    changes = {k: v for k, v in payload.model_dump(exclude_unset=True).items()
               if v is not None and getattr(script, k) != v}
    if not changes:
        return script

    for key, value in changes.items():
        setattr(script, key, value)
    if "narration" in changes:
        # 기존 음성 무효화 → 씬별 TTS 또는 다음 렌더링 때 재생성
        tts_service.delete_media(script.audio_path)
        script.audio_path = None
        script.audio_duration_sec = None
        script.audio_provider = None
    _mark_video_stale(script)
    db.commit()
    db.refresh(script)
    return script


@app.post("/api/scripts/{script_id}/tts", response_model=ScriptOut)
def generate_scene_audio(script_id: int, db: Session = Depends(get_db)):
    """씬 1개 TTS를 동기 실행 (수정 직후 바로 들어보기용)."""
    script = db.get(Script, script_id)
    if script is None:
        raise HTTPException(404, "씬을 찾을 수 없습니다.")
    _ensure_idle(script.episode)
    try:
        pipeline.synthesize_scene(db, script_id, force=True)
    except tts_service.TTSRateLimited as e:
        raise _rate_limited(e)
    except Exception as e:
        raise HTTPException(502, f"TTS 실패: {e}")
    return db.get(Script, script_id)


@app.post("/api/scripts/{script_id}/image", response_model=ScriptOut)
async def upload_scene_image(script_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """씬 배경 이미지 업로드 (없으면 렌더링 시 섹션 색 그라데이션 배경 사용)."""
    script = db.get(Script, script_id)
    if script is None:
        raise HTTPException(404, "씬을 찾을 수 없습니다.")
    _ensure_idle(script.episode)
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"이미지는 {MAX_IMAGE_BYTES // 1024 // 1024}MB 이하만 업로드할 수 있습니다.")
    try:
        url = render_service.save_scene_image(data, f"ep{script.episode_id}_scene{script.id}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    old = script.image_path
    script.image_path = url
    _mark_video_stale(script)
    db.commit()
    tts_service.delete_media(old)
    db.refresh(script)
    return script


@app.delete("/api/scripts/{script_id}/image", response_model=ScriptOut)
def delete_scene_image(script_id: int, db: Session = Depends(get_db)):
    script = db.get(Script, script_id)
    if script is None:
        raise HTTPException(404, "씬을 찾을 수 없습니다.")
    _ensure_idle(script.episode)
    old, script.image_path = script.image_path, None
    if old:
        _mark_video_stale(script)
    db.commit()
    tts_service.delete_media(old)
    db.refresh(script)
    return script


@app.post("/api/tts/preview", response_model=TTSPreviewOut)
def preview_tts(payload: TTSPreviewRequest):
    """저장 전 내레이션 미리듣기 (DB 변경 없음)."""
    try:
        result = tts_service.preview(payload.text)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except tts_service.TTSRateLimited as e:
        raise _rate_limited(e)
    except Exception as e:
        raise HTTPException(502, f"TTS 실패: {e}")
    return TTSPreviewOut(audio_url=result.url, duration_sec=result.duration_sec, provider=result.provider)


@app.get("/api/status")
def app_status():
    """API 모드, 대본/TTS 엔진(한도 초과 시 실제 사용 경로), 렌더링 환경 (대시보드 표시용)."""
    return {
        "api_mode": API_MODE,
        "llm": {"provider": LLM_PROVIDER, "fallback": LLM_FALLBACK or None, "active": llm_service.active_model()},
        "tts": {"provider": TTS_PROVIDER, "fallback": TTS_FALLBACK or None, "active": tts_service.active_provider()},
        "render": {"ffmpeg": render_service.find_ffmpeg(), "font": render_service.find_font()},
        "host": {"mode": host_service.mode(), "poses": host_service.pose_catalog()},
        "video_ai": luma_service.status(),
        "heroine": {"default_dir": HEROINE_DIR},
    }


def _rate_limited(e: "tts_service.TTSRateLimited") -> HTTPException:
    return HTTPException(429, str(e), headers={"Retry-After": str(int(e.retry_after) + 1)})


def _mark_video_stale(script: Script) -> None:
    """대본·이미지가 바뀌면 완성 영상이 최신이 아님 → 씬은 pending/ready, 에피소드는 scripted로
    되돌려 재렌더링 필요를 표시."""
    script.mark_render_stale()
    episode = script.episode
    if episode.status in (EpisodeStatus.DONE, EpisodeStatus.FAILED):
        episode.status = EpisodeStatus.CLIPPED if any(s.clip_path for s in episode.scripts) else EpisodeStatus.SCRIPTED
        episode.error_message = None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
