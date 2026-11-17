"""영상 제작 3·4단계 API (1·2단계 대본·음성은 routers/production.py).

3단계 POST /api/generate-scene-clips — 씬별 캐릭터 연기 클립만 생성·다운로드 (AI 씬 영상: VIDEO_PROVIDER=luma|runway).
       크레딧 부족·씬 실패·AI 사용 불가 시 대체 클립으로 자동 전환 (mode=auto).
       진행 상황: GET /api/episodes/{id}/clips (또는 에피소드 조회의 scripts[].clip_status)
4단계 POST /api/render-final-video   — 준비된 씬 클립 + TTS + BGM + 자막·제목 바를 FFmpeg로 최종 합성.
       클립이 없거나 대본이 바뀐 씬은 대체 클립(기본) 또는 캐릭터 카드 연출로 채운다.
두 작업 모두 백그라운드로 실행되고 즉시 202를 반환한다.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import ClipStatus, EpisodeStatus
from routers.common import ensure_idle, get_episode_or_404
from schemas import ClipGenerateRequest, RenderFinalRequest
from services import clip_service, host_service, pipeline, render_service, video_ai_service

router = APIRouter(prefix="/api", tags=["video"])


@router.post("/generate-scene-clips", status_code=202)
def generate_scene_clips(payload: ClipGenerateRequest, bg: BackgroundTasks, db: Session = Depends(get_db)):
    """3단계: 대본(씬)과 캐릭터 포즈 이미지로 씬별 연기 클립 생성."""
    episode = get_episode_or_404(db, payload.episode_id)
    if not episode.scripts:
        raise HTTPException(400, "대본이 없습니다. 먼저 대본을 생성하세요.")
    ensure_idle(episode)
    if host_service.mode() != "character":
        raise HTTPException(400, "캐릭터 포즈 이미지(HOST_POSE_DIR)가 없어 씬 영상을 만들 수 없습니다.")
    if render_service.find_ffmpeg() is None:
        raise HTTPException(503, "ffmpeg를 찾을 수 없습니다 (대체 클립·렌더링에 필요).")
    name = video_ai_service.label()
    reason = video_ai_service.unavailable_reason()
    if payload.mode == "ai" and reason:
        raise HTTPException(400, f"{name}를 사용할 수 없습니다: {reason}")
    episode.status = EpisodeStatus.CLIPPING
    db.commit()
    bg.add_task(pipeline.generate_clips_for_episode, episode.id, payload.mode, payload.force)
    if payload.mode == "fallback":
        how = "대체 클립(무료)으로"
    elif reason:
        how = f"{name}를 사용할 수 없어({reason}) 대체 클립으로"
    else:
        model = video_ai_service.status()["model"]
        how = f"{name} {model}로 (씬마다 수 분, 실패 시 대체 클립)" if payload.mode == "auto" else f"{name} {model}로"
    return {"message": f"3단계 씬 영상 생성을 시작했습니다 — {how}.", "episode_id": episode.id, "mode": payload.mode}


@router.get("/episodes/{episode_id}/clips")
def clip_progress(episode_id: int, db: Session = Depends(get_db)):
    """3단계 진행 상황 요약 + 씬별 클립 상태."""
    episode = get_episode_or_404(db, episode_id)
    scripts = list(episode.scripts)
    keys = clip_service.expected_keys([clip_service.spec_for(s, episode) for s in scripts]) \
        if host_service.mode() == "character" else [(None, None)] * len(scripts)
    scenes = [{
        "script_id": s.id, "scene_order": s.scene_order, "status": s.clip_status, "source": s.clip_source,
        "clip_path": s.clip_path, "error": s.clip_error,
        "current": clip_service.is_current(s.clip_key, k) if s.clip_path else None,
    } for s, k in zip(scripts, keys)]
    count = lambda pred: sum(1 for sc in scenes if pred(sc))  # noqa: E731
    return {
        "episode_status": episode.status.value,
        "total": len(scenes),
        "ready": count(lambda sc: sc["clip_path"] and sc["current"]),
        "ai": count(lambda sc: sc["clip_path"] and sc["current"] and video_ai_service.is_ai_source(sc["source"])),
        "fallback": count(lambda sc: sc["clip_path"] and sc["current"] and sc["source"] == "fallback"),
        "generating": count(lambda sc: sc["status"] in (ClipStatus.QUEUED.value, ClipStatus.GENERATING.value)),
        "failed": count(lambda sc: sc["status"] == ClipStatus.FAILED.value),
        "scenes": scenes,
    }


@router.post("/render-final-video", status_code=202)
def render_final_video(payload: RenderFinalRequest, bg: BackgroundTasks, db: Session = Depends(get_db)):
    """4단계: 씬 클립 + TTS + BGM + 자막을 FFmpeg로 최종 합성."""
    episode = get_episode_or_404(db, payload.episode_id)
    if not episode.scripts:
        raise HTTPException(400, "대본이 없습니다. 먼저 대본을 생성하세요.")
    ensure_idle(episode)
    try:
        render_service.check_ready()  # ffmpeg·폰트가 없으면 백그라운드로 넘기기 전에 바로 안내
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    episode.status = EpisodeStatus.RENDERING
    db.commit()
    bg.add_task(pipeline.render_video_for_episode, episode.id, payload.missing_clip, payload.use_outdated)
    return {"message": "4단계 최종 렌더링을 시작했습니다. 완료되면 아래에서 미리보기·다운로드할 수 있습니다.",
            "episode_id": episode.id}
