"""영상 제작 1·2단계 API (3·4단계는 routers/video.py).

1단계 POST /api/generate-script  — 원문으로 씬별 대본(내레이션·화면 자막·영상 프롬프트) 생성. 음성은 만들지 않는다.
       생성된 대본은 PATCH /api/scripts/{id}로 수정·저장.
2단계 POST /api/generate-tts     — 확정된 대본으로 씬별 음성(TTS) 생성. 기본은 음성이 없는 씬만, force면 전부,
       script_ids로 특정 씬만. 씬 하나는 POST /api/scripts/{id}/tts(즉시 생성 후 결과 반환)로도 가능.
두 작업 모두 백그라운드로 실행되고 즉시 202를 반환한다.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import EpisodeStatus
from routers.common import ensure_idle, get_episode_or_404
from schemas import ScriptGenerateRequest, TTSGenerateRequest
from services import pipeline

router = APIRouter(prefix="/api", tags=["production"])


@router.post("/generate-script", status_code=202)
def generate_script(payload: ScriptGenerateRequest, bg: BackgroundTasks, db: Session = Depends(get_db)):
    """1단계: 대본 (재)생성. 기존 씬의 음성·씬 영상 연결은 새 대본으로 바뀌므로 다시 만들어야 한다."""
    episode = get_episode_or_404(db, payload.episode_id)
    ensure_idle(episode)
    episode.status = EpisodeStatus.SCRIPTING
    db.commit()
    bg.add_task(pipeline.generate_script_for_episode, episode.id)
    return {"message": "1단계 대본 생성을 시작했습니다. 완료되면 씬별로 확인·수정한 뒤 2단계 음성을 만드세요.",
            "episode_id": episode.id}


@router.post("/generate-tts", status_code=202)
def generate_tts(payload: TTSGenerateRequest, bg: BackgroundTasks, db: Session = Depends(get_db)):
    """2단계: 씬별 음성 생성."""
    episode = get_episode_or_404(db, payload.episode_id)
    if not episode.scripts:
        raise HTTPException(400, "대본이 없습니다. 먼저 1단계 대본을 생성하세요.")
    ensure_idle(episode)
    script_ids = payload.script_ids
    if script_ids is not None:
        own = {s.id for s in episode.scripts}
        if not script_ids or not set(script_ids) <= own:
            raise HTTPException(400, "이 에피소드의 씬이 아닌 script_id가 있습니다.")
    targets = [s for s in episode.scripts if (script_ids is None or s.id in script_ids)
               and (payload.force or script_ids is not None or not s.audio_path)]
    if not targets:
        return {"message": "모든 씬에 음성이 이미 있습니다. 다시 만들려면 '모두 다시 생성'을 선택하세요.",
                "episode_id": episode.id, "scenes": 0}
    episode.status = EpisodeStatus.VOICING
    db.commit()
    bg.add_task(pipeline.generate_audio_for_episode, episode.id, payload.force or script_ids is not None,
                [s.id for s in targets])
    return {"message": f"2단계 음성 생성을 시작했습니다 ({len(targets)}개 씬).", "episode_id": episode.id,
            "scenes": len(targets)}
