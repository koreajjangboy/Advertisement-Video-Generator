"""라우터 공용 헬퍼 (에피소드 조회·작업 중복 방지)."""

from fastapi import HTTPException
from sqlalchemy.orm import Session

from models import Episode, EpisodeStatus

# 백그라운드 작업이 진행 중인 상태 — 이 상태에서는 다른 작업·수정을 받지 않는다
BUSY_STATUSES = (EpisodeStatus.SCRIPTING, EpisodeStatus.VOICING, EpisodeStatus.CLIPPING, EpisodeStatus.RENDERING)


def ensure_idle(episode: Episode) -> None:
    if episode.status in BUSY_STATUSES:
        raise HTTPException(409, "작업이 진행 중인 에피소드입니다. 완료 후 다시 시도하세요.")


def get_episode_or_404(db: Session, episode_id: int) -> Episode:
    episode = db.get(Episode, episode_id)
    if episode is None:
        raise HTTPException(404, "에피소드를 찾을 수 없습니다.")
    return episode
