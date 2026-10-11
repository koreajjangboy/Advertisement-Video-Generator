import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class EpisodeStatus(str, enum.Enum):
    DRAFT = "draft"              # 생성됨
    SCRIPTING = "scripting"      # 대본 생성 중
    SCRIPTED = "scripted"        # 대본 생성 완료
    VOICING = "voicing"          # TTS 음성 생성 중
    CLIPPING = "clipping"        # 1단계: 씬 영상(Luma/대체 클립) 생성 중
    CLIPPED = "clipped"          # 1단계 완료 — 최종 렌더링 대기
    RENDERING = "rendering"      # 2단계: 최종 영상 렌더링 중
    DONE = "done"                # 완료
    FAILED = "failed"            # 실패


class SceneRenderStatus(str, enum.Enum):
    """씬 단위 렌더링 상태 (DB에는 문자열로 저장 — enum 타입 마이그레이션 불필요)."""

    PENDING = "pending"          # 음성 없음 (TTS 필요)
    READY = "ready"              # 음성 준비됨, 렌더링 대기 (또는 수정되어 재렌더링 필요)
    GENERATING = "generating"    # AI 씬 영상(Luma) 생성 중
    RENDERING = "rendering"      # 렌더링 중
    COMPLETED = "completed"      # 최신 영상에 반영됨
    FAILED = "failed"            # 렌더링 실패


class ClipStatus(str, enum.Enum):
    """씬 영상(1단계) 상태 (DB에는 문자열)."""

    NONE = "none"                # 아직 없음
    QUEUED = "queued"            # 생성 대기
    GENERATING = "generating"    # 생성 중 (Luma 요청·폴링 또는 대체 클립 합성)
    COMPLETED = "completed"      # 준비됨 (clip_source: luma | fallback)
    FAILED = "failed"            # 실패 (clip_error)


class Episode(Base):
    """광고 영상 1편 단위."""

    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    product_name: Mapped[str] = mapped_column(String(200))  # 도서명
    product_description: Mapped[str] = mapped_column(Text, default="")
    source_text: Mapped[str] = mapped_column(Text, default="")  # 원문 발췌
    target_audience: Mapped[str] = mapped_column(String(200), default="")
    duration_sec: Mapped[int] = mapped_column(Integer, default=30)
    status: Mapped[EpisodeStatus] = mapped_column(
        Enum(EpisodeStatus, values_callable=lambda e: [m.value for m in e]),
        default=EpisodeStatus.DRAFT,
    )
    video_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 여자주인공: 등장 여부, 레퍼런스 이미지 폴더(프로젝트 기준 상대 경로), 대표 이미지 파일명
    heroine_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    heroine_dir: Mapped[str] = mapped_column(String(500), default="")
    heroine_ref: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    scripts: Mapped[list["Script"]] = relationship(
        back_populates="episode", cascade="all, delete-orphan", order_by="Script.scene_order"
    )


class Script(Base):
    """에피소드의 장면(Scene)별 대본."""

    __tablename__ = "scripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episodes.id", ondelete="CASCADE"), index=True)
    scene_order: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(10), default="body")  # hook | body | cta
    narration: Mapped[str] = mapped_column(Text)              # 내레이션(TTS 대상)
    on_screen_text: Mapped[str] = mapped_column(Text, default="")  # 화면 자막
    visual_prompt: Mapped[str] = mapped_column(Text, default="")   # 이미지/영상 생성용 프롬프트
    emoji: Mapped[str] = mapped_column(String(32), default="")      # 장면 스티커 일러스트 (비면 렌더링 시 자동 선택)
    host_pose: Mapped[str] = mapped_column(String(20), default="")  # 호스트 캐릭터 포즈 (비면 대본에 맞게 자동)
    heroine: Mapped[bool] = mapped_column(Boolean, default=False)   # 이 씬에 여자주인공 등장 (에피소드 설정이 켜져 있을 때)
    duration_sec: Mapped[float] = mapped_column(default=5.0)
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)  # /media/audio/...
    audio_duration_sec: Mapped[float | None] = mapped_column(nullable=True)
    audio_provider: Mapped[str | None] = mapped_column(String(20), nullable=True)  # gemini | gtts
    image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)  # /media/images/... 배경
    render_status: Mapped[str] = mapped_column(String(20), default=SceneRenderStatus.PENDING.value)
    # 1단계 씬 영상
    clip_path: Mapped[str | None] = mapped_column(String(500), nullable=True)  # /media/clips/...
    clip_source: Mapped[str | None] = mapped_column(String(20), nullable=True)  # luma | fallback
    clip_status: Mapped[str] = mapped_column(String(20), default=ClipStatus.NONE.value)
    clip_key: Mapped[str | None] = mapped_column(String(64), nullable=True)   # 만들 때의 대본 기준 키 (최신 여부 판단)
    clip_error: Mapped[str | None] = mapped_column(String(500), nullable=True)

    def mark_render_stale(self) -> None:
        """대본·음성·이미지가 바뀌어 최신 영상과 달라졌을 때 — 음성 유무로 pending/ready."""
        self.render_status = (SceneRenderStatus.READY if self.audio_path else SceneRenderStatus.PENDING).value

    episode: Mapped[Episode] = relationship(back_populates="scripts")
