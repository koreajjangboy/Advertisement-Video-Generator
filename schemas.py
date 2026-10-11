from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models import EpisodeStatus


class EpisodeCreate(BaseModel):
    title: str
    product_name: str = "래퍼의 노트"
    product_description: str = ""
    source_text: str = Field(min_length=1)
    target_audience: str = ""
    duration_sec: int = Field(30, ge=5, le=180)
    heroine_enabled: bool = False
    heroine_dir: str = Field("", max_length=500)   # 비우면 기본 폴더 (HEROINE_DIR). 절대 경로도 가능 (media/ 안쪽)
    heroine_ref: str = Field("", max_length=255)   # 대표 이미지 파일명 (비우면 폴더의 첫 이미지)


class EpisodeUpdate(BaseModel):
    """에피소드 설정 부분 수정 (여자주인공 설정)."""

    heroine_enabled: bool | None = None
    heroine_dir: str | None = Field(None, max_length=500)
    heroine_ref: str | None = Field(None, max_length=255)


class ScriptUpdate(BaseModel):
    """씬 부분 수정 (보낸 필드만 반영)."""

    narration: str | None = Field(None, min_length=1)
    on_screen_text: str | None = None
    visual_prompt: str | None = None
    emoji: str | None = Field(None, max_length=16)
    host_pose: str | None = Field(None, max_length=20)
    heroine: bool | None = None
    duration_sec: float | None = Field(None, ge=0.5, le=60)


class TTSPreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class TTSPreviewOut(BaseModel):
    audio_url: str
    duration_sec: float
    provider: str


class ScriptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    scene_order: int
    section: str
    narration: str
    on_screen_text: str
    visual_prompt: str
    emoji: str
    host_pose: str
    heroine: bool = False
    duration_sec: float
    audio_path: str | None
    audio_duration_sec: float | None
    audio_provider: str | None
    image_path: str | None
    render_status: str
    clip_path: str | None = None
    clip_source: str | None = None
    clip_status: str = "none"
    clip_error: str | None = None
    clip_key: str | None = Field(None, exclude=True)  # 최신 여부 계산용 (응답에는 포함하지 않음)
    clip_frame: str | None = None      # Luma가 만든 장면 키프레임 이미지 (generate 방식일 때)
    # Luma에 실제로 보낼 프롬프트 미리보기 {mode, user_prompt, frame, video} — 캐릭터 모드가 아니면 None
    luma_prompt: dict | None = None
    # 클립이 지금 대본 기준인지 (대본·포즈를 고치면 False → 1단계 다시 실행 필요). 클립이 없으면 None
    clip_current: bool | None = None
    # 렌더링 때 캐릭터가 이 씬에서 연기할 내용 (지정 포즈 또는 대본 감정선 기반 자동). 캐릭터 모드가 아니면 None.
    # resolved_pose: 씬 첫 포즈, resolved_acting: 감정 비트 [{pose, emotion, label, at}]
    # 에피소드 단위 응답에서는 앞 씬과 겹치지 않게 순서대로 결정한 값으로 덮어쓴다 (EpisodeOut).
    resolved_pose: str | None = None
    resolved_acting: list[dict] | None = None

    def _apply_acting(self, acting) -> None:
        self.resolved_pose = acting.pose if acting else None
        self.resolved_acting = acting.summary() if acting else None

    @model_validator(mode="after")
    def _resolve_pose(self):
        if self.resolved_pose is None:
            from services import host_service

            self._apply_acting(host_service.plan_acting(
                [(self.section, self.on_screen_text, self.narration, self.scene_order, self.host_pose)])[0])
        return self


class ScriptGenerateRequest(BaseModel):
    episode_id: int


class TTSGenerateRequest(BaseModel):
    episode_id: int
    force: bool = False                 # 이미 음성이 있는 씬도 다시 생성
    script_ids: list[int] | None = None  # 특정 씬만 (지정하면 기존 음성이 있어도 다시 생성)


class ClipGenerateRequest(BaseModel):
    episode_id: int
    # auto: Luma로 생성하고 실패·크레딧 부족(402)이면 대체 클립 / luma: Luma만 (실패 씬은 실패로 남김)
    # fallback: Luma를 호출하지 않고 대체 클립만 (무료)
    mode: str = Field("auto", pattern="^(auto|luma|fallback)$")
    force: bool = False  # 이미 최신 클립이 있는 씬도 다시 생성


class RenderFinalRequest(BaseModel):
    episode_id: int
    # 클립이 없거나 대본이 바뀌어 최신이 아닌 씬: fallback(대체 클립을 그 자리에서 만들어 사용) | card(캐릭터 카드 연출)
    missing_clip: str = Field("fallback", pattern="^(fallback|card)$")
    # 대본·영상 프롬프트가 바뀌어 '다시 생성 필요'인 기존 씬 영상도 사용할지 (False면 missing_clip 방식으로 대체)
    use_outdated: bool = True


class EpisodeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    product_name: str
    product_description: str
    source_text: str
    target_audience: str
    duration_sec: int
    status: EpisodeStatus
    video_path: str | None
    error_message: str | None
    heroine_enabled: bool = False
    heroine_dir: str = ""
    heroine_ref: str = ""
    created_at: datetime
    updated_at: datetime
    scripts: list[ScriptOut] = []

    @model_validator(mode="after")
    def _resolve_poses_in_order(self):
        from services import host_service

        acting = host_service.plan_acting(
            [(s.section, s.on_screen_text, s.narration, s.scene_order, s.host_pose) for s in self.scripts])
        from services import clip_service, luma_service

        specs = [clip_service.spec_for(s, self) for s in self.scripts]
        for script, act, spec in zip(self.scripts, acting, specs):
            script._apply_acting(act)
            if act is not None:
                script.luma_prompt = luma_service.prompt_preview(script.section, script.visual_prompt, act,
                                                                 spec.heroine_refs)
            if script.clip_path:
                from services import tts_service

                clip = tts_service.media_url_to_path(script.clip_path)
                frame = next((clip.with_name(clip.stem + "_frame" + ext) for ext in (".jpg", ".png", ".webp")
                              if clip and clip.with_name(clip.stem + "_frame" + ext).exists()), None)
                script.clip_frame = tts_service.media_path_to_url(frame) if frame else None
        if any(s.clip_key for s in self.scripts):
            keys = clip_service.expected_keys(specs)
            for script, key in zip(self.scripts, keys):
                if script.clip_path:
                    script.clip_current = clip_service.is_current(script.clip_key, key)
        return self
