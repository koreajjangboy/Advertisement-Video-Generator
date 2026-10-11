"""장면 정보 → 세로형(1080x1920) 숏폼 영상(.mp4) 렌더링.

씬 배경은 두 종류
- AI 씬 영상 (scene.video_path): Luma Dream Machine이 JB 포즈 이미지에서 생성한 캐릭터 연기 클립
  (services/luma_service.py). 씬 길이에 맞게 정규화(1080x1920·30fps, 짧으면 앞뒤 왕복·살짝 느리게)한 뒤 그대로
  배경으로 쓰고, 캐릭터가 영상 안에서 직접 연기하므로 캐릭터 카드·이모지·입자는 얹지 않는다. 자막은 영화 자막처럼
  하단 1/3(얼굴을 가리지 않는 위치)에 배치 (LAYOUTS["cinema"]).
- 기존 방식: 업로드 이미지/그라데이션 + 켄 번즈 + 입자 + 캐릭터 카드(호스트) — AI 클립이 없는 씬의 대체(Fallback)
두 종류가 섞여도 전환·제목 바·자막·BGM 덕킹은 같은 체인에서 합성된다.

화면 구성 (위에서부터)
- 상단 고정 제목 바: 훅 문구 + 도서명 — 영상 전체에 고정 (RENDER_TITLE_BAR)
- 키워드 스티커: 씬의 화면 자막(on_screen_text)을 기울어진 컬러 박스로, '쿵' 하고 튀어나오는 팝 애니메이션
- 이모지 일러스트 스티커: 씬 내용에 맞는 이모지(대본의 emoji, 비면 키워드 규칙으로 자동 선택)를 흰 테두리
  다이컷 스티커로 — 키워드 뒤에 순차로 튀어나온 뒤 계속 둥실거리며 살짝 흔들림
- 하단 자막: 내레이션을 짧은 구절로 끊어 말하는 속도에 맞춰 교체, 지금 말하는 단어는 노란색 (가라오케)
- 배경: 업로드 이미지 또는 섹션 색 그라데이션(빛 번짐·그레인) 위에 씬마다 다른 카메라 무빙
  (줌인/줌아웃/좌우·상하 팬 + 씬 시작 순간의 펀치 줌, 이징) + 속도가 다른 입자 2겹이 떠오르는 패럴랙스
- 씬 전환: xfade (씬마다 다른 효과) + 짧은 '휙' 효과음

구현
1. Pillow로 배경 PNG, 제목 PNG, 그리고 씬별 '텍스트 트랙'을 만든다. 텍스트 트랙은 프레임 단위로
   화면 상태(키워드 팝 크기, 현재 자막 구절)가 바뀌는 지점마다 PNG를 저장하고 ffconcat 목록(파일+표시 시간)으로
   묶는다. → 애니메이션을 Python에서 정확히 제어하고, 한글 폰트·줄바꿈 문제를 피한다.
2. FFmpeg 한 번의 filter_complex: 씬별 [zoompan 배경 + 텍스트 트랙 overlay] → xfade 체인 → 제목 overlay,
   오디오는 씬 시작 시각에 맞춰 adelay 후 amix → 라우드니스 정규화(-14 LUFS) → H.264/AAC mp4.

타이밍: 씬 길이 = 전환 진입(절반) + 내레이션 길이 + RENDER_SCENE_PADDING_SEC + 전환 겹침.
자막 구절의 표시 시간은 글자 수(문장부호 뒤 쉼 가중치 포함)에 비례해 내레이션 길이에 배분한다
(단어별 타임스탬프가 없으므로 근사치).
"""

import logging
import math
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from config import (
    BGM_DUCKING, BGM_VOLUME, FFMPEG_PATH, IMAGE_DIR, RENDER_CRF, RENDER_FONT_PATH, RENDER_KEN_BURNS, RENDER_MOTION_STRENGTH,
    RENDER_NARRATION_CAPTION, RENDER_PRESET, RENDER_SCENE_PADDING_SEC, RENDER_TIMEOUT_SEC, RENDER_TITLE_BAR,
    RENDER_TRANSITION, RENDER_TRANSITION_SEC, RENDER_TRANSITION_SFX, TTS_STYLE, VIDEO_DIR,
)

from services import bgm_service, host_service
from services.ffconcat import StateTrack, write_ffconcat
from services.narration import sanitize_narration

logger = logging.getLogger(__name__)

WIDTH, HEIGHT, FPS = 1080, 1920, 30
SAMPLE_RATE = 44100
NARRATION_FADE_IN = 0.01   # 내레이션 시작 '틱' 방지
NARRATION_FADE_OUT = 0.08  # 내레이션 끝 여음 페이드 아웃
MIN_SCENE_SEC = 1.0
END_HOLD_SEC = 0.5        # 마지막 씬(CTA) 뒤 여유
MAX_IMAGE_SIDE = 2880

# ---- 카메라 무빙 ----
KB_SCALE = 2.0            # 배경을 2배 해상도로 만들어 zoompan의 정수 좌표 떨림을 줄인다
KB_ZOOM = 0.22            # 씬 동안 확대/축소 폭 (RENDER_MOTION_STRENGTH 배수 적용)
PUNCH_ZOOM = 0.10         # 씬 시작 순간 확대됐다가 빠르게 풀리는 펀치 줌
PUNCH_FRAMES = 9
MOTIONS = ["zoom_in", "pan_right", "zoom_out", "pan_up", "pan_left"]

# ---- 전환 ----
# zoomin/fade 계열은 화면 중앙(이모지 스티커)을 크게 확대하거나 색이 뿌옇게 섞여 제외 — 밀기·원형 계열만
AUTO_TRANSITIONS = ["smoothup", "slideleft", "circleopen", "smoothleft", "slideup", "circleclose"]

# ---- 레이아웃 (YouTube Shorts / Reels 안전 영역 고려: 상단 ~150px, 하단 ~380px, 오른쪽 버튼 영역) ----
TITLE_TOP = 150
BOOK_TITLE_SIZE = 72      # 제목 바 아래 도서명 글자 크기 (긴 제목은 한 줄에 들어가게 자동 축소)
KEYWORD_TITLE_GAP = 130   # 상단 고정 레이어 끝과 키워드 스티커 중심 사이 최소 거리 (겹침 방지)
KEYWORD_Y = 600
EMOJI_Y = 935
CAPTION_Y = 1360
# 호스트 모드별 배치 (emoji_x: 이모지 중심 x, emoji_scale: 이모지 크기 배수)
# character 모드: 화면 아래 1/3은 연기하는 캐릭터 카드 → 자막은 키워드와 카드 사이(화면 중앙)에
#                 캐릭터 얼굴을 가리키는 꼬리가 달린 말풍선으로, 이모지는 작게 줄여 카드 위 모서리에 붙은 스티커처럼
#                 (캐릭터가 자리를 옮기면 말풍선 꼬리·이모지도 따라감)
LAYOUTS = {
    "character": {"keyword_y": 610, "caption_y": 960, "emoji_x": 815, "emoji_y": 1135, "emoji_scale": 0.62,
                  "emoji_on_host": True, "speech_bubble": True},
    "avatar": {"keyword_y": KEYWORD_Y, "caption_y": CAPTION_Y, "emoji_x": 540 + 95, "emoji_y": EMOJI_Y,
               "emoji_scale": 1.0},
    "off": {"keyword_y": KEYWORD_Y, "caption_y": CAPTION_Y, "emoji_x": 540, "emoji_y": EMOJI_Y, "emoji_scale": 1.0},
    # AI 씬 영상: 캐릭터 얼굴이 화면 위쪽 1/3~중앙에 오므로 키워드·자막은 영화 자막처럼 하단 1/3에
    "cinema": {"keyword_y": 1170, "caption_y": 1425, "emoji_x": 540, "emoji_y": EMOJI_Y, "emoji_scale": 1.0},
}
CLIP_MAX_SLOWDOWN = 1.25  # AI 클립이 씬보다 짧을 때 허용하는 최대 감속 (그 이상은 앞뒤 왕복 후 마지막 프레임 유지)
CAPTION_MAX_W = 840
CAPTION_MAX_CHARS = 9     # 자막 한 구절 최대 글자 수 (공백 포함) — 2~3어절, 한 줄에 크게
KEYWORD_POP = [0.55, 0.85, 1.12, 1.05, 1.0]   # 프레임별 크기 (튀어나왔다 자리 잡기)
CAPTION_POP = [0.82, 1.06, 1.0]
EMOJI_POP = [0.25, 0.6, 1.0, 1.18, 1.1, 0.97, 1.0]
EMOJI_DELAY = 5           # 키워드 스티커 뒤 이모지가 등장하기까지의 프레임 (순차 등장)
EMOJI_SIZE = 330          # 메인 이모지 크기(px)
EMOJI_CANVAS = 620        # 이모지 트랙 캔버스 (팝 확대·회전 여유 포함)
ACTIVE_WORD_COLOR = (255, 221, 51, 255)  # 자막에서 지금 말하는 단어 (가라오케 강조)
BUBBLE_FILL = (14, 14, 28, 178)
BUBBLE_TAIL_H = 46

# ---- 배경 입자 (2겹 패럴랙스, 위로 떠오름) ----
PARTICLE_LAYERS = [  # (개수, 반지름 범위, 불투명도 범위, 상승 속도 px/s, 블러)
    (70, (3, 7), (60, 140), 45, 1.5),
    (16, (14, 34), (35, 80), 110, 6),
]

EMOJI_FONT_CANDIDATES = [
    "C:/Windows/Fonts/seguiemj.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]
# 이모지가 지정되지 않은 씬: 내레이션·자막 키워드로 고른다 (앞쪽 규칙 우선)
EMOJI_RULES = [
    (("래퍼", "랩", "힙합", "마이크", "무대"), "🎤"),
    (("비트", "음악", "노래", "리듬", "플로우"), "🎧"),
    (("노트", "가사", "적었", "쓰", "펜", "기록"), "✍"),
    (("책", "읽", "서점", "도서", "페이지"), "📚"),
    (("링크", "프로필", "지금"), "👉"),
    (("팔로우", "구독", "알림"), "🔔"),
    (("밤", "새벽", "어둠"), "🌙"),
    (("꿈", "희망", "빛"), "✨"),
    (("실패", "상처", "아픔", "무너"), "💔"),
    (("눈물", "울"), "😢"),
    (("열정", "불", "뜨거"), "🔥"),
    (("돈", "가난", "월세"), "💸"),
    (("시간", "기다", "오래"), "⏳"),
    (("마음", "사랑", "진심"), "❤"),
    (("살아", "버텨", "견디", "성장"), "🌱"),
    (("성공", "정상", "이겼", "증명"), "🏆"),
    (("생각", "고민", "기억"), "💭"),
    (("왜", "?", "무엇", "어떻게"), "🤔"),
    (("떨", "두려", "무서", "긴장"), "😰"),
]
SECTION_EMOJI = {"hook": "🤔", "body": "💭", "cta": "📚"}

YELLOW = (255, 221, 51)
STICKER_STYLES = {  # (박스 색, 글자 색)
    "hook": (YELLOW, (17, 17, 17)),
    "body": (YELLOW, (17, 17, 17)),
    "cta": ((255, 59, 92), (255, 255, 255)),
}
SECTION_GRADIENTS = {  # (위, 아래, 빛 번짐 색들)
    "hook": ((76, 29, 149), (12, 6, 30), [(236, 72, 153), (139, 92, 246), (250, 204, 21)]),
    "body": ((12, 74, 110), (6, 12, 28), [(56, 189, 248), (99, 102, 241), (45, 212, 191)]),
    "cta": ((127, 29, 29), (20, 6, 10), [(251, 113, 133), (250, 204, 21), (249, 115, 22)]),
}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


@dataclass
class RenderScene:
    section: str
    on_screen_text: str
    narration: str
    duration_sec: float
    audio_path: Path | None = None
    audio_duration_sec: float | None = None
    image_path: Path | None = None
    emoji: str = ""
    host_pose: str = ""   # 호스트 캐릭터 포즈 (비면 대본으로 자동 매칭)
    order: int = 0        # 씬 순서 (자동 포즈 매칭 동점 처리)
    video_path: Path | None = None  # AI 생성 씬 영상 (있으면 배경 이미지·캐릭터 카드 대신 사용)

    @property
    def has_clip(self) -> bool:
        return bool(self.video_path and self.video_path.exists())

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_path and self.audio_path.exists() and self.audio_duration_sec)

    @property
    def speech_sec(self) -> float:
        """내레이션(또는 음성이 없을 때 자막을 읽을 시간) 길이."""
        if self.has_audio:
            return self.audio_duration_sec
        return max(self.duration_sec or 0, MIN_SCENE_SEC)


@dataclass
class _Timing:
    frames: int          # 씬 클립 전체 프레임 (앞뒤 전환 겹침 포함)
    start: float         # 최종 영상에서 씬 클립이 시작하는 시각(초)
    lead: int            # 클립 시작 → 내레이션 시작 프레임 (전환 진입 구간)
    speech: int          # 내레이션 프레임 수
    settle: int = 0      # 전환이 끝나는 프레임 (키워드 팝은 이 뒤에 시작해야 묻히지 않음)
    chunks: list[tuple[str, int, int]] = field(default_factory=list)  # (자막 구절, 시작, 끝 프레임)
    # 구절별 단어 타이밍 [(단어 시작 프레임, ...)] — 가라오케 강조용
    word_starts: list[list[int]] = field(default_factory=list)


# ---------- 환경 점검 ----------
@lru_cache
def find_ffmpeg() -> str | None:
    """FFMPEG_PATH → imageio-ffmpeg 번들 → 시스템 PATH 순으로 ffmpeg 실행 파일을 찾는다.
    번들을 우선해 시스템에 설치된 ffmpeg의 버전·빌드 옵션(libx264 유무 등)과 관계없이 같은 결과를 낸다."""
    if FFMPEG_PATH:
        return FFMPEG_PATH if Path(FFMPEG_PATH).exists() else shutil.which(FFMPEG_PATH)
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        logger.warning("imageio-ffmpeg 번들을 사용할 수 없어 시스템 ffmpeg를 찾습니다.")
    return shutil.which("ffmpeg")


def ffmpeg_available() -> bool:
    return find_ffmpeg() is not None


@lru_cache
def find_font() -> str | None:
    for candidate in filter(None, [RENDER_FONT_PATH, *FONT_CANDIDATES]):
        if Path(candidate).exists():
            return candidate
    return None


def check_ready() -> None:
    """렌더링 전제 조건 확인 — 실패 시 사용자에게 보여줄 메시지로 RuntimeError."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg를 찾을 수 없습니다. `pip install imageio-ffmpeg`로 설치하거나 "
                           ".env의 FFMPEG_PATH를 설정하세요.")
    if not find_font():
        raise RuntimeError("한글 자막 폰트를 찾을 수 없습니다. .env의 RENDER_FONT_PATH에 .ttf/.otf 경로를 지정하세요.")


# ---------- 렌더링 ----------
def render_episode(episode_id: int, scenes: list[RenderScene], brand: str = "", title: str | None = None) -> Path:
    """씬 목록을 세로형 mp4로 렌더링해 media/video/episode_{id}_{시각}.mp4로 저장하고 경로를 반환.
    title: 상단 고정 제목 (기본값: 훅 씬의 화면 자막)."""
    check_ready()
    if not scenes:
        raise ValueError("렌더링할 씬이 없습니다.")
    if title is None:
        title = next((s.on_screen_text for s in scenes if s.section == "hook"), scenes[0].on_screen_text)
    title = (title or "").strip() if RENDER_TITLE_BAR else ""

    out_path = VIDEO_DIR / f"episode_{episode_id}_{int(time.time() * 1000)}.mp4"
    tmp_path = out_path.with_name(out_path.stem + ".tmp.mp4")
    started = time.time()
    timings, total_frames, transition_frames = _timeline(scenes)

    # 작업 폴더(배경·자막 PNG, 정규화 클립 등)는 성공·실패와 관계없이 블록을 벗어나면 삭제.
    # Windows에서 파일이 잠깐 잠겨 있어도 예외로 렌더링 결과를 망치지 않도록 정리 오류는 무시 (남은 폴더는 시작 시 정리)
    with tempfile.TemporaryDirectory(prefix=f"render_ep{episode_id}_", ignore_cleanup_errors=True) as workdir:
        work = Path(workdir)
        overlay, title_bottom = _title_layer(title, brand)
        if overlay:
            overlay.save(work / "title.png")
        host_mode = host_service.mode()
        # 상단 고정 레이어(제목 바 + 책 제목 배지)가 길어지면 키워드 스티커를 그만큼 아래로
        layout = _below_title(LAYOUTS[host_mode], title_bottom)
        cinema_layout = _below_title(LAYOUTS["cinema"], title_bottom)
        # 캐릭터 연기 계획: 씬별 감정 비트(포즈) + 서는 자리
        acting = host_service.plan_acting(
            [(s.section, s.on_screen_text, s.narration, s.order, s.host_pose) for s in scenes])
        has_emoji = []
        clip_shade = None
        for i, (scene, timing) in enumerate(zip(scenes, timings)):
            hide_keyword = bool(title) and scene.on_screen_text.strip() == title
            if scene.has_clip:  # AI 씬 영상: 정규화한 클립 + 영화 자막 배치, 카드·이모지 없음
                if clip_shade is None:
                    clip_shade = work / "clip_shade.png"
                    _clip_shade().save(clip_shade)
                _prepare_clip(scene.video_path, work / f"clip{i}.mp4", timing.frames, clip_shade)
                _write_text_track(work, i, scene, timing, cinema_layout, hide_keyword=hide_keyword)
                has_emoji.append(False)
                continue
            _background(scene, i).save(work / f"bg{i}.png", compress_level=1)
            speaker_x = acting[i].speaker_x if acting[i] and layout.get("speech_bubble") else None
            _write_text_track(work, i, scene, timing, layout, hide_keyword=hide_keyword, speaker_x=speaker_x)
            has_emoji.append(_write_emoji_track(work, i, scene, timing, layout["emoji_scale"]))
        for section in {s.section for s in scenes if not s.has_clip}:
            for layer in range(len(PARTICLE_LAYERS)):
                _particle_layer(section, layer).save(work / f"p_{section}_{layer}.png", compress_level=1)
        # 호스트: 감정 비트마다 포즈 교체(씬 전환 한가운데 / 씬 안에서는 그 문장이 시작될 때),
        # 자막 구절마다 홉(핵심 키워드 구절은 크게), 내레이션 음량에 반응, 씬마다 자리 이동
        starts = [round(t.start * FPS) for t in timings]
        beats, hops = _acting_cues(scenes, timings, starts, acting)
        if all(s.has_clip for s in scenes):
            host_mode = "off"  # 모든 씬에서 캐릭터가 AI 영상 안에서 직접 연기 → 카드 불필요
        host_mode = host_mode if host_mode == "off" else host_service.write_host_track(
            work, find_ffmpeg(), find_font(), FPS, total_frames,
            speech=[(s.audio_path, start + t.lead) for s, t, start in zip(scenes, timings, starts) if s.has_audio],
            emphasis=[(start + t.settle, start + t.frames) for s, t, start in zip(scenes, timings, starts)
                      if s.section == "cta"],
            beats=beats, hops=hops,
        )
        host_x = None
        if host_mode == "character" and all(acting):
            host_x = host_service.character_x_expr([(t.start, a.x) for t, a in zip(timings, acting)])
        emoji_dx = [a.emoji_dx if a else 0 for a in acting]
        bgm = bgm_service.resolve_bgm()
        cmd = _ffmpeg_command(scenes, timings, total_frames, transition_frames, work, tmp_path,
                              has_title=overlay is not None, has_emoji=has_emoji, host_mode=host_mode,
                              layout=layout, bgm=bgm, host_x=host_x, emoji_dx=emoji_dx,
                              host_enable=_card_scene_ranges(scenes, timings, total_frames, transition_frames))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=RENDER_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg 렌더링 시간 초과 ({RENDER_TIMEOUT_SEC}초)")
    if result.returncode != 0 or not tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
        logger.error("ffmpeg 실패 (episode=%s):\n%s", episode_id, result.stderr[-4000:])
        raise RuntimeError(f"ffmpeg 렌더링 실패:\n{result.stderr.strip()[-1500:]}")
    tmp_path.replace(out_path)
    logger.info("렌더링 완료: %s (씬 %d개 중 AI 영상 %d개, %.1fs 영상, %.1fs 소요, 호스트 %s, BGM %s)",
                out_path.name, len(scenes), sum(s.has_clip for s in scenes), total_frames / FPS,
                time.time() - started, host_mode, bgm.name if bgm else "off")
    return out_path


def _card_scene_ranges(scenes, timings, total_frames: int, ft: int) -> str | None:
    """캐릭터 카드를 보여줄 구간 (AI 영상이 아닌 씬들)의 ffmpeg enable 식. 모든 씬이 카드 씬이면 None (항상 표시).
    씬 경계는 전환 구간의 한가운데."""
    if not any(s.has_clip for s in scenes):
        return None
    half = ft / 2 / FPS
    ranges = []
    for i, (scene, t) in enumerate(zip(scenes, timings)):
        if scene.has_clip:
            continue
        start = t.start + half if i else 0.0
        end = timings[i + 1].start + half if i + 1 < len(timings) else total_frames / FPS + 1
        ranges.append(f"between(t,{start:.3f},{end:.3f})")
    return "+".join(ranges) or "0"


# ---------- AI 씬 영상 정규화 ----------
def _probe_duration(path: Path) -> float | None:
    result = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None


def _prepare_clip(src: Path, dst: Path, frames: int, shade: Path) -> None:
    """AI 클립 → 정확히 frames 프레임의 1080x1920·30fps 영상 (가독성 그라데이션 포함, 오디오 없음).
    씬보다 짧으면: 최대 CLIP_MAX_SLOWDOWN배까지 느리게 → 그래도 짧으면 정방향+역방향 왕복 → 남으면 마지막 프레임 유지."""
    need = frames / FPS
    duration = _probe_duration(src) or 5.0
    pingpong = need > duration * CLIP_MAX_SLOWDOWN
    if pingpong:
        duration *= 2
    speed = min(max(need / duration, 1.0), CLIP_MAX_SLOWDOWN)
    head = "[0:v]split[f][b];[b]reverse[r];[f][r]concat=n=2:v=1:a=0," if pingpong else "[0:v]"
    graph = (f"{head}setpts={speed:.4f}*(PTS-STARTPTS),"
             f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,crop={WIDTH}:{HEIGHT},"
             f"fps={FPS},setsar=1[v];[v][1:v]overlay=0:0,tpad=stop_mode=clone:stop_duration={need + 1:.3f},"
             f"trim=end_frame={frames},format=yuv420p[out]")
    cmd = [find_ffmpeg(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(src),
           "-loop", "1", "-i", str(shade), "-filter_complex", graph, "-map", "[out]", "-an",
           "-c:v", "libx264", "-preset", RENDER_PRESET, "-crf", "16", "-r", str(FPS), str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=RENDER_TIMEOUT_SEC)
    if result.returncode != 0 or not dst.exists():
        raise RuntimeError(f"AI 씬 영상 변환 실패 ({src.name}): {result.stderr.strip()[-800:]}")


def _clip_shade() -> Image.Image:
    """AI 영상 위 가독성 그라데이션: 위(제목 바)와 아래(자막)만 어둡게, 얼굴이 있는 가운데는 그대로."""
    return _alpha_column((WIDTH, HEIGHT), [(0.0, 150), (0.22, 40), (0.3, 0), (0.55, 0), (0.68, 120), (1.0, 210)])


def _acting_cues(scenes, timings, starts, acting) -> tuple[list[tuple[int, str, str]], list[tuple[int, int]]]:
    """연기 계획 → 최종 타임라인 프레임 기준 (포즈 비트 [(프레임, 포즈, 감정)], 홉 [(프레임, 세기)])."""
    beats, hops = [], []
    for i, (scene, t, start, act) in enumerate(zip(scenes, timings, starts, acting)):
        if act is None:
            continue
        for beat in act.beats:
            beats.append((start + (t.lead if i else 0) + round(t.speech * beat.at), beat.pose, beat.emotion))
        keywords = [w for w in re.findall(r"\w{2,}", scene.on_screen_text)]
        for text, chunk_start, _ in t.chunks:
            strong = any(w[:2] in text for w in keywords)  # 어미·조사가 달라도 앞 두 글자로 비교
            hops.append((start + chunk_start, 2 if strong else 1))
    return beats, hops


def _transition_name(i: int) -> str | None:
    if RENDER_TRANSITION == "none":
        return None
    if RENDER_TRANSITION == "auto":
        return AUTO_TRANSITIONS[i % len(AUTO_TRANSITIONS)]
    return RENDER_TRANSITION


def _timeline(scenes: list[RenderScene]) -> tuple[list[_Timing], int, int]:
    """씬별 프레임 수·시작 시각·자막 구절 타이밍 계산. 전환 구간만큼 이웃 씬과 겹친다."""
    n = len(scenes)
    ft = round(RENDER_TRANSITION_SEC * FPS) if n > 1 and _transition_name(0) else 0
    pad = round(RENDER_SCENE_PADDING_SEC * FPS)
    timings, start = [], 0
    for i, scene in enumerate(scenes):
        lead = ft // 2 if i > 0 else 0
        speech = math.ceil(scene.speech_sec * FPS)
        tail = ft if i < n - 1 else round(END_HOLD_SEC * FPS)
        frames = lead + speech + pad + tail
        timing = _Timing(frames=frames, start=start / FPS, lead=lead, speech=speech, settle=ft if i > 0 else 0)
        if RENDER_NARRATION_CAPTION:
            # 자막도 음성과 같은 '순수 대사' 기준 (편집 중 들어간 연출 지시가 화면에 뜨지 않게)
            timing.chunks = _time_chunks(_caption_chunks(sanitize_narration(scene.narration, (TTS_STYLE,))),
                                         lead, speech)
            timing.word_starts = [_word_starts(text, s, e) for text, s, e in timing.chunks]
        timings.append(timing)
        start += frames - ft
    return timings, start + ft, ft


def _ffmpeg_command(scenes, timings, total_frames, ft, work: Path, out_path: Path, has_title: bool,
                    has_emoji: list[bool], host_mode: str = "off", layout: dict | None = None,
                    bgm: Path | None = None, host_x: str | None = None,
                    emoji_dx: list[int] | None = None, host_enable: str | None = None) -> list[str]:
    layout = layout or LAYOUTS[host_mode]
    inputs: list[str] = []
    vf: list[str] = []
    voice_labels: list[str] = []   # 내레이션 (BGM 덕킹 기준)
    sfx_labels: list[str] = []
    af: list[str] = []
    total_sec = f"{total_frames / FPS:.3f}"

    def add_input(*args: str) -> int:
        add_input.n += 1
        inputs.extend(args)
        return add_input.n
    add_input.n = -1

    # 오디오 바탕(무음, 전체 길이) — amix 길이 기준
    base = add_input("-f", "lavfi", "-t", total_sec, "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo")
    voice_labels.append(f"[{base}:a]")

    for i, (scene, t) in enumerate(zip(scenes, timings)):
        clip_sec = f"{t.frames / FPS:.3f}"
        # 배경: AI 씬 영상(이미 정규화됨) 또는 이미지 + 카메라 무빙
        if scene.has_clip:
            bg = add_input("-i", str(work / f"clip{i}.mp4"))
            vf.append(f"[{bg}:v]setpts=PTS-STARTPTS,setsar=1[bg{i}]")
        elif RENDER_KEN_BURNS:
            bg = add_input("-i", str(work / f"bg{i}.png"))  # 단일 프레임 → zoompan이 frames 장으로 늘림
            z, x, y = _motion(i, t)
            vf.append(f"[{bg}:v]zoompan=z='{z}':x='{x}':y='{y}':d={t.frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
                      f"setsar=1[bg{i}]")
        else:
            bg = add_input("-loop", "1", "-framerate", str(FPS), "-t", clip_sec, "-i", str(work / f"bg{i}.png"))
            vf.append(f"[{bg}:v]scale={WIDTH}:{HEIGHT},setsar=1[bg{i}]")
        layer = f"bg{i}"
        # 떠오르는 입자 2겹 (서로 다른 속도 → 깊이감), 씬마다 위상을 달리해 반복이 티 나지 않게
        for k, (_, _, _, speed, _) in enumerate(PARTICLE_LAYERS if not scene.has_clip else []):
            p = add_input("-loop", "1", "-framerate", str(FPS), "-t", clip_sec, "-i",
                          str(work / f"p_{scene.section}_{k}.png"))
            phase = (i * 677 + k * 331) % HEIGHT
            vf.append(f"[{layer}][{p}:v]overlay=x=0:y='-mod(t*{speed}+{phase},{HEIGHT})'[p{i}_{k}]")
            layer = f"p{i}_{k}"
        # 이모지 스티커: 팝 등장(트랙) + 둥실거림·흔들림(식)
        # emoji_on_host 레이아웃은 호스트 카드 위에 붙여야 하므로 씬 클립이 아니라 최종 타임라인에서 합성(아래)
        if has_emoji[i] and not layout.get("emoji_on_host"):
            e = add_input("-f", "concat", "-safe", "0", "-i", str(work / f"emoji{i}.ffconcat"))
            vf.append(f"[{e}:v]format=rgba,rotate=a='0.07*sin(2*PI*t/1.9)':c=none[em{i}]")
            vf.append(f"[{layer}][em{i}]overlay=x='{layout['emoji_x']}-w/2+16*sin(2*PI*t/2.3)'"
                      f":y='{layout['emoji_y']}-h/2+20*sin(2*PI*t/1.7)':eof_action=repeat[e{i}]")
            layer = f"e{i}"
        # 텍스트 트랙 (키워드 팝 + 가라오케 자막)
        txt = add_input("-f", "concat", "-safe", "0", "-i", str(work / f"text{i}.ffconcat"))
        vf.append(f"[{layer}][{txt}:v]overlay=0:0:eof_action=repeat,trim=end_frame={t.frames},"
                  f"setpts=PTS-STARTPTS,format=yuv420p,fps={FPS}[v{i}]")
        # 내레이션: 씬 시작 + 전환 진입 시각에 배치
        if scene.has_audio:
            a = add_input("-i", str(scene.audio_path))
            delay = round((t.start + t.lead / FPS) * 1000)
            # 내레이션 앞뒤 짧은 페이드 — 끝이 '툭' 끊기는 팝 노이즈 없이 여음을 부드럽게 마감
            fade_out_st = max(_audio_seconds(scene) - NARRATION_FADE_OUT, 0)
            af.append(f"[{a}:a]aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
                      f"afade=t=in:d={NARRATION_FADE_IN},"
                      f"afade=t=out:st={fade_out_st:.3f}:d={NARRATION_FADE_OUT},"
                      f"adelay={delay}:all=1[a{i}]")
            voice_labels.append(f"[a{i}]")
        # 전환 효과음 (전환 시작 직전부터)
        if i > 0 and ft and RENDER_TRANSITION_SFX:
            s = add_input("-f", "lavfi", "-t", "0.45", "-i", f"anoisesrc=r={SAMPLE_RATE}:c=pink:a=0.5")
            delay = max(round((t.start - 0.08) * 1000), 0)
            af.append(f"[{s}:a]highpass=f=600,lowpass=f=5500,afade=t=in:d=0.18,afade=t=out:st=0.18:d=0.27,"
                      f"volume=0.22,aformat=sample_fmts=fltp:channel_layouts=stereo,adelay={delay}:all=1[sfx{i}]")
            sfx_labels.append(f"[sfx{i}]")

    # 씬 연결: xfade 체인 (전환 없음이면 concat)
    if ft:
        prev = "v0"
        for i in range(1, len(scenes)):
            offset = f"{timings[i].start:.3f}"
            vf.append(f"[{prev}][v{i}]xfade=transition={_transition_name(i - 1)}:duration={ft / FPS:.3f}"
                      f":offset={offset}[x{i}]")
            prev = f"x{i}"
    else:
        vf.append("".join(f"[v{i}]" for i in range(len(scenes))) + f"concat=n={len(scenes)}:v=1:a=0[x0]")
        prev = "x0"
    # 영상 전체에 고정되는 요소는 전환 체인 뒤에 합성 (전환 효과에 같이 휩쓸리지 않도록)
    if has_title:
        title = add_input("-loop", "1", "-framerate", str(FPS), "-t", total_sec, "-i", str(work / "title.png"))
        vf.append(f"[{prev}][{title}:v]overlay=0:0[vt]")
        prev = "vt"
    if host_mode != "off":
        host = add_input("-f", "concat", "-safe", "0", "-i", str(work / "host.ffconcat"))
        pre, hx, hy = host_service.overlay_filter(host_mode, host_x)
        vf.append(f"[{host}:v]{pre}[host]")
        enable = f":enable='{host_enable}'" if host_enable else ""  # AI 영상 씬에서는 카드 숨김
        vf.append(f"[{prev}][host]overlay=x='{hx}':y='{hy}':eof_action=repeat{enable}[vh]")
        prev = "vh"
    if layout.get("emoji_on_host"):
        # 씬별 이모지 트랙을 씬 시작 시각만큼 밀어서, 그 씬 동안만(다음 전환 시작 전까지) 카드 위에 표시
        for i, t in enumerate(timings):
            if not has_emoji[i]:
                continue
            end = timings[i + 1].start if i + 1 < len(timings) else total_frames / FPS
            e = add_input("-f", "concat", "-safe", "0", "-i", str(work / f"emoji{i}.ffconcat"))
            vf.append(f"[{e}:v]format=rgba,rotate=a='0.07*sin(2*PI*t/1.9)':c=none,"
                      f"setpts=PTS+{t.start:.3f}/TB[em{i}]")
            # 캐릭터가 자리를 옮기면 같이 따라가도록 카드 x 식 기준
            ex = f"({host_x})+({emoji_dx[i]})" if host_x and emoji_dx else str(layout["emoji_x"])
            vf.append(f"[{prev}][em{i}]overlay=x='{ex}-w/2+12*sin(2*PI*t/2.3)'"
                      f":y='{layout['emoji_y']}-h/2+14*sin(2*PI*t/1.7)':eof_action=pass"
                      f":enable='between(t,{t.start:.3f},{end:.3f})'[ge{i}]")
            prev = f"ge{i}"
    vf.append(f"[{prev}]format=yuv420p[vout]")

    # 오디오: 내레이션 합치기 → BGM(반복·페이드, 내레이션이 나오면 자동으로 줄어듦) → 효과음 → 라우드니스 정규화
    af.append(f"{''.join(voice_labels)}amix=inputs={len(voice_labels)}:normalize=0:duration=first:"
              f"dropout_transition=0[voice]")
    mix_labels = ["[voice]"]
    if bgm:
        b = add_input("-stream_loop", "-1", "-i", str(bgm))
        fade_out = max(total_frames / FPS - 1.5, 0)
        af.append(f"[{b}:a]aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
                  f"atrim=duration={total_sec},asetpts=PTS-STARTPTS,volume={BGM_VOLUME},"
                  f"afade=t=in:d=0.8,afade=t=out:st={fade_out:.3f}:d=1.5[bgm_raw]")
        if BGM_DUCKING:
            af.append("[voice]asplit=2[voice_mix][voice_key]")
            af.append("[bgm_raw][voice_key]sidechaincompress=threshold=0.015:ratio=8:attack=20:release=400[bgm]")
            mix_labels = ["[voice_mix]"]
        else:
            af.append("[bgm_raw]anull[bgm]")
        mix_labels.append("[bgm]")
    mix_labels += sfx_labels
    af.append(f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:normalize=0:duration=first:"
              f"dropout_transition=0,loudnorm=I=-14:TP=-1.5:LRA=11,aresample={SAMPLE_RATE}[aout]")

    return [
        find_ffmpeg(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
        *inputs,
        "-filter_complex", ";".join(vf + af),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", RENDER_PRESET, "-crf", str(RENDER_CRF),
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE),
        "-t", total_sec, "-movflags", "+faststart",
        str(out_path),
    ]


def _audio_seconds(scene: RenderScene) -> float:
    """실제 오디오 파일 길이 (DB 값이 예전 손상 파일 기준일 수 있으므로 파일에서 다시 읽음)."""
    try:
        from services.tts_service import read_duration

        return read_duration(scene.audio_path) or scene.audio_duration_sec or 0.0
    except Exception:
        return scene.audio_duration_sec or 0.0


def _motion(i: int, t: _Timing) -> tuple[str, str, str]:
    """zoompan 식 (z, x, y). 씬마다 다른 무빙 + 부드러운 이징 + 씬 시작 펀치 줌."""
    strength = max(RENDER_MOTION_STRENGTH, 0)
    zoom = KB_ZOOM * strength
    ease = f"(1-cos(PI*on/{t.frames}))/2"
    punch = f"+{PUNCH_ZOOM * strength:.3f}*pow(max(0,1-max(0,on-{t.lead})/{PUNCH_FRAMES}),2)"
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    motion = MOTIONS[i % len(MOTIONS)]
    if motion == "zoom_in":
        return f"1+{zoom:.3f}*{ease}{punch}", cx, cy
    if motion == "zoom_out":
        return f"1+{zoom:.3f}-{zoom:.3f}*{ease}{punch}", cx, cy
    hold = f"1+{zoom * 0.8:.3f}{punch}"  # 팬은 여유 공간이 필요하므로 일정 배율로 확대한 채 이동
    if motion == "pan_right":
        return hold, f"(iw-iw/zoom)*{ease}", cy
    if motion == "pan_left":
        return hold, f"(iw-iw/zoom)*(1-{ease})", cy
    return hold, cx, f"(ih-ih/zoom)*(1-{ease})"  # pan_up


# ---------- 자막 구절 ----------
def _caption_chunks(text: str) -> list[str]:
    """내레이션을 한눈에 읽히는 짧은 구절로 나눈다 (쉼표·문장 끝에서 우선 끊음)."""
    chunks, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > CAPTION_MAX_CHARS:
            chunks.append(current)
            current = word
        else:
            current = candidate
        if re.search(r"[,.!?…~]$", current) and len(current) >= 4:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _time_chunks(chunks: list[str], lead: int, speech: int) -> list[tuple[str, int, int]]:
    """구절별 표시 구간을 글자 수(+문장부호 뒤 쉼)에 비례해 내레이션 길이에 배분."""
    if not chunks:
        return []
    weights = [len(c.replace(" ", "")) + (2 if re.search(r"[,.!?…]$", c) else 0) for c in chunks]
    total, acc, result = sum(weights), 0, []
    for chunk, w in zip(chunks, weights):
        start = lead + round(speech * acc / total)
        acc += w
        end = lead + round(speech * acc / total)
        display = re.sub(r"[,.]+$", "", chunk)  # 쇼츠 자막 관례: 마침표·쉼표 생략 (?! 는 유지)
        result.append((display, start, max(end, start + 1)))
    return result


def _word_starts(text: str, start: int, end: int) -> list[int]:
    """구절 안 단어별 시작 프레임 (글자 수 비례)."""
    words = text.split()
    weights = [len(w) for w in words]
    total, acc, starts = sum(weights) or 1, 0, []
    for w in weights:
        starts.append(start + round((end - start) * acc / total))
        acc += w
    return starts


# ---------- 텍스트 트랙 (Pillow → ffconcat) ----------
def _write_text_track(work: Path, i: int, scene: RenderScene, t: _Timing, layout: dict,
                      hide_keyword: bool, speaker_x: int | None = None) -> None:
    """프레임마다 화면 상태(키워드 팝 크기, 현재 자막 구절과 팝 크기)를 계산해, 상태가 바뀌는 지점마다
    PNG를 저장하고 표시 시간과 함께 ffconcat 목록으로 기록한다. 같은 상태는 파일을 재사용한다."""
    font_path = find_font()
    keyword = None if hide_keyword or not scene.on_screen_text.strip() else \
        _sticker(scene.on_screen_text.strip(), scene.section, font_path, angle=-3 if i % 2 == 0 else 3)
    captions: dict[tuple[int, int], Image.Image] = {}  # (구절, 강조 단어) → 이미지

    def caption(idx: int, active: int) -> Image.Image:
        if (idx, active) not in captions:
            captions[idx, active] = _caption_image(
                t.chunks[idx][0], font_path, active, None if speaker_x is None else speaker_x - WIDTH / 2)
        return captions[idx, active]

    def state(f: int):
        kw = None
        if keyword is not None and f >= t.settle:
            kw = KEYWORD_POP[min(f - t.settle, len(KEYWORD_POP) - 1)]
        cap = None
        for idx, (_, start, end) in enumerate(t.chunks):
            if start <= f < end:
                active = sum(1 for s in t.word_starts[idx] if s <= f) - 1
                cap = (idx, CAPTION_POP[min(f - start, len(CAPTION_POP) - 1)], active)
                break
        return kw, cap

    track = StateTrack(work, f"t{i}", lambda key: _compose_text_frame(key, keyword, caption, layout))
    for f in range(t.frames):
        track.push(state(f))
    track.write(work / f"text{i}.ffconcat", FPS)


def _compose_text_frame(key, keyword: Image.Image | None, caption, layout: dict) -> Image.Image:
    kw_scale, cap = key
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    if keyword is not None and kw_scale is not None:
        _composite_centered(canvas, keyword, kw_scale, layout["keyword_y"])
    if cap is not None:
        idx, scale, active = cap
        _composite_centered(canvas, caption(idx, active), scale, layout["caption_y"])
    return canvas


# ---------- 이모지 스티커 트랙 ----------
@lru_cache
def find_emoji_font() -> str | None:
    return next((c for c in EMOJI_FONT_CANDIDATES if Path(c).exists()), None)


def suggest_emoji(scene: RenderScene) -> str:
    """이모지가 지정되지 않은 씬: 화면 자막·내레이션 키워드 규칙 → 섹션 기본값."""
    text = f"{scene.on_screen_text} {scene.narration}"
    picked = [emoji for words, emoji in EMOJI_RULES if any(w in text for w in words)]
    return "".join(dict.fromkeys(picked[:2])) or SECTION_EMOJI.get(scene.section, "💭")


def _emoji_chars(text: str) -> list[str]:
    """이모지 문자열을 개별 이모지로 분리. 변형 선택자·ZWJ·피부색은 제거한다 — Pillow 기본 레이아웃(raqm 없음)은
    조합 이모지를 한 글자로 그리지 못하므로 구성 요소(예: 👨‍🎤 → 👨 🎤)로 나눠 그린다."""
    chars = []
    for ch in text:
        cp = ord(ch)
        if cp in (0xFE0F, 0xFE0E, 0x200D, 0x20E3) or 0x1F3FB <= cp <= 0x1F3FF:
            continue
        if cp >= 0x2190 and not (0xAC00 <= cp <= 0xD7A3 or 0x3000 <= cp <= 0x9FFF):
            chars.append(ch)
    return chars


@lru_cache
def _emoji_glyph(ch: str, size: int) -> Image.Image | None:
    """컬러 이모지 한 글자를 투명 배경에 꼭 맞게 렌더링. 폰트에 없는 글자(빈 네모)면 None."""
    font_path = find_emoji_font()
    if not font_path:
        return None
    # 컬러 이모지 폰트는 비트맵 크기가 고정인 경우가 있어(Noto: 109) 실패하면 기본 크기로 그린 뒤 확대
    for font_size in (size, 109):
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except OSError:
            continue
    else:
        return None
    canvas = Image.new("RGBA", (font_size * 2, font_size * 2), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text((font_size, font_size), ch, font=font, embedded_color=True, anchor="mm")
    bbox = canvas.getbbox()
    if bbox is None or _is_missing_glyph(canvas, font, font_size):
        return None
    glyph = canvas.crop(bbox)
    scale = size / max(glyph.size)
    return glyph.resize((max(round(glyph.width * scale), 1), max(round(glyph.height * scale), 1)),
                        Image.Resampling.LANCZOS)


def _is_missing_glyph(rendered: Image.Image, font, font_size: int) -> bool:
    """폰트에 없는 글자는 .notdef(빈 네모)로 그려지므로, 확실히 없는 문자의 렌더링 결과와 비교."""
    probe = Image.new("RGBA", rendered.size, (0, 0, 0, 0))
    ImageDraw.Draw(probe).text((font_size, font_size), "\U0010FFFD", font=font, embedded_color=True, anchor="mm")
    return probe.tobytes() == rendered.tobytes()


def _emoji_sticker(emoji: str) -> Image.Image | None:
    """다이컷 스티커: 메인 이모지(크게) + 보조 이모지(작게, 오른쪽 아래 겹침), 흰 테두리 + 그림자."""
    glyphs = [g for g in (_emoji_glyph(ch, EMOJI_SIZE if n == 0 else EMOJI_SIZE * 5 // 9)
                          for n, ch in enumerate(_emoji_chars(emoji)[:2])) if g is not None]
    if not glyphs:
        return None
    art = Image.new("RGBA", (EMOJI_CANVAS, EMOJI_CANVAS), (0, 0, 0, 0))
    main = glyphs[0]
    cx = EMOJI_CANVAS // 2 - (40 if len(glyphs) > 1 else 0)
    art.alpha_composite(main, (cx - main.width // 2, EMOJI_CANVAS // 2 - main.height // 2))
    if len(glyphs) > 1:
        sub = glyphs[1].rotate(12, resample=Image.Resampling.BICUBIC, expand=True)
        art.alpha_composite(sub, (min(cx + main.width // 2 - sub.width // 2 + 30, EMOJI_CANVAS - sub.width - 30),
                                  EMOJI_CANVAS // 2 + main.height // 2 - sub.height + 10))
    # 흰 테두리: 알파를 팽창시킨 실루엣
    alpha = art.getchannel("A").point(lambda a: 255 if a > 40 else 0)
    outline = alpha.filter(ImageFilter.MaxFilter(23)).filter(ImageFilter.GaussianBlur(1.5))
    border = Image.new("RGBA", art.size, (255, 255, 255, 0))
    border.putalpha(outline)
    shadow = Image.new("RGBA", art.size, (0, 0, 0, 0))
    shadow.putalpha(outline.point(lambda a: a * 120 // 255))
    sticker = Image.new("RGBA", art.size, (0, 0, 0, 0))
    sticker.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(12)), (0, 14))
    sticker.alpha_composite(border)
    sticker.alpha_composite(art)
    return sticker


def _write_emoji_track(work: Path, i: int, scene: RenderScene, t: _Timing, scale: float = 1.0) -> bool:
    """이모지 스티커의 등장 애니메이션(팝)을 ffconcat 트랙으로 기록. 이후 둥실거림·흔들림은 ffmpeg 식으로 처리.
    스티커를 만들 수 없으면(폰트 없음 등) False."""
    sticker = _emoji_sticker(scene.emoji.strip() or suggest_emoji(scene))
    if sticker is None:
        return False
    if scale != 1.0:
        sticker = sticker.resize((round(sticker.width * scale), round(sticker.height * scale)),
                                 Image.Resampling.LANCZOS)
    appear = t.settle + EMOJI_DELAY
    blank = Image.new("RGBA", sticker.size, (0, 0, 0, 0))
    entries = []
    if appear > 0:
        blank.save(work / f"e{i}_blank.png")
        entries.append((f"e{i}_blank.png", appear))
    for k, scale in enumerate(EMOJI_POP):
        name = f"e{i}_{k}.png"
        frame = blank.copy()
        _composite_centered_on(frame, sticker, scale)
        frame.save(work / name, compress_level=1)
        hold = 1 if k < len(EMOJI_POP) - 1 else max(t.frames - appear - len(EMOJI_POP) + 1, 1)
        entries.append((name, hold))
    write_ffconcat(work / f"emoji{i}.ffconcat", entries, FPS)
    return True


def _composite_centered_on(canvas: Image.Image, img: Image.Image, scale: float) -> None:
    if scale != 1.0:
        img = img.resize((max(round(img.width * scale), 1), max(round(img.height * scale), 1)),
                         Image.Resampling.BICUBIC)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(img, (round(canvas.width / 2 - img.width / 2), round(canvas.height / 2 - img.height / 2)))
    canvas.alpha_composite(layer)


# ---------- 배경 입자 (패럴랙스) ----------
def _particle_layer(section: str, layer: int) -> Image.Image:
    """세로로 이어 붙여도 끊김 없는 입자 레이어 (1080 x 3840 — 위로 스크롤하며 반복)."""
    count, (r_min, r_max), (a_min, a_max), _, blur = PARTICLE_LAYERS[layer]
    _, _, glows = SECTION_GRADIENTS.get(section, SECTION_GRADIENTS["body"])
    rng = random.Random(f"{section}-{layer}")
    tile = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    for _ in range(count):
        r = rng.uniform(r_min, r_max)
        x, y = rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT)
        tint = rng.choice([(255, 255, 255), *glows])
        color = tint + (rng.randint(a_min, a_max),)
        for dy in (-HEIGHT, 0, HEIGHT):  # 위아래 경계에 걸친 입자도 이어지도록
            draw.ellipse((x - r, y + dy - r, x + r, y + dy + r), fill=color)
    tile = tile.filter(ImageFilter.GaussianBlur(blur))
    full = Image.new("RGBA", (WIDTH, HEIGHT * 2), (0, 0, 0, 0))
    full.paste(tile, (0, 0))
    full.paste(tile, (0, HEIGHT))
    return full


def _composite_centered(canvas: Image.Image, img: Image.Image, scale: float, center_y: int) -> None:
    if scale != 1.0:
        img = img.resize((max(round(img.width * scale), 1), max(round(img.height * scale), 1)),
                         Image.Resampling.BICUBIC)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(img, (round(WIDTH / 2 - img.width / 2), round(center_y - img.height / 2)))
    canvas.alpha_composite(layer)


def _sticker(text: str, section: str, font_path: str, angle: float) -> Image.Image:
    """키워드 스티커: 둥근 컬러 박스 + 굵은 글씨 + 그림자, 살짝 기울임."""
    box_color, text_color = STICKER_STYLES.get(section, STICKER_STYLES["body"])
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    font, lines = _fit_text(probe, text, font_path, 860, sizes=range(112, 63, -6), max_lines=2)
    line_h = round(font.size * 1.2)
    widest = max(probe.textlength(line, font=font) for line in lines)
    pad_x, pad_y, shadow = 46, 26, 14
    w, h = round(widest + pad_x * 2), line_h * len(lines) + pad_y * 2
    img = Image.new("RGBA", (w + shadow * 4, h + shadow * 4), (0, 0, 0, 0))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((shadow * 2, shadow * 2 + 10, shadow * 2 + w, shadow * 2 + h + 10),
                                         radius=28, fill=(0, 0, 0, 150))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((shadow * 2, shadow * 2, shadow * 2 + w, shadow * 2 + h), radius=28, fill=box_color)
    y = shadow * 2 + pad_y
    for line in lines:
        draw.text((img.width / 2, y + line_h / 2), line, font=font, fill=text_color, anchor="mm")
        y += line_h
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)


def _caption_image(text: str, font_path: str, active: int | None = None,
                   tail_dx: float | None = None) -> Image.Image:
    """하단 자막 구절: 굵은 흰 글씨 + 두꺼운 검은 외곽선 + 부드러운 그림자.
    active: 지금 말하는 단어 인덱스 → 노란색으로 강조 (가라오케). 단어 배치는 active와 무관하게 동일.
    tail_dx: 주어지면 말풍선(반투명 박스 + 흰 테두리)으로 감싸고, 화면 중앙 기준 tail_dx만큼 떨어진
    화자(캐릭터) 쪽으로 꼬리를 단다."""
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    # 한 줄에 들어가는 가장 큰 크기 우선, 안 되면 두 줄 (한 단어만 다음 줄로 넘어가는 것 방지)
    font, lines = _fit_text(probe, text, font_path, CAPTION_MAX_W, sizes=range(88, 61, -4), max_lines=1,
                            ellipsis=False)
    if len(lines) > 1:
        font, lines = _fit_text(probe, text, font_path, CAPTION_MAX_W, sizes=range(76, 51, -4), max_lines=2)
        if len(lines) == 2:
            lines = _split_balanced(probe, " ".join(lines), font)
    stroke = max(font.size // 9, 6)
    line_h = round(font.size * 1.22)
    widest = max(probe.textlength(line, font=font) for line in lines)
    margin = stroke + 24
    img = Image.new("RGBA", (round(widest) + margin * 2, line_h * len(lines) + margin * 2), (0, 0, 0, 0))

    space = probe.textlength(" ", font=font)

    def draw_lines(target, color_of, stroke_fill, dy=0):
        d = ImageDraw.Draw(target)
        y, word_idx = margin, 0
        for line in lines:
            words = line.split()
            x = target.width / 2 - probe.textlength(line, font=font) / 2
            for word in words:
                d.text((x, y + line_h / 2 + dy), word, font=font, fill=color_of(word_idx), anchor="lm",
                       stroke_width=stroke, stroke_fill=stroke_fill)
                x += probe.textlength(word, font=font) + space
                word_idx += 1
            y += line_h

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_lines(shadow, lambda _: (0, 0, 0, 170), (0, 0, 0, 170), dy=8)
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(8)))
    draw_lines(img, lambda k: ACTIVE_WORD_COLOR if k == active else (255, 255, 255, 255), (0, 0, 0, 255))
    return img if tail_dx is None else _speech_bubble(img, margin - 14, tail_dx)


def _speech_bubble(text_img: Image.Image, inset: int, tail_dx: float) -> Image.Image:
    """자막 이미지를 말풍선으로 감싼다. inset: 자막 이미지 가장자리의 여백 중 말풍선 밖으로 뺄 만큼."""
    edge, border = 12, 5
    box_w, box_h = text_img.width - inset * 2, text_img.height - inset * 2
    w, h = box_w + edge * 2, box_h + BUBBLE_TAIL_H + edge * 2
    x0, y0, x1, y1 = edge, edge, edge + box_w, edge + box_h
    tx = w / 2 + max(min(tail_dx, box_w / 2 - 70), -(box_w / 2 - 70))
    lean = 16 if tail_dx > 0 else -16
    shape = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(shape)
    d.rounded_rectangle((x0, y0, x1, y1), radius=min(box_h / 2, 44), fill=255)
    d.polygon([(tx - 28, y1 - 6), (tx + 28, y1 - 6), (tx + lean, y1 + BUBBLE_TAIL_H - edge)], fill=255)
    outline = ImageChops.subtract(shape.filter(ImageFilter.MaxFilter(border * 2 + 1)), shape)
    bubble = Image.new("RGBA", (w, h), BUBBLE_FILL[:3] + (0,))
    bubble.putalpha(shape.point(lambda a: a * BUBBLE_FILL[3] // 255))
    white = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    white.putalpha(outline.point(lambda a: a * 235 // 255))
    bubble.alpha_composite(white)
    bubble.alpha_composite(text_img, (edge - inset, edge - inset))
    return bubble


def _title_layer(title: str, brand: str) -> tuple[Image.Image | None, int]:
    """상단 고정 제목 바: 반투명 검은 박스에 2줄 제목 (첫 줄 흰색, 둘째 줄 노란색) + 도서명 라벨.
    (레이어, 레이어가 끝나는 y)를 반환."""
    if not title and not brand:
        return None, 0
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font_path = find_font()
    y = TITLE_TOP
    if title:
        font, lines = _fit_text(draw, title, font_path, WIDTH - 200, sizes=range(84, 55, -4), max_lines=2)
        if len(lines) == 1 and len(title) > 8:
            lines = _split_balanced(draw, title, font)
        line_h = round(font.size * 1.25)
        box_h = line_h * len(lines) + 56
        draw.rounded_rectangle((60, y, WIDTH - 60, y + box_h), radius=28, fill=(0, 0, 0, 185))
        for n, line in enumerate(lines):
            color = YELLOW if n == len(lines) - 1 and len(lines) > 1 else (255, 255, 255)
            draw.text((WIDTH / 2, y + 28 + line_h * n + line_h / 2), line, font=font, fill=color + (255,),
                      anchor="mm")
        y += box_h + 18
    if brand:
        label = brand if brand.startswith("『") else f"『{brand}』"
        # 한 줄에 들어가는 가장 큰 크기 (기본 BOOK_TITLE_SIZE, 긴 제목은 자동 축소)
        for size in range(BOOK_TITLE_SIZE, 31, -2):
            font = ImageFont.truetype(font_path, size)
            tw = draw.textlength(label, font=font)
            if tw <= WIDTH - 200:
                break
        pill_h = round(font.size * 1.6)
        draw.rounded_rectangle((WIDTH / 2 - tw / 2 - 30, y, WIDTH / 2 + tw / 2 + 30, y + pill_h), radius=pill_h / 2,
                               fill=(255, 255, 255, 40), outline=(255, 255, 255, 120), width=2)
        draw.text((WIDTH / 2, y + pill_h / 2), label, font=font, fill=(255, 255, 255, 235), anchor="mm")
        y += pill_h
    return layer, y


def _below_title(layout: dict, title_bottom: int) -> dict:
    if not title_bottom:
        return layout
    return {**layout, "keyword_y": max(layout["keyword_y"], title_bottom + KEYWORD_TITLE_GAP)}


def _split_balanced(draw, text: str, font) -> list[str]:
    """한 줄에 들어가는 제목도 두 줄로 나눠 둘째 줄을 강조 (가장 균형 잡힌 띄어쓰기 위치)."""
    words = text.split()
    if len(words) < 2:
        return [text]
    best = min(range(1, len(words)), key=lambda k: abs(
        draw.textlength(" ".join(words[:k]), font=font) - draw.textlength(" ".join(words[k:]), font=font)))
    return [" ".join(words[:best]), " ".join(words[best:])]


def _fit_text(draw, text, font_path, max_w, sizes, max_lines, ellipsis=True):
    """가장 큰 글자 크기부터 줄여가며 max_lines 안에 들어가는 크기를 찾는다.
    끝까지 안 맞으면 ellipsis=True는 잘라서 '…', False는 줄 수 초과 상태 그대로 반환."""
    for size in sizes:
        font = ImageFont.truetype(font_path, size)
        lines = _wrap(draw, text, font, max_w)
        if len(lines) <= max_lines:
            return font, lines
    if not ellipsis:
        return font, lines
    lines = lines[:max_lines]
    lines[-1] = lines[-1].rstrip()[:-1] + "…"
    return font, lines


def _wrap(draw, text: str, font, max_w: float) -> list[str]:
    """띄어쓰기 단위로 줄바꿈하고, 한 단어가 너무 길면 글자 단위로 자른다."""
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=font) <= max_w:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = ""
            for ch in word:
                if draw.textlength(current + ch, font=font) > max_w and current:
                    lines.append(current)
                    current = ""
                current += ch
        if current:
            lines.append(current)
    return lines


# ---------- 배경 (Pillow) ----------
def _background(scene: RenderScene, index: int) -> Image.Image:
    scale = KB_SCALE if RENDER_KEN_BURNS else 1.0
    size = (round(WIDTH * scale), round(HEIGHT * scale))
    if scene.image_path and scene.image_path.exists():
        with Image.open(scene.image_path) as img:
            bg = ImageOps.fit(ImageOps.exif_transpose(img).convert("RGB"), size, Image.Resampling.LANCZOS)
    else:
        bg = _placeholder(size, scene.section, index)
    # 가독성: 제목이 있는 위쪽과 자막이 있는 아래쪽을 더 어둡게
    shade = _alpha_column(size, [(0.0, 150), (0.28, 60), (0.5, 50), (0.75, 130), (1.0, 200)])
    return Image.alpha_composite(bg.convert("RGBA"), shade).convert("RGB")


def _placeholder(size, section: str, index: int) -> Image.Image:
    """이미지가 없는 씬: 그라데이션 + 큰 빛 번짐 + 필름 그레인 (카메라 무빙이 눈에 보이도록 질감을 둔다)."""
    top, bottom, glows = SECTION_GRADIENTS.get(section, SECTION_GRADIENTS["body"])
    small_size = (size[0] // 4, size[1] // 4)  # 작은 해상도에서 블러 후 확대 (빠르고 부드러움)
    small = _gradient(small_size, top, bottom).convert("RGBA")
    rng = random.Random(index * 7919 + len(section))
    glow = Image.new("RGBA", small_size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    w, h = small_size
    for color in glows:
        r = rng.uniform(0.25, 0.45) * w
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.15, 0.85) * h
        gd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color + (rng.randint(90, 150),))
    small.alpha_composite(glow.filter(ImageFilter.GaussianBlur(w * 0.12)))
    bg = small.convert("RGB").resize(size, Image.Resampling.BICUBIC)
    grain = Image.effect_noise(size, 28).convert("RGB")
    return Image.blend(bg, grain, 0.05)


def _gradient(size, top, bottom) -> Image.Image:
    w, h = size
    column = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(h - 1, 1)
        column.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))
    return column.resize((w, h))


def _alpha_column(size, stops: list[tuple[float, int]]) -> Image.Image:
    """검은색 세로 그라데이션 레이어 (stops: (위치 0~1, 알파))."""
    w, h = size
    column = Image.new("RGBA", (1, h))
    for y in range(h):
        p = y / max(h - 1, 1)
        for (p0, a0), (p1, a1) in zip(stops, stops[1:]):
            if p0 <= p <= p1:
                alpha = a0 + (a1 - a0) * (p - p0) / (p1 - p0)
                break
        column.putpixel((0, y), (0, 0, 0, round(alpha)))
    return column.resize((w, h))


# ---------- 씬 배경 이미지 업로드 ----------
def save_scene_image(data: bytes, basename: str) -> str:
    """업로드된 이미지를 검증·정규화(회전 보정, RGB, 최대 크기 제한)해 media/images/에 JPEG로 저장.
    /media/images/... URL을 반환한다."""
    from io import BytesIO

    try:
        with Image.open(BytesIO(data)) as img:
            img.load()
            img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise ValueError(f"이미지 파일을 읽을 수 없습니다: {e}") from e
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    path = IMAGE_DIR / f"{basename}_{int(time.time() * 1000)}.jpg"
    img.save(path, "JPEG", quality=92)
    return f"/media/images/{path.name}"
