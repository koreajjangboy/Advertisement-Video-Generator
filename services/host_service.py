"""호스트 캐릭터 — 영상 전체를 이끌며 대본을 '연기'하는 진행자.

두 가지 모드 (HOST_MODE, 기본 auto)
- character: HOST_POSE_DIR(기본 media/images/JB)의 포즈 이미지로 대사를 연기하는 캐릭터 카드
- avatar: HOST_IMAGE 한 장의 HOST_CROP 영역을 원형 아바타로 잘라 화면 왼쪽 중단에 고정

character 모드 연기 (Acting)
1. 감정선 분석: 씬 내레이션을 문장 단위로 나눠 감정(인사·질문·놀람·강조·공감·결론)을 판별하고,
   감정마다 어울리는 포즈를 고른다. 한 씬 안에서도 감정이 바뀌면(예: 질문 → 강조) 그 문장이 시작되는
   순간 포즈를 바꾼다 (씬당 최대 2비트). 바로 앞 포즈와 같은 포즈는 피한다.
2. 무대 동선(블로킹): 포즈마다 서는 자리(왼쪽·가운데·오른쪽)가 있다 — 예: 생각하는 JB는 화면 왼쪽을
   바라보므로 오른쪽에 서서 자막 쪽을 바라본다. 씬이 바뀌면 새 자리로 미끄러지듯 이동한다(오버슈트 이징).
3. 캐릭터 교체 모션: 포즈가 바뀌면 이전 포즈와 크로스페이드하면서 찌그러졌다 튀어 오르는
   스쿼시&스트레치 바운스.
4. 살아 있는 움직임: 카드 안 미세 켄 번즈(얼굴 쪽으로 천천히 다가감), 내레이션 음량에 맞춘 맥동·끄덕임,
   자막 구절이 바뀔 때마다 작은 홉(핵심 키워드 구절은 더 크게), 몸 전체가 천천히 흔들리는 스웨이·숨쉬기.
5. 리액션 효과: 감정마다 만화식 효과(놀람 집중선+!, 질문 ?, 생각 말줄임 점, 강조 반짝임, 인사 물결,
   행동 유도 화살표)가 포즈가 바뀌는 순간 튀어나왔다 사라진다.
6. 카드 연출: 감정 색 테두리 + 흰 테두리 + 그림자, 말할 때 감정 색으로 빛나는 테두리, 이름표 음성 막대.

상태(포즈, 교체 단계, 음량 단계, 줌 단계, 효과 단계 …)가 바뀌는 프레임만 PNG로 저장해 ffconcat 트랙으로 만들고,
위치 이동·스웨이·숨쉬기는 ffmpeg 식으로 처리한다 (씬 전환과 관계없이 전환 체인 뒤에 합성).

포즈 이름 → 파일 매핑은 기본값(DEFAULT_POSE_FILES)을 쓰고, 폴더에 poses.json({"surprised": "00.png", ...})이
있으면 그것을 우선한다.
"""

import json
import logging
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from config import HOST_CROP, HOST_ENABLED, HOST_IMAGE, HOST_MODE, HOST_NAME, HOST_POSE_DIR, MEDIA_DIR
from services.bgm_service import audio_envelope
from services.ffconcat import StateTrack

logger = logging.getLogger(__name__)

GLOW_COLOR = (255, 221, 51)
TAG_COLOR = (255, 59, 92)
BAR_PATTERNS = [(0.5, 1.0, 0.7), (1.0, 0.6, 0.9), (0.7, 0.85, 0.5)]
LEVELS = 5                      # 음량 단계 (0 = 조용함)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ---------- 포즈 · 감정 라이브러리 ----------
POSE_LABELS = {
    "surprised": "놀람",
    "thinking": "생각",
    "confident": "확신",
    "pointing": "가리키기",
    "waving": "인사",
}
# media/images/JB 기본 세트: 00 놀람, 01 가리키며 웃음, 02 손 인사, 03 턱 괴고 생각(화면 왼쪽을 봄), 04 자신감
# (05는 두 이미지가 반씩 합성된 컷이라 기본 매핑에서 제외 — poses.json으로 직접 지정 가능)
DEFAULT_POSE_FILES = {"surprised": "00", "pointing": "01", "waving": "02", "thinking": "03", "confident": "04"}


@dataclass(frozen=True)
class EmotionDef:
    label: str
    keywords: tuple[str, ...]    # 문장에 있으면 가산점
    sections: tuple[str, ...]    # 이 섹션이면 가산점
    poses: tuple[str, ...]       # 어울리는 포즈 (선호 순)
    accent: tuple[int, int, int]  # 카드 테두리·빛·효과 색
    fx: str                      # 리액션 효과


EMOTIONS: dict[str, EmotionDef] = {
    "greeting": EmotionDef("인사", ("안녕", "반가", "팔로우", "구독", "만나", "다음에", "또 봐", "함께해", "고마"),
                           ("cta",), ("waving", "pointing"), (52, 211, 153), "wave"),
    "question": EmotionDef("질문", ("왜", "어떻게", "무엇", "뭘", "뭐", "까요", "일까", "을까", "나요", "는지", "할까"),
                           ("hook",), ("surprised", "thinking"), (167, 139, 250), "question"),
    "surprise": EmotionDef("놀람", ("진짜", "설마", "놀라", "충격", "믿기", "몰랐", "처음", "헐", "세상에", "떨", "반전"),
                           ("hook",), ("surprised", "thinking"), (250, 204, 21), "shock"),
    "emphasis": EmotionDef("강조", ("꼭", "반드시", "최고", "핵심", "절대", "이건", "증명", "끝까지", "가장", "살아남",
                                  "해냈", "성공", "아니라", "추천"), ("body",), ("confident", "pointing"),
                           (255, 59, 92), "sparkle"),
    "reflection": EmotionDef("공감·회상", ("기억", "그때", "생각", "마음", "밤", "혼자", "고민", "사실", "적었", "노트",
                                       "기록", "아무도", "눈물"), ("body",), ("thinking", "confident"),
                             (56, 189, 248), "thought"),
    "conclusion": EmotionDef("결론·행동", ("지금", "링크", "프로필", "확인", "바로", "서점", "결국", "그러니", "그래서",
                                       "당신", "여러분"), ("cta",), ("pointing", "confident", "waving"),
                             (251, 146, 60), "action"),
}
EMOTION_ORDER = list(EMOTIONS)
POSE_EMOTION = {"surprised": "surprise", "thinking": "reflection", "confident": "emphasis",
                "pointing": "conclusion", "waving": "greeting"}

# 무대 동선: 포즈별로 서는 자리. JB 포즈들은 손·시선이 화면 왼쪽(자막 쪽)을 향하므로 주로 가운데·오른쪽,
# 확신 포즈는 왼쪽에서 위쪽 자막을 가리킨다.
POSE_SLOTS = {"surprised": "center", "waving": "center", "thinking": "right", "pointing": "right",
              "confident": "left"}
SLOT_X = {"left": 400, "center": 540, "right": 680}
EMOJI_DX = 275                  # 카드 중심 → 위 모서리 이모지 스티커 (가로)
FACE_X = 0.55                   # 원본 이미지에서 얼굴 중심 (폭·높이 비율) — 말풍선 꼬리, 켄 번즈 초점
FACE_Y = 0.34
MAX_BEATS = 2                   # 씬 하나에서 포즈를 바꾸는 최대 횟수 + 1
BEAT_WINDOW = (0.22, 0.82)      # 두 번째 비트가 시작될 수 있는 내레이션 구간 (너무 이르거나 늦으면 생략)


@dataclass(frozen=True)
class Beat:
    at: float                    # 내레이션 안에서 시작하는 위치 (0~1)
    emotion: str
    pose: str


@dataclass(frozen=True)
class SceneActing:
    beats: tuple[Beat, ...]
    explicit: bool               # 대본/대시보드에서 포즈를 직접 지정했는지

    @property
    def pose(self) -> str:
        return self.beats[0].pose

    @property
    def slot(self) -> str:
        return POSE_SLOTS.get(self.pose, "center")

    @property
    def x(self) -> int:
        return SLOT_X[self.slot]

    @property
    def speaker_x(self) -> int:
        """말풍선 꼬리가 향할 화면 x (캐릭터 얼굴)."""
        return round(self.x + (FACE_X - 0.5) * CARD_W)

    @property
    def emoji_dx(self) -> int:
        """카드 중심에서 이모지 스티커까지 가로 거리 — 화면 밖으로 나가지 않는 위 모서리
        (왼쪽 자리는 왼쪽 모서리, 오른쪽 자리는 화면 끝에 걸리지 않게 조금 안쪽)."""
        return {"left": -EMOJI_DX, "right": EMOJI_DX - 60}.get(self.slot, EMOJI_DX)

    def summary(self) -> list[dict]:
        return [{"pose": b.pose, "emotion": b.emotion, "label": EMOTIONS[b.emotion].label, "at": round(b.at, 2)}
                for b in self.beats]


def mode() -> str:
    """실제 사용될 호스트 모드: character | avatar | off."""
    if not HOST_ENABLED or HOST_MODE == "off":
        return "off"
    if HOST_MODE in ("auto", "character") and pose_files():
        return "character"
    if HOST_MODE in ("auto", "avatar") and HOST_IMAGE.exists():
        return "avatar"
    return "off"


def host_enabled() -> bool:
    return mode() != "off"


def pose_files() -> dict[str, Path]:
    """포즈 이름 → 이미지 파일 (존재하는 것만)."""
    if not HOST_POSE_DIR.is_dir():
        return {}
    files = {p.stem: p for p in HOST_POSE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    mapping = DEFAULT_POSE_FILES
    custom = HOST_POSE_DIR / "poses.json"
    if custom.exists():
        try:
            mapping = {k: Path(v).stem for k, v in json.loads(custom.read_text(encoding="utf-8")).items()}
        except Exception:
            logger.warning("poses.json을 읽을 수 없어 기본 포즈 매핑을 사용합니다.")
    return {pose: files[stem] for pose, stem in mapping.items() if pose in POSE_LABELS and stem in files}


def pose_catalog() -> list[dict]:
    """대시보드용 포즈 목록 [{name, label, image_url}]."""
    return [{"name": pose, "label": POSE_LABELS[pose],
             "image_url": "/media/" + path.relative_to(MEDIA_DIR).as_posix() if path.is_relative_to(MEDIA_DIR) else None}
            for pose, path in pose_files().items()]


# ---------- 감정선 분석 → 연기 계획 ----------
def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.findall(r"[^.!?…]+[.!?…]*", text or "") if s.strip(" .!?…")]


def _emotion(text: str, section: str, order: int, first: bool, last: bool) -> str:
    """문장 하나의 감정. 키워드 2점/개 + 섹션 1점 + 문장 부호·위치 가산점, 동점은 씬 순서로 돌려 가며."""
    scores = {e: 2 * sum(k in text for k in d.keywords) + (1 if section in d.sections else 0)
              for e, d in EMOTIONS.items()}
    end = text.rstrip()[-1:] if text.strip() else ""
    if end == "?":
        scores["question"] += 4
    elif end == "!":
        scores["surprise"] += 1
        scores["emphasis"] += 1
    if first:
        scores["surprise"] += 1
    if last and section == "cta":
        scores["greeting"] += 1
    rotation = {e: (k - order) % len(EMOTION_ORDER) for k, e in enumerate(EMOTION_ORDER)}
    return min(EMOTION_ORDER, key=lambda e: (-scores[e], rotation[e]))


def _pose_for(emotion: str, available, avoid: str | None, order: int) -> str:
    """감정에 어울리는 포즈 중 avoid(바로 앞 포즈)가 아닌 첫 번째. 없으면 다른 아무 포즈."""
    preferred = [p for p in EMOTIONS[emotion].poses if p in available]
    names = sorted(available)
    others = sorted(names, key=lambda p: (names.index(p) - order) % len(names))
    for pose in preferred + others:
        if pose != avoid:
            return pose
    return names[0]


def _plan_scene(available, section: str, on_screen: str, narration: str, order: int, explicit: str,
                previous: str | None, first: bool, last: bool) -> SceneActing:
    if explicit in available:
        return SceneActing((Beat(0.0, POSE_EMOTION[explicit], explicit),), True)
    sentences = _sentences(narration) or [on_screen or narration or ""]
    weights = [max(len(s.replace(" ", "")), 1) for s in sentences]
    total, acc, parts = sum(weights), 0, []
    for k, (sentence, w) in enumerate(zip(sentences, weights)):
        text = f"{on_screen} {sentence}" if k == 0 else sentence
        emotion = _emotion(text, section, order, first and k == 0, last and k == len(sentences) - 1)
        if not parts or parts[-1][1] != emotion:
            parts.append((acc / total, emotion))
        acc += w
    beats = [Beat(0.0, parts[0][1], _pose_for(parts[0][1], available, previous, order))]
    for at, emotion in parts[1:]:
        if len(beats) >= MAX_BEATS:
            break
        if BEAT_WINDOW[0] <= at <= BEAT_WINDOW[1]:
            pose = _pose_for(emotion, available, beats[-1].pose, order)
            if pose != beats[-1].pose:
                beats.append(Beat(at, emotion, pose))
    return SceneActing(tuple(beats), False)


def plan_acting(scenes: list[tuple[str, str, str, int, str]]) -> list[SceneActing | None]:
    """씬 순서대로 연기 계획 [(section, on_screen_text, narration, order, 지정 포즈)].
    렌더링과 대시보드가 같은 결과를 쓰도록 한 곳에서 계산한다. 캐릭터 모드가 아니면 None 목록."""
    if mode() != "character":
        return [None] * len(scenes)
    available = pose_files()
    result, previous = [], None
    for k, (section, on_screen, narration, order, explicit) in enumerate(scenes):
        acting = _plan_scene(available, section, on_screen or "", narration or "", order, explicit or "",
                             previous, k == 0, k == len(scenes) - 1)
        previous = acting.beats[-1].pose
        result.append(acting)
    return result


@lru_cache
def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


# ---------- 공용: 이름표 ----------
def _name_tag(draw: ImageDraw.ImageDraw, cx: float, top: float, level: int, phase: int, font_path: str,
              size: int = 34, color: tuple[int, int, int] = TAG_COLOR) -> None:
    font = ImageFont.truetype(font_path, size)
    tw = draw.textlength(HOST_NAME, font=font)
    bars_w = size
    h = round(size * 1.6)
    w = tw + bars_w + 48
    draw.rounded_rectangle((cx - w / 2, top, cx + w / 2, top + h), radius=h / 2, fill=color + (255,),
                           outline=(255, 255, 255, 255), width=4)
    bx = cx - w / 2 + 22
    for k, height in enumerate(BAR_PATTERNS[phase]):
        bh = 6 + (h * 0.42 * height * level / (LEVELS - 1) if level else 0)
        draw.rounded_rectangle((bx + k * 11, top + h / 2 - bh / 2, bx + k * 11 + 6, top + h / 2 + bh / 2),
                               radius=3, fill=(255, 255, 255, 255))
    draw.text((bx + bars_w, top + h / 2), HOST_NAME, font=font, fill=(255, 255, 255, 255), anchor="lm")


def _speech_levels(ffmpeg: str, fps: int, total_frames: int, speech: list[tuple[Path, int]]) -> list[int]:
    """최종 영상 타임라인 기준 프레임별 말하기 음량 단계 (빠르게 올라가고 천천히 가라앉음)."""
    env = [0.0] * total_frames
    for path, offset in speech:
        try:
            values = audio_envelope(ffmpeg, path, fps)
        except Exception as e:
            logger.warning("음량 분석 실패 (%s): %s", path.name, e)
            continue
        for k, v in enumerate(values):
            if 0 <= offset + k < total_frames:
                env[offset + k] = max(env[offset + k], v)
    ranked = sorted(env)
    peak = ranked[int(len(ranked) * 0.97)] or ranked[-1] or 1.0
    levels, current = [], 0.0
    for v in env:
        x = min(v / peak, 1.0)
        current = x if x > current else current * 0.78
        levels.append(round(current * (LEVELS - 1)))
    return levels


def write_host_track(work: Path, ffmpeg: str, font_path: str, fps: int, total_frames: int,
                     speech: list[tuple[Path, int]], emphasis: list[tuple[int, int]],
                     beats: list[tuple[int, str, str]] | None = None,
                     hops: list[tuple[int, int]] | None = None) -> str:
    """호스트 트랙(host.ffconcat) 기록. 사용한 모드(character | avatar | off)를 반환.
    speech: [(내레이션 파일, 시작 프레임)], emphasis: [(강조 시작, 끝 프레임)] (avatar),
    beats: [(포즈가 바뀌는 프레임, 포즈, 감정)] (character), hops: [(자막 구절 시작 프레임, 세기 1|2)] (character)."""
    current = mode()
    if current == "off":
        return "off"
    levels = _speech_levels(ffmpeg, fps, total_frames, speech)
    try:
        if current == "character" and beats:
            _write_character_track(work, font_path, fps, total_frames, levels, beats, hops or [])
        else:
            _avatar()
            _write_avatar_track(work, font_path, fps, total_frames, levels, emphasis)
            current = "avatar"
    except Exception:
        logger.exception("호스트 트랙 생성 실패 — 호스트 없이 렌더링합니다.")
        return "off"
    return current


# ---------- character 모드: 연기하는 캐릭터 카드 ----------
CARD_W = 620                    # 카드 폭
CARD_TOP = 1150                 # 화면에서 카드 윗변 위치 (아래쪽은 화면 밖으로 이어짐)
CARD_VISIBLE_H = 1920 - CARD_TOP
CARD_H = CARD_VISIBLE_H + 60    # 카드 이미지 높이 (튀어 오를 때 아래가 비지 않게 여유)
CARD_RADIUS = 70
CARD_BORDER = 10                # 흰 테두리
CARD_RING = 7                   # 감정 색 바깥 테두리
CARD_SOURCE_TOP = 0.10          # 원본 이미지에서 잘라낼 시작 높이 비율 (머리 위 여백 줄이기)
SIDE, TOP = 170, 170            # 트랙 캔버스에서 카드 바깥 여백 (리액션 효과·이름표·빛)
CHAR_CANVAS = (CARD_W + SIDE * 2, TOP + CARD_H + 40)
CHAR_Y_TOP = CARD_TOP - TOP     # 캔버스 위쪽이 놓이는 화면 y
ENTRY_DELAY_CHAR = 4
ENTRY_RISE = [520, 330, 170, 60, 0, -14, -6, 0]   # 등장: 화면 아래에서 솟아오름 (px)
XFADE_FRAMES = 6                # 포즈 교체 크로스페이드
SQUASH = [(1.07, 0.9, 18), (0.95, 1.07, -16), (1.02, 0.98, -4), (1.0, 1.0, 0)]  # (가로, 세로 배율, 위아래)
HOP = {1: [-7, -12, -8, -3], 2: [-12, -22, -14, -5]}  # 자막 구절 시작 홉 (세기별)
KB_MAX, KB_STEPS = 0.045, 12    # 카드 안 미세 켄 번즈: 비트 동안 얼굴 쪽으로 최대 4.5% 다가감
FX_POP = [0.3, 0.75, 1.15, 1.0]
FX_HOLD = 48                    # 리액션 효과 유지 프레임 (그 뒤 사라짐)
FX_FADE = 5
SLIDE_SEC = 0.5                 # 자리 이동 시간


@lru_cache(maxsize=96)
def _card_content(pose_path: Path, kb: int) -> Image.Image:
    """포즈 이미지를 카드 크기로 — kb 단계만큼 얼굴 쪽으로 확대 (부분 픽셀 box 리샘플링으로 부드럽게)."""
    src = _load_rgb(pose_path)
    scale = CARD_W / src.width
    view_h = CARD_H / scale
    top = src.height * CARD_SOURCE_TOP
    if src.height - top < view_h:  # 원본이 짧으면 위 여백을 덜 자른다
        top = max(src.height - view_h, 0)
    x0, y0, x1, y1 = 0.0, top, float(src.width), top + view_h
    z = 1 + KB_MAX * kb / KB_STEPS
    fx, fy = src.width * FACE_X, min(max(src.height * FACE_Y, y0), y1)
    box = (fx - (fx - x0) / z, fy - (fy - y0) / z, fx + (x1 - fx) / z, fy + (y1 - fy) / z)
    return src.resize((CARD_W, CARD_H), Image.Resampling.BICUBIC, box=box)


@lru_cache
def _card_mask() -> Image.Image:
    mask = Image.new("L", (CARD_W * 2, CARD_H * 2), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, CARD_W * 2 - 1, CARD_H * 2 + CARD_RADIUS * 2),
                                           radius=CARD_RADIUS * 2, fill=255)
    return mask.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)


@lru_cache(maxsize=64)
def _backdrop(level: int, accent: tuple[int, int, int]) -> Image.Image:
    """카드 뒤 그림자 + 빛 (말할 때 감정 색으로 커지고 진해짐). 1/4 크기에서 블러 후 확대."""
    q = 4
    w, h = CHAR_CANVAS[0] // q, CHAR_CANVAS[1] // q
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0, y0 = SIDE / q, TOP / q
    cw = CARD_W / q
    d.rounded_rectangle((x0 - 2, y0 + 6, x0 + cw + 2, h + 40), radius=CARD_RADIUS / q, fill=(0, 0, 0, 150))
    glow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    g = (CARD_BORDER + CARD_RING + 10 + 7 * level) / q
    color = accent + (90 + 38 * level,) if level else (255, 255, 255, 55)
    ImageDraw.Draw(glow_layer).rounded_rectangle((x0 - g, y0 - g, x0 + cw + g, h + 40),
                                                 radius=(CARD_RADIUS + g * q) / q, fill=color)
    layer = layer.filter(ImageFilter.GaussianBlur(22 / q))
    layer.alpha_composite(glow_layer.filter(ImageFilter.GaussianBlur(14 / q)))
    return layer.resize(CHAR_CANVAS, Image.Resampling.BICUBIC)


# ---- 리액션 효과 (만화식 기호) ----
FX_BOX = (240, 300)


def _star(d: ImageDraw.ImageDraw, cx, cy, r, color) -> None:
    pts = []
    for k in range(8):
        rad = r if k % 2 == 0 else r * 0.32
        a = k * math.pi / 4 - math.pi / 2
        pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    d.polygon(pts, fill=color, outline=(30, 30, 30, 255), width=3)


@lru_cache(maxsize=128)
def _fx_image(fx: str, step: int, accent: tuple[int, int, int], side: int, font_path: str) -> Image.Image:
    """리액션 효과 그림 (FX_BOX 크기). side: 캐릭터 기준 효과가 놓이는 쪽 (-1 왼쪽, 1 오른쪽) —
    기울기·방향을 맞추기 위해 사용. step: 반복 애니메이션 단계 (0~3)."""
    w, h = FX_BOX
    img = Image.new("RGBA", FX_BOX, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = accent + (255,)
    dark = (25, 25, 35, 255)
    inner = -side  # 캐릭터(얼굴) 쪽 방향

    def mark(text, x, y, size, angle=0):
        font = ImageFont.truetype(font_path, size)
        layer = Image.new("RGBA", FX_BOX, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((x, y), text, font=font, fill=col, anchor="mm", stroke_width=max(size // 10, 4),
                                   stroke_fill=dark)
        if angle:
            layer = layer.rotate(angle, center=(x, y), resample=Image.Resampling.BICUBIC)
        img.alpha_composite(layer)

    bob = [0, -6, -10, -6][step]
    if fx == "shock":      # 집중선 + 느낌표
        cx, cy = w / 2 + inner * 120, h * 0.55   # 얼굴 쪽 한 점에서 바깥으로 뻗는 선
        for k, a in enumerate((-40, 0, 40)):
            ux, uy = -inner * math.cos(math.radians(a)), math.sin(math.radians(a))
            r0, r1 = 80, 150 + (14 if (k + step) % 2 else 0)
            p0, p1 = (cx + ux * r0, cy + uy * r0), (cx + ux * r1, cy + uy * r1)
            d.line((p0, p1), fill=dark, width=16)
            d.line((p0, p1), fill=col, width=9)
        mark("!", w / 2 - inner * 20, h * 0.42 + bob, 150, angle=8 * inner)
    elif fx == "question":
        mark("?", w / 2 - inner * 10, h * 0.40 + bob, 140, angle=-10 * inner)
        mark("?", w / 2 + inner * 55, h * 0.70 - bob, 82, angle=12 * inner)
    elif fx == "thought":  # 생각 말줄임: 점이 하나씩 차오름
        for k in range(min(step, 2) + 1):
            r = 16 + k * 8
            x, y = w / 2 + inner * 60 - inner * k * 62, h * 0.66 - k * 50
            d.ellipse((x - r - 4, y - r - 4, x + r + 4, y + r + 4), fill=dark)
            d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 255))
    elif fx == "sparkle":
        sizes = [(0.6, 0.30, 52), (0.3, 0.62, 34), (0.72, 0.78, 26)]
        for k, (px, py, r) in enumerate(sizes):
            pulse = 1 + 0.25 * (((step + k) % 4) in (1, 2))
            _star(d, w * (px if side > 0 else 1 - px), h * py, r * pulse, col if k != 1 else (255, 255, 255, 255))
    elif fx == "wave":     # 손 흔드는 물결선
        cx, cy = w / 2 + inner * 90, h * 0.5
        for k in range(3):
            r = 50 + k * 34 + (6 if (step + k) % 2 else 0)
            start = -55 if inner < 0 else 125
            box = (cx - r, cy - r, cx + r, cy + r)
            d.arc(box, start=start, end=start + 110, fill=dark, width=15)
            d.arc(box, start=start, end=start + 110, fill=col, width=8)
    elif fx == "action":   # 자막 쪽(위)으로 향하는 화살표 꺾쇠
        for k in range(3):
            y = h * 0.78 - k * 62 + bob
            x = w / 2 + inner * 15 * k
            alpha = 255 if (k + step) % 4 != 3 else 120
            pts = [(x - 46, y + 22), (x, y - 26), (x + 46, y + 22)]
            d.line(pts, fill=dark, width=24, joint="curve")
            d.line(pts, fill=accent + (alpha,), width=13, joint="curve")
    return img


def _fx_frame(since: int) -> tuple | None:
    """비트 시작 후 since 프레임의 효과 상태 (step, scale, alpha) — 튀어나와 잠시 흔들리다 사라짐."""
    if since < len(FX_POP):
        return 0, FX_POP[since], 255
    if since < FX_HOLD:
        return (since // 6) % 4, 1.0, 255
    if since < FX_HOLD + FX_FADE:
        return (since // 6) % 4, 1.0, round(255 * (1 - (since - FX_HOLD + 1) / (FX_FADE + 1)))
    return None


def _character_frame(key, files: dict[str, Path], font_path: str) -> Image.Image:
    (pose, kb, prev, prev_kb, mix, emotion, level, phase, sx, sy, dy, fx_state, tag_side) = key
    canvas = Image.new("RGBA", CHAR_CANVAS, (0, 0, 0, 0))
    if pose is None:
        return canvas
    accent = EMOTIONS[emotion].accent
    canvas.alpha_composite(_backdrop(level, accent))
    d = ImageDraw.Draw(canvas)
    x0, y0 = SIDE, TOP
    r, b = CARD_RING, CARD_BORDER
    d.rounded_rectangle((x0 - b - r, y0 - b - r, x0 + CARD_W + b + r, CHAR_CANVAS[1] + 100),
                        radius=CARD_RADIUS + b + r, fill=accent + (255,))
    d.rounded_rectangle((x0 - b, y0 - b, x0 + CARD_W + b, CHAR_CANVAS[1] + 100), radius=CARD_RADIUS + b,
                        fill=(255, 255, 255, 255))
    content = _card_content(files[pose], kb)
    if prev is not None and mix is not None:  # 포즈 교체: 이전 포즈에서 크로스페이드
        content = Image.blend(_card_content(files[prev], prev_kb), content, mix / (XFADE_FRAMES + 1))
    canvas.paste(content, (x0, y0), _card_mask())
    if HOST_NAME:
        tag_x = x0 + 130 if tag_side < 0 else x0 + CARD_W - 130
        _name_tag(d, tag_x, y0 - 30, level, phase, font_path, size=32)
    if fx_state is not None:
        fx, step, scale, alpha, side = fx_state
        art = _fx_image(fx, step, accent, side, font_path)
        if scale != 1.0:
            art = art.resize((max(round(art.width * scale), 1), max(round(art.height * scale), 1)),
                             Image.Resampling.BICUBIC)
        if alpha < 255:
            art = art.copy()
            art.putalpha(art.getchannel("A").point(lambda a: a * alpha // 255))
        # 얼굴 높이, 카드 바깥쪽 모서리에 걸치게
        cx = x0 + (CARD_W + 40 if side > 0 else -40)
        cy = y0 + 150
        canvas.alpha_composite(art, (round(cx - art.width / 2), round(cy - art.height / 2)))

    pulse = 1 + 0.022 * level / (LEVELS - 1)    # 말할 때 살짝 커짐
    nod = dy - round(5 * level / (LEVELS - 1))   # 말할 때 살짝 들썩임
    sx, sy = sx * pulse, sy * pulse
    if sx == 1.0 and sy == 1.0 and not nod:
        return canvas
    scaled = canvas.resize((round(canvas.width * sx), round(canvas.height * sy)), Image.Resampling.BICUBIC)
    # 아래쪽 중앙 기준으로 늘이고 줄임 (바닥에 발을 딛고 튀는 느낌). 캔버스 밖은 잘림
    layer = Image.new("RGBA", CHAR_CANVAS, (0, 0, 0, 0))
    layer.paste(scaled, (round((CHAR_CANVAS[0] - scaled.width) / 2), CHAR_CANVAS[1] - scaled.height + nod))
    return layer


def _write_character_track(work: Path, font_path: str, fps: int, total_frames: int, levels: list[int],
                           beats: list[tuple[int, str, str]], hops: list[tuple[int, int]]) -> None:
    files = pose_files()
    beats = sorted(beats)
    hops = sorted(hops)
    ends = [b[0] for b in beats[1:]] + [total_frames]

    def kb_step(since: int, length: int) -> int:
        return min(KB_STEPS, since * KB_STEPS // max(length, 1))

    track = StateTrack(work, "host", lambda key: _character_frame(key, files, font_path))
    blank = (None,) * 13
    hop_idx = 0
    for f in range(total_frames):
        if f < ENTRY_DELAY_CHAR:
            track.push(blank)
            continue
        b = max((k for k, beat in enumerate(beats) if beat[0] <= f), default=0)
        start, pose, emotion = beats[b]
        since = max(f - start, 0)
        length = ends[b] - start
        # 말하기 반응은 15fps로 양자화 (움직임은 충분히 부드럽고 PNG 수는 절반)
        q = f - f % 2
        level = levels[q]
        phase = (q // 4) % len(BAR_PATTERNS) if level else 0
        kb = kb_step(since - since % 2, length)
        prev = prev_kb = mix = None
        sx = sy = 1.0
        dy = 0
        entry = f - ENTRY_DELAY_CHAR
        if entry < len(ENTRY_RISE):
            dy = ENTRY_RISE[entry]
        elif b > 0 and since < len(SQUASH):
            sx, sy, dy = SQUASH[since]
        if b > 0 and since < XFADE_FRAMES:
            prev_start = beats[b - 1][0]
            prev, prev_kb, mix = beats[b - 1][1], kb_step(start - prev_start, start - prev_start), since + 1
        # 자막 구절 홉 (교체 바운스 중이 아닐 때)
        while hop_idx + 1 < len(hops) and hops[hop_idx + 1][0] <= f:
            hop_idx += 1
        if hops and dy == 0 and sx == 1.0:
            h_start, strength = hops[hop_idx]
            pattern = HOP[strength]
            if 0 <= f - h_start < len(pattern):
                dy = pattern[f - h_start]
        side = _fx_side(pose)
        fx_state = None
        state = _fx_frame(since)
        if state is not None:
            step, scale, alpha = state
            fx_state = (EMOTIONS[emotion].fx, step, scale, alpha, side)
        tag_side = 1 if POSE_SLOTS.get(pose) == "left" else -1
        track.push((pose, kb, prev, prev_kb, mix, emotion, level, phase, sx, sy, dy, fx_state, tag_side))
    track.write(work / "host.ffconcat", fps)


def _fx_side(pose: str) -> int:
    """리액션 효과를 놓을 카드 바깥쪽 (화면 안쪽 여유가 있는 쪽, 이모지 스티커 반대편)."""
    return 1 if POSE_SLOTS.get(pose) == "left" else -1


def character_x_expr(moves: list[tuple[float, int]]) -> str:
    """캐릭터 중심 x의 ffmpeg 식. moves: [(이동 시작 시각, 목표 x)] — 첫 항목이 시작 위치.
    이동은 SLIDE_SEC 동안 살짝 지나쳤다 돌아오는 이징(easeOutBack)."""
    expr = str(moves[0][1])
    current = moves[0][1]
    for t, x in moves[1:]:
        if x == current:
            continue
        p = f"clip((t-{t:.3f})/{SLIDE_SEC},0,1)"
        expr += f"+({x - current})*(1+2.70158*pow({p}-1,3)+1.70158*pow({p}-1,2))"
        current = x
    return expr


# ---------- avatar 모드: 왼쪽 원형 아바타 ----------
HOST_X, HOST_Y = 185, 1060      # 아바타 중심 (오른쪽 버튼·하단 UI·중앙 스티커를 피한 왼쪽 중단)
AVATAR_D = 270
AVATAR_CANVAS = (520, 600)
ENTRY_DELAY = 6
ENTRY_POP = [0.2, 0.55, 0.95, 1.14, 1.06, 0.98, 1.0]
EMPHASIS_SCALE = 0.18           # CTA 강조 시 추가 확대
EMPHASIS_STEPS, EMPHASIS_RAMP = 4, 8


@lru_cache
def _avatar() -> Image.Image:
    """원형으로 자른 아바타 (가장자리 안티앨리어싱을 위해 3배 크기 마스크 후 축소)."""
    img = _load_rgb(HOST_IMAGE)
    try:
        x, y, size = (int(v) for v in HOST_CROP.split(","))
    except ValueError:
        size = min(img.size)
        x, y = (img.width - size) // 2, 0
    face = img.crop((x, y, x + size, y + size)).resize((AVATAR_D, AVATAR_D), Image.Resampling.LANCZOS)
    mask = Image.new("L", (AVATAR_D * 3, AVATAR_D * 3), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, AVATAR_D * 3 - 1, AVATAR_D * 3 - 1), fill=255)
    face.putalpha(mask.resize((AVATAR_D, AVATAR_D), Image.Resampling.LANCZOS))
    return face


def _avatar_graphic(level: int, phase: int, font_path: str) -> Image.Image:
    pad = 48
    size = AVATAR_D + pad * 2
    img = Image.new("RGBA", (size, size + 40), (0, 0, 0, 0))
    c = size / 2
    r = AVATAR_D / 2
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gr = r + 10 + 6 * level
    color = GLOW_COLOR + (70 + 40 * level,) if level else (255, 255, 255, 60)
    ImageDraw.Draw(glow).ellipse((c - gr, c - gr, c + gr, c + gr), fill=color)
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(12)))
    draw = ImageDraw.Draw(img)
    ring = 9
    draw.ellipse((c - r - ring, c - r - ring, c + r + ring, c + r + ring), fill=(255, 255, 255, 255))
    img.alpha_composite(_avatar(), (round(c - r), round(c - r)))
    if HOST_NAME:
        _name_tag(draw, c, c + r - 14, level, phase, font_path)
    return img


def _write_avatar_track(work: Path, font_path: str, fps: int, total_frames: int, levels: list[int],
                        emphasis: list[tuple[int, int]]) -> None:
    def emphasis_step(f: int) -> int:
        for start, end in emphasis:
            if start <= f < end:
                ramp = min(f - start, end - f if end < total_frames else EMPHASIS_RAMP, EMPHASIS_RAMP)
                return round(EMPHASIS_STEPS * ramp / EMPHASIS_RAMP)
        return 0

    graphics: dict = {}

    def render(key) -> Image.Image:
        pop, level, phase, emph = key
        canvas = Image.new("RGBA", AVATAR_CANVAS, (0, 0, 0, 0))
        if pop is None:
            return canvas
        if (level, phase) not in graphics:
            graphics[level, phase] = _avatar_graphic(level, phase, font_path)
        g = graphics[level, phase]
        scale = ENTRY_POP[pop] * (1 + EMPHASIS_SCALE * emph / EMPHASIS_STEPS) * (1 + 0.035 * level / (LEVELS - 1))
        if scale != 1.0:
            g = g.resize((max(round(g.width * scale), 1), max(round(g.height * scale), 1)),
                         Image.Resampling.BICUBIC)
        layer = Image.new("RGBA", AVATAR_CANVAS, (0, 0, 0, 0))
        layer.paste(g, (round(AVATAR_CANVAS[0] / 2 - g.width / 2), round(AVATAR_CANVAS[1] / 2 - g.height / 2)))
        canvas.alpha_composite(layer)
        return canvas

    track = StateTrack(work, "host", render)
    for f in range(total_frames):
        pop = None if f < ENTRY_DELAY else min(f - ENTRY_DELAY, len(ENTRY_POP) - 1)
        level = levels[f]
        phase = (f // 3) % len(BAR_PATTERNS) if level else 0
        track.push((pop, level, phase, emphasis_step(f)))
    track.write(work / "host.ffconcat", fps)


def overlay_filter(current_mode: str, x_expr: str | None = None) -> tuple[str, str, str]:
    """(호스트 스트림 전처리 필터, overlay x 식, overlay y 식).
    character: 몸 전체가 천천히 흔들리는 스웨이(회전) + 숨쉬기 + 자리 이동(x_expr)."""
    if current_mode == "character":
        return ("format=rgba,rotate=a='0.011*sin(2*PI*t/3.4)':c=none",
                f"{x_expr or SLOT_X['center']}-w/2", f"{CHAR_Y_TOP}+5*sin(2*PI*t/2.8)")
    return "null", f"{HOST_X}-w/2", f"{HOST_Y}-h/2+7*sin(2*PI*t/2.6)"
