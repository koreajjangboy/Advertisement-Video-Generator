import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _list(name: str) -> list[str]:
    return [v.strip() for v in os.getenv(name, "").split(",") if v.strip()]


DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/adgen"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# 기본 모델이 한도 초과(429)/서버 오류일 때 순서대로 시도할 모델 (쉼표 구분, 모델별로 한도가 따로 잡힘)
GEMINI_FALLBACK_MODELS = _list("GEMINI_FALLBACK_MODELS")
GEMINI_TIMEOUT_SEC = float(os.getenv("GEMINI_TIMEOUT_SEC", "90"))

# API 사용 모드 — 반복 테스트 시 Gemini 한도 소모를 줄이기 위한 일괄 스위치
#   live    : 아래 개별 설정 그대로 사용
#   saver   : 대본만 Gemini, TTS는 gTTS 고정 (TTS 호출 0회)
#   offline : Gemini 호출 없음 — 템플릿 대본(mock) + gTTS
API_MODE = os.getenv("API_MODE", "live").lower()
if API_MODE not in ("live", "saver", "offline"):
    raise RuntimeError(f"지원하지 않는 API_MODE: {API_MODE} (live | saver | offline)")

# 대본 생성: gemini | mock (원문 문장을 잘라 만든 템플릿 대본, API 호출 없음)
LLM_PROVIDER = "mock" if API_MODE == "offline" else os.getenv("LLM_PROVIDER", "gemini").lower()
# 모든 Gemini 모델이 한도 초과일 때: mock 이면 템플릿 대본으로 대체, 빈 값이면 오류 메시지로 종료
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "").lower()
# 백그라운드 대본 생성 중 한도 초과 시, 안내된 대기 시간이 이 값 이하면 기다렸다가 재시도
LLM_MAX_WAIT_SEC = float(os.getenv("LLM_MAX_WAIT_SEC", "60"))

# TTS: gemini (Gemini TTS, WAV) | gtts (무료 Google 번역 TTS, MP3)
TTS_PROVIDER = "gtts" if API_MODE != "live" else os.getenv("TTS_PROVIDER", "gemini").lower()
# 기본 엔진이 한도 초과(429)/오류일 때 즉시 사용할 엔진. 빈 값이면 폴백 없음.
TTS_FALLBACK = os.getenv("TTS_FALLBACK", "gtts").lower()
TTS_MODEL = os.getenv("TTS_MODEL", "gemini-3.8-flash-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "Charon")
# 말투·템포 지시 — 대본과 분리된 별도 필드(speech_metadata.style)로 전달되어 읽히지 않는다.
# 짧은 묘사형 권장 (예: "빠른 템포, 리듬감 있고 또렷한 숏폼 광고 내레이션"). 비우면 지시 없이 기본 낭독.
TTS_STYLE = os.getenv("TTS_STYLE", "빠른 템포, 리듬감 있고 또렷한 숏폼 광고 내레이션")
SOURCE_TEXT_MAX_CHARS = int(os.getenv("SOURCE_TEXT_MAX_CHARS", "30000"))

# ---------- 영상 렌더링 ----------
# ffmpeg 실행 파일 경로. 비우면 imageio-ffmpeg 번들 → 시스템 PATH 순으로 찾는다.
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "")
# 자막 폰트(.ttf/.otf/.ttc). 비우면 OS 기본 한글 폰트(맑은 고딕 Bold 등)를 찾는다.
RENDER_FONT_PATH = os.getenv("RENDER_FONT_PATH", "")
RENDER_KEN_BURNS = _bool("RENDER_KEN_BURNS", "true")            # 배경 카메라 무빙(줌·팬)
RENDER_MOTION_STRENGTH = float(os.getenv("RENDER_MOTION_STRENGTH", "1.0"))  # 무빙 세기 배수 (0.5 잔잔 ~ 1.5 과감)
RENDER_NARRATION_CAPTION = _bool("RENDER_NARRATION_CAPTION", "true")  # 말소리에 맞춰 끊어 보여주는 자막
RENDER_TITLE_BAR = _bool("RENDER_TITLE_BAR", "true")            # 상단 고정 제목(훅 문구) + 도서명
# 씬 전환: auto(씬마다 다른 효과) | none | ffmpeg xfade 이름 (fade, slideleft, zoomin, circleopen ...)
RENDER_TRANSITION = os.getenv("RENDER_TRANSITION", "auto").lower()
RENDER_TRANSITION_SEC = float(os.getenv("RENDER_TRANSITION_SEC", "0.35"))
RENDER_TRANSITION_SFX = _bool("RENDER_TRANSITION_SFX", "true")  # 전환 시 짧은 '휙' 효과음
RENDER_SCENE_PADDING_SEC = float(os.getenv("RENDER_SCENE_PADDING_SEC", "0.15"))  # 내레이션 뒤 여백
# 호스트 캐릭터
#   character: HOST_POSE_DIR의 포즈 이미지들을 씬마다 대본에 맞게 바꿔 가며 화면 중앙 하단에 크게 배치
#   avatar   : HOST_IMAGE 한 장을 원형 아바타로 잘라 왼쪽에 고정
#   auto     : 포즈 폴더에 이미지가 있으면 character, 없으면 avatar
HOST_ENABLED = _bool("HOST_ENABLED", "true")
HOST_MODE = os.getenv("HOST_MODE", "auto").lower()
HOST_POSE_DIR = BASE_DIR / os.getenv("HOST_POSE_DIR", "media/images/JB")
HOST_IMAGE = BASE_DIR / os.getenv("HOST_IMAGE", "media/images/JB.png")
HOST_CROP = os.getenv("HOST_CROP", "830,20,480")  # 아바타로 쓸 정사각형 영역 "x,y,크기" (비우면 이미지 중앙 상단)
HOST_NAME = os.getenv("HOST_NAME", "JB")           # 아바타 아래 이름표 (비우면 표시 안 함)
# BGM: 파일 또는 폴더 (폴더면 첫 번째 음원). 없으면 BGM_GENERATE=true일 때 기본 붐뱁 비트를 합성해 사용
BGM_ENABLED = _bool("BGM_ENABLED", "true")
BGM_PATH = BASE_DIR / os.getenv("BGM_PATH", "media/bgm")
BGM_GENERATE = _bool("BGM_GENERATE", "true")
BGM_VOLUME = float(os.getenv("BGM_VOLUME", "0.25"))   # 내레이션 대비 BGM 크기 (덕킹 전)
BGM_DUCKING = _bool("BGM_DUCKING", "true")            # 내레이션이 나올 때 BGM 자동으로 줄이기
RENDER_PRESET = os.getenv("RENDER_PRESET", "veryfast")  # x264 preset (ultrafast ~ slow)
RENDER_CRF = int(os.getenv("RENDER_CRF", "21"))
RENDER_TIMEOUT_SEC = int(os.getenv("RENDER_TIMEOUT_SEC", "900"))

# AI 씬 영상 (Luma, Image-to-Video)
#   HOST_POSE_DIR의 캐릭터 포즈 이미지를 첫 프레임으로, 씬 대본·감정으로 만든 프롬프트로 씬별 연기 클립을 생성해
#   배경 영상으로 쓴다. 키가 없거나, API_MODE가 live가 아니거나, 호출이 실패한 씬은 기존 렌더링(캐릭터 카드 +
#   켄 번즈 배경)으로 자동 대체된다.
# 앞뒤 공백·따옴표는 제거 (복사·붙여넣기 실수로 인증이 실패하지 않도록)
LUMA_API_KEY = os.getenv("LUMA_API_KEY", "").strip().strip("\"'").strip()
# API 종류는 키 형식으로 자동 선택
#   agents       : Luma API Platform(platform.lumalabs.ai) 키 'luma-api-…' → https://agents.lumalabs.ai/v1, ray-3.2
#   dream-machine: 예전 Dream Machine API 키 'luma-…'           → https://api.lumalabs.ai/dream-machine/v1, ray-2
# (서로 다른 서버라 키를 바꿔 쓰면 401/403 'Not authenticated')
LUMA_API_FLAVOR = "agents" if LUMA_API_KEY.startswith("luma-api-") else "dream-machine"
_LUMA_DEFAULTS = {"agents": ("https://agents.lumalabs.ai/v1", "ray-3.2"),
                  "dream-machine": ("https://api.lumalabs.ai/dream-machine/v1", "ray-2")}[LUMA_API_FLAVOR]
LUMA_ENABLED = _bool("LUMA_ENABLED", "true")
LUMA_API_BASE = (os.getenv("LUMA_API_BASE") or _LUMA_DEFAULTS[0]).rstrip("/")
LUMA_MODEL = os.getenv("LUMA_MODEL") or _LUMA_DEFAULTS[1]  # 비우면 API 종류별 기본 모델
LUMA_RESOLUTION = os.getenv("LUMA_RESOLUTION", "720p")     # 540p | 720p | 1080p
LUMA_DURATION = os.getenv("LUMA_DURATION", "5s")           # agents: 5s (첫 프레임 지정 시 5s만) / dream-machine: 5s | 9s
# 포즈 이미지를 넘기는 방법
#   agents: 비워 두면 이미지를 요청에 직접(base64) 담아 보낸다 — 공개 URL 불필요
#   dream-machine: 공개 URL만 받으므로 이 서버의 /media가 외부에서 열리는 주소(배포 도메인, ngrok/cloudflared 터널)가
#                  필요 → {LUMA_IMAGE_BASE_URL}/media/images/JB/00.png 형태로 전달
# http(s)://로 시작하지 않는 값은 무시한다.
LUMA_IMAGE_BASE_URL = os.getenv("LUMA_IMAGE_BASE_URL", "").strip().rstrip("/")
# 프롬프트에 쓰는 캐릭터 외형 설명 (얼굴·옷이 씬마다 바뀌지 않도록 고정)
LUMA_CHARACTER = os.getenv("LUMA_CHARACTER", "the young Korean man with dark tousled hair in a red hoodie")
# 여자주인공 — 에피소드 생성 화면에서 등장 여부·이미지 폴더를 고른다 (아래는 기본값)
#   폴더 안 이미지 중 대표 이미지(헤어스타일·의상 기준) + 나머지(얼굴 일관성용)를 최대 HEROINE_MAX_REFS장까지
#   Luma 이미지 모델에 캐릭터 레퍼런스로 넘겨, 여자주인공 등장 씬의 키프레임을 JB와 함께 만든다.
HEROINE_DIR = os.getenv("HEROINE_DIR", "media/images/caoting")
HEROINE_CHARACTER = os.getenv("HEROINE_CHARACTER",
                              "a graceful young East Asian woman with long dark hair in an elegant white qipao-style dress")
HEROINE_MAX_REFS = int(os.getenv("HEROINE_MAX_REFS", "3"))
# 씬 첫 프레임 만드는 방법
#   generate: 씬의 '영상 프롬프트' + JB 포즈 이미지(캐릭터 레퍼런스)로 이미지 모델(uni-1)이 '그 장면 속의 JB' 키프레임을
#             만들고, 그 키프레임에서 영상을 시작 → 프롬프트의 장소·구도·조명이 영상에 반영됨 (씬마다 이미지 1장 추가 비용)
#   pose    : JB 포즈 이미지를 그대로 첫 프레임으로 (비용 적음, 대신 영상이 포즈 이미지의 배경·구도에서 시작)
LUMA_SCENE_FRAME = os.getenv("LUMA_SCENE_FRAME", "generate").lower()
LUMA_IMAGE_MODEL = os.getenv("LUMA_IMAGE_MODEL", "uni-1")  # uni-1 | uni-1-max (고품질·고가)
LUMA_CONCURRENCY = int(os.getenv("LUMA_CONCURRENCY", "4"))  # 동시 생성 수 (Build 티어 최대 10)
LUMA_POLL_SEC = float(os.getenv("LUMA_POLL_SEC", "5"))
LUMA_TIMEOUT_SEC = int(os.getenv("LUMA_TIMEOUT_SEC", "600"))  # 클립 하나 생성 대기 한도

MEDIA_DIR = BASE_DIR / os.getenv("MEDIA_DIR", "media")
AUDIO_DIR = MEDIA_DIR / "audio"
VIDEO_DIR = MEDIA_DIR / "video"
IMAGE_DIR = MEDIA_DIR / "images"
BGM_DIR = MEDIA_DIR / "bgm"
CLIP_DIR = MEDIA_DIR / "clips"   # AI 생성 씬 클립 (내용 해시로 캐시 — 같은 씬은 다시 생성하지 않음)

for d in (AUDIO_DIR, VIDEO_DIR, IMAGE_DIR, BGM_DIR, CLIP_DIR):
    d.mkdir(parents=True, exist_ok=True)
