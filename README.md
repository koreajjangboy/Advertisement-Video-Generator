# Advertisement-Video-Generator

FastAPI 기반 숏폼(9:16) 광고 영상 자동 생성 툴 (모놀리식: API + 관리자 대시보드).

## 구조
```
main.py              FastAPI 진입점 (API + 대시보드 + /static, /media 마운트)
config.py            .env 로딩, 경로/설정값
database.py          SQLAlchemy 엔진/세션, get_db, init_db
models.py            Episode(에피소드), Script(장면별 대본) 테이블
schemas.py           Pydantic 요청/응답 스키마
services/
  llm_service.py     Gemini로 장면별 대본 생성 (모델 폴백·한도 대기, 오프라인 템플릿 대본)
  tts_service.py     내레이션 TTS (Gemini TTS / gTTS, 한도 초과·오류 시 자동 폴백)
  render_service.py  Pillow 자막 + FFmpeg로 1080x1920 mp4 렌더링
  video_ai_service.py AI 씬 영상 공통 흐름 — 업체 선택(VIDEO_PROVIDER), 프롬프트, 요청·폴링·다운로드·재시도·캐시
  runway_service.py  Runway 어댑터 (gen4.5 영상, gen4_image 키프레임)
  luma_service.py    Luma 어댑터 (ray-3.2 영상, uni-1 키프레임)
  clip_service.py    씬 클립 공용 규칙 (내용 키, 로컬 대체 클립 생성)
  host_service.py    캐릭터 감정선 분석·포즈 매칭·캐릭터 카드 연기
  pipeline.py        1단계 대본 → 2단계 TTS → 3단계 씬 영상 → 4단계 최종 렌더링 백그라운드 작업
routers/
  production.py      1단계 POST /api/generate-script, 2단계 POST /api/generate-tts
  video.py           3단계 POST /api/generate-scene-clips, 4단계 POST /api/render-final-video
media/               생성된 audio / video / images
templates/index.html 관리자 대시보드
static/              app.js, style.css
```

## 실행
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # DATABASE_URL, GEMINI_API_KEY 수정
# PostgreSQL에 DB 생성:  createdb adgen
uvicorn main:app --reload
```
브라우저에서 http://127.0.0.1:8000 접속. API 문서는 `/docs`.

## 영상 제작 4단계
대시보드의 에피소드마다 단계별 패널과 버튼으로 나눠 실행한다 (각각 백그라운드 작업, 진행 상황은 자동 갱신).

| 단계 | API | 하는 일 |
|---|---|---|
| 1단계: 대본 생성 및 편집 | `POST /api/generate-script` `{"episode_id"}`, 씬 수정 `PATCH /api/scripts/{id}`, 영상 프롬프트 다시 작성 `POST /api/scripts/{id}/visual-prompt` (대시보드는 내레이션·포즈·여자주인공을 바꿔 저장하면 자동 호출) | 씬별 내레이션·화면 자막·영상 프롬프트 생성, 확인·수정·저장 (음성은 만들지 않음) |
| 2단계: 음성(TTS) 제작하기 | `POST /api/generate-tts` `{"episode_id", "force", "script_ids"}`, 씬 하나 `POST /api/scripts/{id}/tts` | 확정된 대본으로 씬별 음성 생성, 씬별·전체 미리 듣기 |
| 3단계: AI 씬 영상 생성하기 | `POST /api/generate-scene-clips` `{"episode_id", "mode", "force"}` | 씬별 캐릭터 연기 클립만 생성·다운로드 (`media/clips/`) |
| 4단계: 최종 FFmpeg 렌더링 실행 | `POST /api/render-final-video` `{"episode_id", "missing_clip", "use_outdated"}` | 씬 클립 + 음성 + BGM 덕킹 + 자막·제목 바 → mp4 |

- 에피소드를 만들면 1단계 대본만 자동 생성된다. 1단계에서 내레이션을 고치면 그 씬의 음성은 지워지고 2단계에서 다시 만든다
- 2단계 기본은 음성이 없는 씬만, `force=true`면 전부, `script_ids`면 그 씬들만 다시 만든다
- 4단계 실행 시 음성이 없는 씬이 남아 있으면 렌더링 전에 자동으로 만들고 안내 문구를 남긴다
- 3단계 `mode`: `auto`(기본, AI 씬 영상으로 만들고 실패한 씬·크레딧 부족·사용 불가 시 **대체 클립**으로 전환) /
  `ai`(AI 씬 영상만, 실패 씬은 실패로 남김) / `fallback`(AI 호출 없이 대체 클립만, 무료)
- 대체 클립: 씬 감정선에 맞는 JB 포즈 이미지에 얼굴 쪽으로 다가가는 카메라 무빙, 감정이 바뀌면 다음 포즈로 디졸브 (로컬 FFmpeg)
- 3단계는 지금 대본 기준 클립이 있는 씬을 건너뛴다 (`force=true`면 전부 다시). 대본·영상 프롬프트·포즈를 바꿔 내용이
  달라진 씬은 '다시 생성 필요'로 표시된다. 진행 상황: `GET /api/episodes/{id}/clips`
- 4단계 `missing_clip`: 씬 영상이 없는 씬을 `fallback`(대체 클립, 기본) 또는 `card`(캐릭터 카드 연출)로 채운다.
  `use_outdated`(기본 true)면 '다시 생성 필요'인 기존 씬 영상도 그대로 사용
- 예전 `POST /api/episodes/{id}/script`·`/tts`·`/render`는 각각 1·2·4단계와 같다 (호환용)
- 여자주인공: 에피소드 생성 화면에서 '여자주인공 등장'을 켜고 레퍼런스 이미지 폴더(기본 `media/images/caoting`,
  media 폴더 안만 허용)와 대표 이미지를 고른다 (나중에 1단계 패널의 '설정 변경' 또는 `PATCH /api/episodes/{id}`).
  대본 생성 시 LLM이 등장 씬(`heroine`)을 정하고(없으면 본문 씬), 씬 '수정'에서 바꿀 수 있다.
  등장 씬은 3단계에서 키프레임 이미지 모델에 JB 포즈 + 여자주인공 이미지(대표 이미지 먼저, Luma 최대 3장·Runway 최대
  2장)를 레퍼런스로 넘겨 두 사람이 함께 있는 첫 프레임을 만든 뒤 영상을 생성한다 (첫 프레임 설정이 pose여도 키프레임을
  만듦. Runway 또는 Luma Agents API 키 필요, 대체 클립·캐릭터 카드 연출에는 등장하지 않음)

## 영상 렌더링
- 세로형 1080x1920 / 30fps / H.264·AAC mp4로 `media/video/`에 저장합니다. 화면 구성:
  - 상단 고정 제목 바(훅 문구, 둘째 줄 노란 강조) + 도서명 라벨
  - 키워드 스티커(씬의 화면 자막) — 기울어진 컬러 박스가 전환 직후 '쿵' 하고 튀어나옴
  - 이모지 일러스트 스티커 — 씬 내용에 맞는 이모지(대본 생성 시 LLM이 선택, 비우면 키워드로 자동)를
    흰 테두리 다이컷 스티커로 순차 등장 → 둥실거림·흔들림. 대시보드 씬 편집에서 바꿀 수 있음
  - 하단 자막 — 내레이션을 2~3어절씩 끊어 말소리 길이에 맞춰 교체, 지금 말하는 단어는 노란색 (가라오케)
  - 배경 — 업로드 이미지 또는 섹션 색 그라데이션(빛 번짐·그레인), 씬마다 다른 카메라 무빙 + 펀치 줌,
    속도가 다른 입자 2겹이 떠오르는 패럴랙스
  - 씬 전환 — xfade 효과(씬마다 다름) + 짧은 효과음
  - 호스트 캐릭터 (`HOST_MODE=auto`)
    - **character** — `media/images/JB/`의 포즈 이미지(놀람·생각·확신·가리키기·인사)로 캐릭터가 대본을 연기
      - 감정선: 내레이션을 문장 단위로 인사·질문·놀람·강조·공감·결론으로 분석해 포즈를 고르고, 한 씬 안에서
        감정이 바뀌면 그 문장이 시작될 때 포즈를 바꾼다 (씬당 최대 2비트, 앞 포즈와 겹치지 않게)
      - 동선: 포즈마다 서는 자리(왼쪽·가운데·오른쪽)가 있어 씬이 바뀌면 미끄러지듯 이동 — 시선·손짓이 자막 쪽을 향하게
      - 교체 모션: 크로스페이드 + 스쿼시&스트레치 바운스, 감정별 만화 효과(!, ?, …, 반짝임, 물결, 화살표)
      - 살아 있는 움직임: 카드 안 미세 켄 번즈, 내레이션 음량에 맞춘 맥동·끄덕임·감정 색 빛, 자막 구절마다 홉
        (핵심 키워드 구절은 크게), 몸 전체 스웨이·숨쉬기
      - 자막은 캐릭터 얼굴을 가리키는 꼬리가 달린 말풍선, 이모지 스티커는 카드 위 모서리에 붙어 함께 이동
      - 포즈는 대본 생성 시 LLM이 고를 수 있고(`host_pose`, 지정하면 씬 전체 고정), 대시보드 씬 편집에서 바꿀 수 있다.
    - **avatar** — 포즈 폴더가 없으면 `HOST_IMAGE`의 `HOST_CROP` 영역을 원형 아바타로 왼쪽에 고정
  - BGM — `media/bgm/`에 음원(mp3/wav 등)을 넣으면 첫 번째 파일을 반복 사용, 없으면 기본 붐뱁 비트를 합성.
    내레이션이 나올 때 자동으로 줄어들고(덕킹) 시작·끝은 페이드
- **AI 씬 영상 (Runway / Luma)** — `.env`의 `VIDEO_PROVIDER`(runway | luma)로 업체를 고른다. 그 업체의 키를 설정하고
  `API_MODE=live`이면 3단계에서 씬마다 캐릭터 연기 클립을 생성해 4단계에서 배경으로 사용 (`services/video_ai_service.py`)
  - 씬 포즈·감정 연기는 AI 영상에 쓰지 않는다 — 표정·동작은 '영상 프롬프트'의 장면 분위기를 따른다
    (포즈는 4단계 캐릭터 카드·대체 클립에만 사용)
  - 첫 프레임: 이미지 모델(`gen4_image` / `uni-1`)이 씬 '영상 프롬프트' + 고정 캐릭터 레퍼런스(`VIDEO_REF_POSE`, 기본
    `thinking` 포즈 이미지 — 얼굴·머리·옷만 참고)로 '그 장면 속 JB' 키프레임을 만들고 → 영상 모델(`gen4.5` / `ray-3.2`)이
    그 키프레임에서 영상 생성 (씬마다 이미지 1장 추가 비용). `RUNWAY_SCENE_FRAME`/`LUMA_SCENE_FRAME`은 무시되며,
    키프레임을 못 만드는 예전 Luma API만 레퍼런스 이미지를 그대로 첫 프레임으로 쓴다
  - 프롬프트 조립: 사용자의 '영상 프롬프트'를 맨 앞에 그대로 → 캐릭터(표정·몸짓은 장면 분위기를 따름) →
    프롬프트에 카메라 지시가 없을 때만 섹션별 카메라 → 빠진 스타일 키워드(9:16·cinematic)만 → 캐릭터 일관성·텍스트 금지.
    비어 있으면 섹션별 기본 장면. 업체 길이 한도(Runway 1000자)를 넘으면 장면 묘사부터 줄인다.
    대시보드 씬의 '… 프롬프트 보기'에서 실제로 보낼 내용을 확인할 수 있다
  - 씬들을 동시에 요청 → 폴링 → 다운로드, `media/clips/`에 내용 해시로 캐시 (같은 씬을 다시 렌더링하면 API를 부르지 않음)
  - 한도 초과·서버·네트워크 오류는 재시도, 재시도 가능한 생성 실패는 다시 제출, 인증 실패·크레딧 부족은 남은 요청 중단
    → 대체 클립, 에피소드에 안내 문구 표시
  - 업체를 바꿔도 이미 만든 클립은 그대로 '최신'으로 인정 (내용 키는 업체와 무관). 씬 포즈를 바꿔도 AI 영상은 그대로
    최신. 포즈 연기 방식으로 만든 예전 클립과 레퍼런스 이미지를 바꾼 경우는 '다시 생성 필요'로 표시된다
  - AI 영상 씬은 1080x1920·30fps로 맞추고(짧으면 살짝 느리게 → 앞뒤 왕복) 캐릭터 카드 없이 영화 자막 배치로 합성.
    제목 바·자막·전환·BGM 덕킹은 그대로
  - Runway: `api.dev.runwayml.com/v1` (`X-Runway-Version: 2024-11-06`), 포즈 이미지는 data URI로 직접 첨부 (공개 URL 불필요).
    크레딧 부족은 400 오류 메시지로 판별. `gen4.5`는 12크레딧/초 (5초 클립 60크레딧)
  - Luma: API는 키 형식으로 자동 선택 — platform.lumalabs.ai의 `luma-api-…` 키는 Agents API(`agents.lumalabs.ai/v1`, `ray-3.2`),
    예전 `luma-…` 키는 Dream Machine API(`api.lumalabs.ai/dream-machine/v1`, `ray-2`). 키와 서버가 맞지 않으면 401/403
  - 포즈 이미지: Agents API는 요청에 직접(base64) 담아 보내므로 공개 URL 불필요. 예전 API는 공개 URL만 받으므로
    `LUMA_IMAGE_BASE_URL`에 이 서버의 `/media`가 외부에서 열리는 주소 지정 (예: `ngrok http 8000`의 주소)
- 씬 길이는 내레이션 길이 + `RENDER_SCENE_PADDING_SEC` (+전환), 음량은 -14 LUFS로 정규화합니다.
- 자막 구절 타이밍은 글자 수 비례 근사치입니다 (TTS가 단어별 타임스탬프를 주지 않음).
- ffmpeg는 `FFMPEG_PATH` → `imageio-ffmpeg` 번들 → 시스템 PATH 순으로 찾으므로 시스템 설치 여부와 관계없이 `pip install -r requirements.txt`만으로 동작합니다.
- 완성 영상은 대시보드에서 재생하거나 `GET /api/episodes/{id}/video`로 다운로드합니다.

## Gemini 한도(429) 대응
| 설정 | 동작 |
|---|---|
| `API_MODE=live` | 개별 설정대로 사용 (기본) |
| `API_MODE=saver` | 대본만 Gemini, TTS는 gTTS (TTS 호출 0회) |
| `API_MODE=offline` | Gemini 호출 없음 — 원문 기반 템플릿 대본 + gTTS (UI·렌더링 반복 테스트용) |
| `GEMINI_FALLBACK_MODELS` | 기본 모델이 한도 초과/서버 오류일 때 순서대로 시도할 모델 |
| `LLM_FALLBACK=mock` | 모든 모델이 막히면 템플릿 대본으로 대체 (비우면 안내 메시지로 실패 처리) |
| `TTS_FALLBACK=gtts` | Gemini TTS가 한도·네트워크·인증 등 어떤 이유로든 실패하면 gTTS로 생성 |
