"""씬 클립 — 3단계(씬 영상 생성)와 4단계(최종 렌더링)가 공유하는 규칙.

클립 종류 (Script.clip_source)
- luma | runway: AI 씬 영상 업체(VIDEO_PROVIDER)로 생성한 캐릭터 연기 영상 (services/video_ai_service.py)
- fallback: AI 씬 영상을 쓸 수 없을 때(크레딧 부족, 키 없음, 씬 생성 실패, '대체 클립만' 선택) 로컬에서 만드는 대체 클립.
            씬 감정선에 맞는 JB 포즈 이미지에 얼굴 쪽으로 다가가는 카메라 무빙(+ 미세한 흔들림)을 주고,
            씬 안에서 감정이 바뀌면(비트 2개) 그 시점에 다음 포즈로 디졸브. API 호출·비용 없음.

클립 키 (Script.clip_key): 씬 내용(감정 비트·프롬프트·포즈 이미지)에서 계산한 해시 — API 설정·키 유무와는 무관.
대본·포즈를 고쳐 연기 내용이 바뀌면 키가 달라져 기존 클립은 '다시 생성 필요'로 표시된다
(4단계는 기본적으로 그 클립을 계속 쓰고, 3단계를 다시 실행하면 새로 만든다). 같은 감정선 안에서 문구만 고친 경우는 그대로 사용.
"""

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from config import CLIP_DIR, RENDER_PRESET
from services import host_service, video_ai_service

logger = logging.getLogger(__name__)

FALLBACK_SEC = 6.0              # 대체 클립 길이 (렌더링 때 씬 길이에 맞게 늘림)
FALLBACK_FPS = 30
FALLBACK_SIZE = (1080, 1920)
FALLBACK_ZOOM = 0.14            # 클립 동안 얼굴 쪽으로 다가가는 정도
FALLBACK_DISSOLVE = 0.45        # 포즈 전환 디졸브 (초)
FALLBACK_VERSION = "1"          # 대체 클립 모양이 바뀌면 올려서 캐시 무효화


@dataclass(frozen=True)
class SceneSpec:
    section: str
    on_screen_text: str
    narration: str
    visual_prompt: str
    order: int
    host_pose: str
    heroine_refs: tuple = ()  # 이 씬에 등장하는 여자주인공의 레퍼런스 이미지 경로들 (없으면 JB 혼자)


def spec_for(script, episode) -> SceneSpec:
    """Script(또는 같은 필드를 가진 객체) + 에피소드 설정 → SceneSpec. 렌더링·대시보드·진행 API가 같은 규칙을 쓰도록."""
    from services import heroine_service

    refs = ()
    if getattr(episode, "heroine_enabled", False) and getattr(script, "heroine", False):
        refs = tuple(heroine_service.references(episode.heroine_dir, episode.heroine_ref))
    return SceneSpec(script.section, script.on_screen_text, script.narration, script.visual_prompt,
                     script.scene_order, script.host_pose, refs)


def plan(specs: list[SceneSpec]) -> list[host_service.SceneActing | None]:
    return host_service.plan_acting([(s.section, s.on_screen_text, s.narration, s.order, s.host_pose) for s in specs])


def fallback_key(acting: host_service.SceneActing) -> str:
    files = host_service.pose_files()
    beats = [(b.pose, b.emotion, round(b.at, 3), files[b.pose].name, files[b.pose].stat().st_mtime_ns)
             for b in acting.beats]
    raw = json.dumps([FALLBACK_VERSION, FALLBACK_SEC, beats])
    return "fallback_" + hashlib.sha1(raw.encode()).hexdigest()[:20]


def expected_keys(specs: list[SceneSpec]) -> list[tuple[str | None, str | None]]:
    """씬별 지금 대본 기준 (AI 클립 키, fallback 키). 캐릭터 포즈가 없으면 (None, None)."""
    keys = []
    for spec, act in zip(specs, plan(specs)):
        if act is None:
            keys.append((None, None))
            continue
        keys.append((video_ai_service.content_key(spec.section, spec.visual_prompt, act,
                                                  heroine_refs=spec.heroine_refs),
                     fallback_key(act)))
    return keys


def needs_adoption(script) -> bool:
    """지금 형식의 내용 키가 없는 예전 클립: clip_key가 비었거나, 파일 이름 그대로(luma_…)이거나, v1 키(luma_c_…)."""
    if not script.clip_path:
        return False
    key = script.clip_key or ""
    return not key or (key.startswith("luma_") and not key.startswith("luma_k_"))


def adopt_legacy_clips(scripts) -> int:
    """예전 방식으로 만든 Luma 클립을 지금 키 형식으로 등록 — 다시 생성(과금)할지 판단을 정확히 하기 위해.
    v1 클립은 '포즈 이미지를 첫 프레임으로' 만든 것이므로 pose 방식의 내용 키를 받는다:
    첫 프레임 설정(LUMA_/RUNWAY_SCENE_FRAME)이 pose면 그대로 최신, generate면 '다시 생성 필요'(영상 프롬프트가 장면에
    반영되지 않은 클립).
    대본이 바뀌어 맞지 않는 클립은 이전 키로 남긴다. 등록한 수를 반환."""
    from services import clip_legacy

    specs = [spec_for(s, s.episode) for s in scripts]
    adopted = 0
    for script, spec, act in zip(scripts, specs, plan(specs)):
        if not needs_adoption(script):
            continue
        stem = Path(script.clip_path).stem
        script.clip_status = "completed"
        script.clip_source = script.clip_source or ("luma" if stem.startswith("luma_") else "fallback")
        if act is None:
            script.clip_key = script.clip_key or stem
            continue
        if stem == fallback_key(act):
            script.clip_key = stem
            adopted += 1
            continue
        prompt = clip_legacy.prompt_v1(spec.section, spec.visual_prompt, act)
        signature = video_ai_service.pose_signature(act)
        if script.clip_key == clip_legacy.content_key_v1(prompt, signature) or                 stem in clip_legacy.file_keys_v1(prompt, signature):
            script.clip_key = video_ai_service.content_key(spec.section, spec.visual_prompt, act, mode="pose")
            adopted += 1
        else:
            script.clip_key = script.clip_key or stem
    return adopted


def is_current(clip_key: str | None, keys: tuple[str | None, str | None]) -> bool:
    return bool(clip_key) and clip_key in keys


def make_fallback_clip(ffmpeg: str, acting: host_service.SceneActing) -> Path:
    """대체 클립 생성 (같은 키의 파일이 있으면 재사용). 경로를 반환."""
    dest = CLIP_DIR / f"{fallback_key(acting)}.mp4"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    files = host_service.pose_files()
    w, h = FALLBACK_SIZE
    frames = round(FALLBACK_SEC * FALLBACK_FPS)
    beats = list(acting.beats)
    inputs, graph = [], []
    # 비트별 구간 길이 (디졸브 겹침 포함)
    starts = [round(b.at * FALLBACK_SEC * FALLBACK_FPS) for b in beats]
    ends = starts[1:] + [frames]
    overlap = round(FALLBACK_DISSOLVE * FALLBACK_FPS)
    for k, (beat, start, end) in enumerate(zip(beats, starts, ends)):
        length = end - start + (overlap if k + 1 < len(beats) else 0)
        inputs += ["-i", str(files[beat.pose])]
        z = f"1+{FALLBACK_ZOOM:.3f}*(1-cos(PI*on/{length}))/2"
        fx, fy = host_service.FACE_X, host_service.FACE_Y
        # 얼굴 쪽으로 다가가며 손으로 든 카메라처럼 아주 조금 흔들림 (2배 해상도에서 zoompan → 떨림 감소)
        x = f"(iw-iw/zoom)*{fx:.2f}+6*sin(on/17)"
        y = f"(ih-ih/zoom)*{fy:.2f}+5*sin(on/23)"
        graph.append(f"[{k}:v]scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
                     f"zoompan=z='{z}':x='{x}':y='{y}':d={length}:s={w}x{h}:fps={FALLBACK_FPS},setsar=1[c{k}]")
    prev = "c0"
    for k in range(1, len(beats)):
        offset = (starts[k]) / FALLBACK_FPS
        graph.append(f"[{prev}][c{k}]xfade=transition=fade:duration={FALLBACK_DISSOLVE}:offset={offset:.3f}[x{k}]")
        prev = f"x{k}"
    graph.append(f"[{prev}]trim=end_frame={frames},format=yuv420p[out]")
    tmp = dest.with_suffix(".part.mp4")
    cmd = [ffmpeg, "-y", "-hide_banner", "-nostdin", "-loglevel", "error", *inputs,
           "-filter_complex", ";".join(graph), "-map", "[out]", "-an",
           "-c:v", "libx264", "-preset", RENDER_PRESET, "-crf", "18", "-r", str(FALLBACK_FPS), str(tmp)]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if result.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"대체 클립 생성 실패: {result.stderr.strip()[-600:]}")
    tmp.replace(dest)
    return dest
