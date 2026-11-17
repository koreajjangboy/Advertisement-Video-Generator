"""대본 생성 → TTS → 렌더링 파이프라인 (BackgroundTasks에서 실행)."""

import logging
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, object_session

from database import SessionLocal
from models import ClipStatus, Episode, EpisodeStatus, SceneRenderStatus, Script
from services import clip_service, llm_service, render_service, tts_service, video_ai_service
from services.narration import sanitize_narration

logger = logging.getLogger(__name__)


# ---------- 공용 헬퍼 ----------
def synthesize_scene(
    db: Session, script_id: int, force: bool = False, wait_on_limit: bool = False
) -> bool:
    """씬 1개의 내레이션을 음성으로 변환해 audio_path/audio_duration_sec/audio_provider에 기록하고 커밋.
    이미 유효한 오디오가 있으면 건너뛴다. 새로 생성했으면 True.

    TTS 호출(수 초~한도 대기 수십 초) 동안 DB 트랜잭션을 열어두지 않는다.
    열어두면 'idle in transaction' 상태로 락을 잡아 스키마 변경 등 다른 작업이 막힌다."""
    script = db.get(Script, script_id)
    if script is None:
        return False
    if not force and tts_service.audio_exists(script.audio_path):
        return False
    narration, episode_id, old = script.narration, script.episode_id, script.audio_path
    db.commit()  # 읽기 트랜잭션 종료

    result = tts_service.synthesize(narration, f"ep{episode_id}_scene{script_id}", wait_on_limit=wait_on_limit)

    saved = False
    try:
        script = db.get(Script, script_id)
        if script is None or script.narration != narration:  # 생성 중 삭제·수정됨
            db.commit()
            return False
        script.audio_path = result.url
        script.audio_duration_sec = result.duration_sec
        script.audio_provider = result.provider
        script.mark_render_stale()  # 새 음성 → ready (렌더링 대기)
        db.commit()  # 씬 단위로 저장해 진행 상황이 대시보드에 바로 보이도록
        saved = True
    finally:
        if not saved:  # 저장되지 않은 새 오디오는 어디서도 참조하지 않으므로 바로 삭제 (예외 포함)
            tts_service.delete_media(result.url)
    if old and old != result.url:
        tts_service.delete_media(old)
    return True


def _synthesize_all(db: Session, episode: Episode, force: bool = False) -> int:
    script_ids = [s.id for s in episode.scripts]
    return sum(synthesize_scene(db, sid, force=force, wait_on_limit=True) for sid in script_ids)


def delete_episode_audio(episode: Episode) -> None:
    for script in episode.scripts:
        tts_service.delete_media(script.audio_path)


def delete_episode_media(episode: Episode) -> None:
    """에피소드 삭제 시 오디오·씬 이미지·완성 영상 파일 정리."""
    delete_episode_audio(episode)
    db = object_session(episode)
    for script in episode.scripts:
        tts_service.delete_media(script.image_path)
        # 씬 클립은 내용 해시로 캐시되므로 다른 에피소드가 같은 파일을 쓰고 있으면 남겨 둔다
        if script.clip_path and not (db and db.scalar(select(func.count()).select_from(Script).where(
                Script.clip_path == script.clip_path, Script.episode_id != episode.id))):
            _delete_clip(script.clip_path)
    tts_service.delete_media(episode.video_path)


# ---------- 임시·고아 파일 정리 ----------
CLEANUP_GRACE_SEC = 10 * 60  # 이보다 최근 파일은 다른 작업이 쓰는 중일 수 있으므로 건드리지 않음


def cleanup_stale_media(grace_sec: float = CLEANUP_GRACE_SEC) -> int:
    """작업이 끝난 뒤 남은 임시·고아 파일 삭제. 삭제한 개수를 반환. (앱 시작 시 + 각 백그라운드 작업 종료 시)
    - 생성 도중 중단된 임시 파일: media/audio/*.tmp.*, media/video/*.tmp.mp4, media/clips/*.part*, *.repair.tmp.*
    - 어느 씬도 참조하지 않는 씬 오디오(ep*_scene*), 어느 에피소드도 참조하지 않는 렌더링 영상(episode_*.mp4)
    - 오래된 미리듣기 오디오, 비정상 종료로 남은 렌더링 작업 폴더(시스템 임시 폴더의 render_ep*)
    씬 클립(media/clips/*.mp4)은 비용을 들여 만든 캐시이므로 삭제하지 않는다."""
    import shutil
    import tempfile
    import time as _time

    from config import AUDIO_DIR, CLIP_DIR, VIDEO_DIR

    cutoff = _time.time() - grace_sec

    def old(path: Path) -> bool:
        try:
            return path.stat().st_mtime < cutoff
        except OSError:
            return False

    removed = 0

    def remove(path: Path) -> None:
        nonlocal removed
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed += 1
        except OSError as e:  # Windows에서 재생 중인 파일 등 — 다음 정리 때 다시 시도
            logger.debug("정리 실패 (%s): %s", path, e)

    with SessionLocal() as db:
        audio_refs = {tts_service.media_url_to_path(u) for u in db.scalars(select(Script.audio_path)) if u}
        video_refs = {tts_service.media_url_to_path(u) for u in db.scalars(select(Episode.video_path)) if u}
    for path in [*AUDIO_DIR.glob("*.tmp.*"), *AUDIO_DIR.glob("*.repair.tmp.*"), *VIDEO_DIR.glob("*.tmp.mp4"),
                 *CLIP_DIR.glob("*.part*")]:
        if old(path):
            remove(path)
    for path in AUDIO_DIR.glob("ep*_scene*.*"):
        if path not in audio_refs and old(path):
            remove(path)
    for path in VIDEO_DIR.glob("episode_*.mp4"):
        if path not in video_refs and ".tmp" not in path.name and old(path):
            remove(path)
    for path in CLIP_DIR.glob("*_frame.*"):  # 클립이 지워진 키프레임
        clip = path.with_name(path.name.split("_frame.")[0] + ".mp4")
        if not clip.exists() and old(path):
            remove(path)
    for path in Path(tempfile.gettempdir()).glob("render_ep*"):
        if path.is_dir() and old(path):
            remove(path)
    tts_service.cleanup_previews()
    if removed:
        logger.info("임시·고아 파일 %d개 정리", removed)
    return removed


def _delete_clip(url: str) -> None:
    """씬 클립과 함께 만든 키프레임 이미지(…_frame.jpg 등)까지 삭제."""
    path = tts_service.media_url_to_path(url)
    tts_service.delete_media(url)
    if path:
        for frame in path.parent.glob(path.stem + "_frame.*"):
            frame.unlink(missing_ok=True)


def _cleanup_after_job() -> None:
    try:
        cleanup_stale_media()
    except Exception:  # 정리 실패가 작업 결과에 영향을 주지 않도록
        logger.exception("작업 후 파일 정리 실패")


def _mark_failed(db: Session, episode_id: int, message: str) -> None:
    episode = db.get(Episode, episode_id)
    if episode:
        episode.status = EpisodeStatus.FAILED
        episode.error_message = message[:2000]
        db.commit()


# ---------- 백그라운드 작업 ----------
def generate_script_for_episode(episode_id: int) -> None:
    notice = None
    db = SessionLocal()
    try:
        episode = db.get(Episode, episode_id)
        if episode is None:
            return
        episode.status = EpisodeStatus.SCRIPTING
        episode.error_message = None
        db.commit()

        result = llm_service.generate_script(
            book_title=episode.product_name,
            source_text=episode.source_text,
            book_description=episode.product_description,
            target_audience=episode.target_audience,
            heroine=episode.heroine_enabled,
            duration_sec=episode.duration_sec,
            wait_on_limit=True,
        )
        if result.provider == "mock-fallback":
            notice = "Gemini 한도 초과/오류로 원문 기반 템플릿 대본을 사용했습니다. 한도가 풀리면 '대본 재생성'을 실행하세요."
        delete_episode_audio(episode)
        # 업로드한 배경 이미지는 같은 순서의 새 씬으로 이어받고, 남는 것은 삭제
        old_images = [s.image_path for s in episode.scripts]
        episode.scripts.clear()
        db.flush()  # 기존 장면 삭제를 먼저 반영
        scenes = result.to_scenes()
        for i, (section, scene) in enumerate(scenes, start=1):
            image = old_images[i - 1] if i <= len(old_images) else None
            data = scene.model_dump()
            # LLM이 연출 지시를 섞어 써도 저장 단계에서 걷어냄 (자막·음성 모두 순수 대사만)
            data["narration"] = sanitize_narration(data["narration"]) or data["narration"]
            # 훅은 여자주인공이 기본으로 등장 (편집기에서 끌 수 있음)
            data["heroine"] = (bool(data.get("heroine")) or section == "hook") and episode.heroine_enabled
            episode.scripts.append(Script(scene_order=i, section=section, image_path=image, **data))
        if episode.heroine_enabled:
            ensure_heroine_scenes(episode)
        for leftover in old_images[len(scenes):]:
            tts_service.delete_media(leftover)
        episode.status = EpisodeStatus.SCRIPTED  # 음성은 2단계에서 따로 만든다
        db.commit()
    except Exception as e:
        logger.exception("대본 생성 실패 (episode=%s)", episode_id)
        db.rollback()
        _mark_failed(db, episode_id, f"대본 생성 실패: {e}")
        return
    finally:
        db.close()

    if notice:
        _set_notice(episode_id, notice)


def ensure_heroine_scenes(episode: Episode) -> None:
    """여자주인공 등장이 켜져 있는데 등장 씬이 하나도 없으면 훅(hook)·본문(body) 씬에 등장시킨다 (대본이 정하지 않은 경우의 기본값)."""
    if episode.heroine_enabled and not any(s.heroine for s in episode.scripts):
        for s in episode.scripts:
            s.heroine = s.section in ("hook", "body")


def _set_notice(episode_id: int, message: str) -> None:
    """실패는 아니지만 사용자에게 알려야 할 내용 (다른 오류 메시지가 없을 때만 기록)."""
    with SessionLocal() as db:
        episode = db.get(Episode, episode_id)
        if episode and not episode.error_message:
            episode.error_message = message
            db.commit()


def _idle_status(episode: Episode) -> EpisodeStatus:
    """작업이 끝난 뒤 돌아갈 상태: 최종 영상이 최신이면 done, 씬 영상이 있으면 clipped, 아니면 scripted."""
    if episode.video_path and all(s.render_status == SceneRenderStatus.COMPLETED.value for s in episode.scripts):
        return EpisodeStatus.DONE
    if any(s.clip_path for s in episode.scripts):
        return EpisodeStatus.CLIPPED
    return EpisodeStatus.SCRIPTED


def generate_audio_for_episode(episode_id: int, force: bool = False, script_ids: list[int] | None = None) -> None:
    """2단계: 씬별 TTS. 기본은 음성이 없는 씬만, force면 전부, script_ids면 그 씬들만.
    실패해도 대본·이미 만든 음성은 유지하고 error_message로 알린다."""
    db = SessionLocal()
    try:
        episode = db.get(Episode, episode_id)
        if episode is None:
            return
        episode.status = EpisodeStatus.VOICING
        episode.error_message = None
        db.commit()

        ids = [s.id for s in episode.scripts if script_ids is None or s.id in script_ids]
        made = sum(synthesize_scene(db, sid, force=force, wait_on_limit=True) for sid in ids)
        db.expire_all()
        episode = db.get(Episode, episode_id)
        if episode is None:
            return
        episode.status = _idle_status(episode)
        db.commit()
        logger.info("음성 생성 완료 (episode=%s): %d개 씬", episode_id, made)
    except Exception as e:
        logger.exception("TTS 실패 (episode=%s)", episode_id)
        db.rollback()
        db.expire_all()
        episode = db.get(Episode, episode_id)
        if episode:
            episode.status = _idle_status(episode)
            episode.error_message = f"음성 생성 실패: {e}"[:2000]
            db.commit()
    finally:
        db.close()
        _cleanup_after_job()  # 성공·실패와 관계없이 임시·고아 파일 정리


# ---------- 3단계: 씬 영상 (AI 씬 영상 Luma|Runway → 실패·크레딧 부족 시 대체 클립) ----------
def _specs(scripts: list[Script]) -> list[clip_service.SceneSpec]:
    return [clip_service.spec_for(s, s.episode) for s in scripts]


def _clip_usable(script: Script, keys: tuple[str | None, str | None]) -> bool:
    """지금 대본 기준으로 만든 클립 파일이 있는지 (상태와 무관 — 재생성에 실패해도 이전 최신 클립은 사용 가능)."""
    path = tts_service.media_url_to_path(script.clip_path)
    return bool(path and path.exists()) and clip_service.is_current(script.clip_key, keys)


def _update_clip(script_id: int, state: str, path: Path | None = None, source: str | None = None,
                 key: str | None = None, error: str | None = None) -> None:
    """씬 클립 상태 기록 (작업 도중 진행 상황이 대시보드에 바로 보이도록 씬마다 즉시 커밋)."""
    with SessionLocal() as db:
        script = db.get(Script, script_id)
        if script is None:
            return
        script.clip_status = state
        if state == ClipStatus.COMPLETED.value and path is not None:
            old = script.clip_path
            script.clip_path = tts_service.media_path_to_url(path)
            script.clip_source, script.clip_key, script.clip_error = source, key, None
            # 이전 클립 정리 — 다른 씬이 쓰지 않을 때만. 비용을 들인 AI 클립은 새 AI 클립으로 바뀔 때만 지운다
            # (대체 클립으로 바뀌는 경우엔 보존 → 나중에 대본을 되돌리면 캐시로 다시 쓰임)
            replaceable = video_ai_service.is_ai_source(source) or Path(old or "").stem.startswith("fallback_")
            if old and old != script.clip_path and replaceable and not db.scalar(
                    select(func.count()).select_from(Script).where(Script.clip_path == old, Script.id != script_id)):
                _delete_clip(old)
        elif state == ClipStatus.FAILED.value:
            script.clip_error = (error or "알 수 없는 오류")[:500]
        db.commit()


def generate_clips_for_episode(episode_id: int, mode: str = "auto", force: bool = False) -> None:
    """3단계: 씬별 캐릭터 연기 클립 생성.
    mode — auto: AI 씬 영상(VIDEO_PROVIDER)으로 생성, 실패한 씬·크레딧 부족·사용 불가 시 대체 클립 (대체 모드)
           ai: AI 씬 영상만 (실패한 씬은 failed로 남김) / fallback: AI 호출 없이 대체 클립만
    force=False면 지금 대본 기준 클립이 이미 있는 씬은 건너뛴다 (비용 절약)."""
    db = SessionLocal()
    try:
        episode = db.get(Episode, episode_id)
        if episode is None:
            return
        scripts = list(episode.scripts)
        specs = _specs(scripts)
        acting = clip_service.plan(specs)
        if not scripts or not all(acting):
            raise RuntimeError("캐릭터 포즈 이미지(HOST_POSE_DIR)가 없어 씬 영상을 만들 수 없습니다.")
        keys = clip_service.expected_keys(specs)
        ids = [s.id for s in scripts]
        pending = {i for i, s in enumerate(scripts) if force or not _clip_usable(s, keys[i])}
        for i, s in enumerate(scripts):
            if i in pending:
                s.clip_status, s.clip_error = ClipStatus.QUEUED.value, None
        episode.error_message = None
        db.commit()
        logger.info("씬 영상 생성 (episode=%s, mode=%s): %d/%d개 씬", episode_id, mode, len(pending), len(scripts))

        notices = []
        if pending and mode != "fallback":
            name = video_ai_service.label()
            reason = video_ai_service.unavailable_reason()
            if reason:
                notices.append(f"{name}를 사용할 수 없습니다 ({reason})."
                               + (" 대체 클립으로 만들었습니다." if mode == "auto" else ""))
                if mode == "ai":
                    for i in pending:
                        _update_clip(ids[i], ClipStatus.FAILED.value, error=reason)
                    pending = set()
            else:
                jobs = [j if j is not None and j.index in pending else None
                        for j in video_ai_service.plan_jobs(
                            [(sp.section, sp.visual_prompt, sp.heroine_refs) for sp in specs], acting)]

                def on_update(index: int, state: str, path: Path | None, error: str | None) -> None:
                    _update_clip(ids[index], state, path, video_ai_service.provider().NAME, keys[index][0], error)

                results = video_ai_service.generate_clips(jobs, on_update)
                pending -= {i for i in pending if results.paths[i] is not None}
                if results.fatal_kind == "credits":
                    notices.append(f"{name} 크레딧(잔액)이 부족하거나 사용 한도를 넘어 {len(pending)}개 씬을 만들지 못했습니다."
                                   + (" 대체 모드로 전환해 대체 클립으로 만들었습니다." if mode == "auto" else
                                      " 충전 후 다시 실행하거나 '대체 클립만'으로 실행하세요."))
                elif results.errors:
                    notices.append(f"{name}로 {len(results.errors)}개 씬을 만들지 못했습니다: {results.reasons()}."
                                   + (" 해당 씬은 대체 클립으로 만들었습니다." if mode == "auto" else ""))
        if pending and mode != "ai":
            ffmpeg = render_service.find_ffmpeg()
            for i in sorted(pending):
                _update_clip(ids[i], ClipStatus.GENERATING.value)
                try:
                    path = clip_service.make_fallback_clip(ffmpeg, acting[i])
                    _update_clip(ids[i], ClipStatus.COMPLETED.value, path, "fallback", keys[i][1])
                except Exception as e:
                    logger.exception("대체 클립 생성 실패 (script=%s)", ids[i])
                    _update_clip(ids[i], ClipStatus.FAILED.value, error=str(e))

        db.expire_all()
        episode = db.get(Episode, episode_id)
        if episode is None:
            return
        episode.status = EpisodeStatus.CLIPPED
        episode.error_message = " ".join(notices)[:2000] or None
        db.commit()
        logger.info("씬 영상 생성 완료 (episode=%s)", episode_id)
    except Exception as e:
        logger.exception("씬 영상 생성 실패 (episode=%s)", episode_id)
        db.rollback()
        _mark_clips_failed(db, episode_id, f"씬 영상 생성 실패: {e}")
    finally:
        db.close()
        _cleanup_after_job()  # 성공·실패와 관계없이 임시·고아 파일 정리


def _mark_clips_failed(db: Session, episode_id: int, message: str) -> None:
    try:
        db.expire_all()
        episode = db.get(Episode, episode_id)
        if episode is None:
            return
        for s in episode.scripts:
            if s.clip_status in (ClipStatus.QUEUED.value, ClipStatus.GENERATING.value):
                s.clip_status = ClipStatus.FAILED.value
                s.clip_error = message[:500]
        episode.status = EpisodeStatus.FAILED
        episode.error_message = message[:2000]
        db.commit()
    except Exception:
        logger.exception("씬 영상 실패 상태 기록 실패 (episode=%s)", episode_id)
        db.rollback()


# ---------- 4단계: 최종 렌더링 (씬 클립 + TTS + BGM + 자막 → FFmpeg) ----------
def render_video_for_episode(episode_id: int, missing_clip: str = "fallback", use_outdated: bool = True) -> None:
    """missing_clip — 씬 영상이 없는 씬을 fallback(그 자리에서 대체 클립을 만들어 사용) | card(캐릭터 카드 연출)로 채움.
    use_outdated — 대본·영상 프롬프트가 바뀌어 '다시 생성 필요'인 클립도 사용 (기본). False면 그 씬도 missing_clip으로."""
    db = SessionLocal()
    try:
        episode = db.get(Episode, episode_id)
        if episode is None:
            return
        episode.status = EpisodeStatus.RENDERING
        episode.error_message = None
        db.commit()

        # 음성은 2단계에서 만들지만, 빠진 씬이 있으면 렌더링 전에 채운다 (한도 초과 시 폴백 엔진)
        auto_voiced = _synthesize_all(db, episode)

        scripts = list(db.get(Episode, episode_id).scripts)
        specs = _specs(scripts)
        keys = clip_service.expected_keys(specs)
        scenes, missing, outdated = [], [], []
        for i, s in enumerate(scripts):
            s.render_status = SceneRenderStatus.RENDERING.value
            audio_file = tts_service.media_url_to_path(s.audio_path)
            if audio_file and audio_file.exists() and s.audio_duration_sec is None:
                s.audio_duration_sec = round(tts_service.read_duration(audio_file), 2)
            usable = _clip_usable(s, keys[i])
            if not usable:
                clip_file = tts_service.media_url_to_path(s.clip_path)
                if use_outdated and clip_file and clip_file.exists():
                    outdated.append(i)
                    usable = True
                else:
                    missing.append(i)
            scenes.append(render_service.RenderScene(
                section=s.section,
                on_screen_text=s.on_screen_text,
                narration=s.narration,
                duration_sec=s.duration_sec,
                audio_path=audio_file,
                audio_duration_sec=s.audio_duration_sec,
                image_path=tts_service.media_url_to_path(s.image_path),
                emoji=s.emoji,
                host_pose=s.host_pose,
                order=s.scene_order,
                video_path=tts_service.media_url_to_path(s.clip_path) if usable else None,
            ))
        ids = [s.id for s in scripts]
        brand = episode.product_name
        db.commit()  # 렌더링(수십 초) 동안 트랜잭션을 열어두지 않음

        ai_kept = {i for i, s in enumerate(scripts) if video_ai_service.is_ai_source(s.clip_source) and s.clip_path}
        notices = []
        if missing:
            acting = clip_service.plan(specs)
            how = "캐릭터 카드 연출"
            if missing_clip == "fallback" and all(acting[i] for i in missing):
                ffmpeg = render_service.find_ffmpeg()
                for i in missing:
                    path = clip_service.make_fallback_clip(ffmpeg, acting[i])
                    scenes[i].video_path = path
                    if i not in ai_kept:  # 기존 AI 클립 기록은 덮어쓰지 않음 (이번 렌더링에만 대체 클립 사용)
                        _update_clip(ids[i], ClipStatus.COMPLETED.value, path, "fallback", keys[i][1])
                how = "대체 클립"
            scene_list = ", ".join(f"#{i + 1}" for i in missing)
            notices.append(f"씬 영상이 없는 {len(missing)}개 씬({scene_list})은 {how}로 렌더링했습니다.")
        if outdated:
            scene_list = ", ".join(f"#{i + 1}" for i in outdated)
            notices.append(f"{len(outdated)}개 씬({scene_list})은 대본·영상 프롬프트 변경 전에 만든 씬 영상을 사용했습니다.")
        if notices:
            notices.append("최신 내용으로 바꾸려면 3단계를 다시 실행하세요.")
        if auto_voiced:
            notices.insert(0, f"음성이 없던 {auto_voiced}개 씬은 렌더링 전에 자동으로 음성을 만들었습니다.")
        notice = " ".join(notices) or None

        video = render_service.render_episode(episode_id, scenes, brand=brand)

        db.expire_all()
        episode = db.get(Episode, episode_id)
        if episode is None:  # 렌더링 중 삭제됨
            video.unlink(missing_ok=True)
            return
        old_video = episode.video_path
        episode.video_path = tts_service.media_path_to_url(video)
        episode.status = EpisodeStatus.DONE
        if notice:
            episode.error_message = notice
        for s in episode.scripts:
            s.render_status = SceneRenderStatus.COMPLETED.value
        db.commit()
        logger.info("에피소드 %s 렌더링 완료 → %s", episode_id, episode.video_path)
        if old_video and old_video != episode.video_path:
            tts_service.delete_media(old_video)
    except Exception as e:
        logger.exception("렌더링 실패 (episode=%s)", episode_id)
        db.rollback()
        _mark_render_failed(db, episode_id, f"렌더링 실패: {e}")
    finally:
        db.close()
        _cleanup_after_job()  # 성공·실패와 관계없이 임시·고아 파일 정리


def _mark_render_failed(db: Session, episode_id: int, message: str) -> None:
    """에피소드는 failed, 렌더링 중이던 씬은 failed로 기록 (이전 영상 video_path는 유지)."""
    try:
        episode = db.get(Episode, episode_id)
        if episode is None:
            return
        for s in episode.scripts:
            if s.render_status == SceneRenderStatus.RENDERING.value:
                s.render_status = SceneRenderStatus.FAILED.value
        episode.status = EpisodeStatus.FAILED
        episode.error_message = message[:2000]
        db.commit()
    except Exception:  # DB 장애 등 — 상태 기록까지 실패해도 작업 스레드는 조용히 종료
        logger.exception("렌더링 실패 상태 기록 실패 (episode=%s)", episode_id)
        db.rollback()
