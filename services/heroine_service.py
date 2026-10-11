"""여자주인공 레퍼런스 이미지 — 에피소드별 폴더·대표 이미지 관리.

- 폴더는 에피소드마다 지정 (기본 HEROINE_DIR = media/images/caoting). 프로젝트 기준 상대 경로나 절대 경로 모두 받되,
  웹에서 임의의 디스크 경로를 읽지 못하도록 media/ 안쪽만 허용한다. DB에는 프로젝트 기준 상대 경로로 저장.
- 레퍼런스: 대표 이미지(heroine_ref)를 맨 앞에, 나머지는 이름순으로 최대 HEROINE_MAX_REFS장.
  대표 이미지는 헤어스타일·의상 기준, 나머지는 같은 사람 얼굴 확인용으로 Luma 이미지 모델에 전달된다.
"""

from pathlib import Path

from config import BASE_DIR, HEROINE_DIR, HEROINE_MAX_REFS, HOST_POSE_DIR, IMAGE_DIR, MEDIA_DIR

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def resolve_dir(value: str | None) -> Path:
    """입력값(상대·절대 경로, 비면 기본값) → 실제 폴더. media/ 밖이거나 폴더가 없으면 ValueError."""
    raw = (value or "").strip().strip('"') or HEROINE_DIR
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    path = path.resolve()
    if not path.is_relative_to(MEDIA_DIR.resolve()):
        raise ValueError(f"여자주인공 이미지 폴더는 {MEDIA_DIR} 안에 있어야 합니다: {raw}")
    if not path.is_dir():
        raise ValueError(f"여자주인공 이미지 폴더가 없습니다: {raw}")
    return path


def to_setting(path: Path) -> str:
    """DB 저장용 — 프로젝트 기준 상대 경로 (예: media/images/caoting)."""
    return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def references(dir_value: str | None, main: str = "") -> list[Path]:
    """Luma에 넘길 레퍼런스 이미지 (대표 이미지 먼저). 폴더가 없거나 비었으면 []."""
    try:
        images = list_images(resolve_dir(dir_value))
    except ValueError:
        return []
    images.sort(key=lambda p: p.name != main)  # 대표 이미지를 맨 앞으로 (나머지는 이름순 유지)
    return images[:HEROINE_MAX_REFS]


def list_folders() -> list[dict]:
    """여자주인공 후보 폴더: media/images/ 바로 아래 폴더 중 이미지가 있는 것 (남자 주인공 포즈 폴더 HOST_POSE_DIR 제외).
    [{dir, name, count, thumb}] — 대시보드 드롭다운용."""
    host = HOST_POSE_DIR.resolve()
    folders = []
    for folder in sorted(IMAGE_DIR.iterdir(), key=lambda p: p.name.lower()) if IMAGE_DIR.is_dir() else []:
        if not folder.is_dir() or folder.resolve() == host:
            continue
        images = list_images(folder)
        if images:
            folders.append({"dir": to_setting(folder), "name": folder.name, "count": len(images),
                            "thumb": "/media/" + images[0].resolve().relative_to(MEDIA_DIR.resolve()).as_posix()})
    return folders


def catalog(dir_value: str | None) -> dict:
    """대시보드용: 폴더 확인 + 이미지 목록 [{name, url}]."""
    folder = resolve_dir(dir_value)
    images = list_images(folder)
    if not images:
        raise ValueError(f"폴더에 이미지(png·jpg·webp)가 없습니다: {to_setting(folder)}")
    return {"dir": to_setting(folder),
            "images": [{"name": p.name, "url": "/media/" + p.relative_to(MEDIA_DIR.resolve()).as_posix()}
                       for p in images]}


def signature(paths: list[Path]) -> list:
    """클립 내용 키용 (파일 이름·크기·수정 시각) — 이미지를 바꾸면 '다시 생성 필요'가 된다."""
    return [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in paths]
