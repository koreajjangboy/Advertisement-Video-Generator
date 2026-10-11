"""Pillow로 만든 PNG 상태들을 ffmpeg concat demuxer용 목록(ffconcat)으로 기록하는 공용 헬퍼."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# PNG 인코딩(zlib)과 Pillow 리샘플링은 GIL을 놓으므로 스레드로 병렬 처리하면 렌더링 준비 시간이 크게 준다
_WORKERS = min(os.cpu_count() or 4, 8)


def write_ffconcat(path: Path, entries: list[tuple[str, int]], fps: int) -> None:
    """entries: [(PNG 파일명(같은 폴더), 표시 프레임 수)]. 파일명은 목록 파일 기준 상대 경로."""
    lines = ["ffconcat version 1.0"]
    for name, count in entries:
        lines += [f"file '{name}'", f"duration {count / fps:.6f}"]
    lines.append(f"file '{entries[-1][0]}'")  # concat demuxer는 마지막 항목의 duration을 무시하므로 반복
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class StateTrack:
    """프레임마다 상태 키를 받아, 같은 상태가 이어지면 표시 시간을 늘리고 새 상태만 PNG로 저장한다.
    이미 저장한 상태가 다시 나오면 파일을 재사용한다. PNG 생성은 스레드 풀에서 병렬로 처리하고 write()에서 기다린다."""

    def __init__(self, work: Path, prefix: str, render):
        self.work, self.prefix, self.render = work, prefix, render
        self.files: dict = {}
        self.entries: list[list] = []  # [파일명, 프레임 수, 상태]
        self.pool = ThreadPoolExecutor(_WORKERS)
        self.jobs = []

    def _save(self, key, name: str) -> None:
        self.render(key).save(self.work / name, compress_level=1)

    def push(self, key) -> None:
        if self.entries and self.entries[-1][2] == key:
            self.entries[-1][1] += 1
            return
        if key not in self.files:
            self.files[key] = f"{self.prefix}_{len(self.files):03d}.png"
            self.jobs.append(self.pool.submit(self._save, key, self.files[key]))
        self.entries.append([self.files[key], 1, key])

    def write(self, path: Path, fps: int) -> None:
        try:
            for job in self.jobs:
                job.result()  # 렌더링 중 예외는 여기서 다시 발생
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
        write_ffconcat(path, [(name, count) for name, count, _ in self.entries], fps)
