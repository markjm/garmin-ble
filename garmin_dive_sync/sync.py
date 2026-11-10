"""Download the dives on a watch into a folder Subsurface can import.

Subsurface reads Garmin dives from the watch's USB storage: it is pointed at a
mount point and looks for FIT files under ``Garmin/Activity/`` inside it. This
module rebuilds that layout from files fetched over BLE, so the folder can
stand in for the mount point.

Two things about the watch shape the design:

* Its directory says a file is an activity but not which sport, so a file has
  to be downloaded before it can be recognised as a dive. A state file
  remembers every activity already looked at, so none is fetched twice.
* Garmin Connect decides what to upload from an "archived" flag on each file.
  Nothing here sets it, so syncing to Subsurface never stops a dive from also
  reaching Garmin's own apps.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from garmin_ble import ConnectionFailed, DirectoryEntry, NotConnected, TransferError, Watch

from .fit import FitError, FitSummary, inspect

#: Where Subsurface looks for dives, relative to the folder it is pointed at.
ACTIVITY_DIR = Path("Garmin") / "Activity"

#: Where activities that turned out not to be dives go, if they are kept.
OTHER_DIR = Path("other-activities")

STATE_FILE = ".garmin-dive-sync.json"

#: Called as ``on_progress(entry, bytes_received, total_bytes)``.
ProgressCallback = Callable[[DirectoryEntry, int, int], None]

#: Called once per activity as its outcome becomes known.
ResultCallback = Callable[["SyncedFile"], None]


@dataclass(frozen=True)
class SyncedFile:
    """What happened to one activity on the watch."""

    entry: DirectoryEntry
    #: ``dive``, ``not-a-dive``, ``unreadable`` or ``failed``.
    outcome: str
    path: Optional[Path] = None
    summary: Optional[FitSummary] = None
    error: Optional[str] = None


@dataclass
class SyncReport:
    """The result of one sync run."""

    out_dir: Path
    #: Activities found on the watch.
    on_watch: int = 0
    #: Activities skipped because an earlier run already handled them.
    already_synced: int = 0
    files: List[SyncedFile] = field(default_factory=list)
    #: Set if the run stopped early because the link to the watch dropped.
    interrupted: Optional[str] = None

    def _with(self, outcome: str) -> List[SyncedFile]:
        return [f for f in self.files if f.outcome == outcome]

    @property
    def dives(self) -> List[SyncedFile]:
        """New dives saved by this run, plus any file that could not be
        classified and was kept for Subsurface to judge."""
        return self._with("dive") + self._with("unreadable")

    @property
    def not_dives(self) -> List[SyncedFile]:
        return self._with("not-a-dive")

    @property
    def failed(self) -> List[SyncedFile]:
        return self._with("failed")


def _key(entry: DirectoryEntry) -> str:
    """An identity for a file that survives the watch renumbering its indexes."""
    return f"{int(entry.modified.timestamp())}-{entry.size}"


class _State:
    """The record of activities already handled, kept next to the dives."""

    def __init__(self, out_dir: Path):
        self._path = out_dir / STATE_FILE
        self.files: Dict[str, dict] = {}
        try:
            loaded = json.loads(self._path.read_text(encoding="utf-8"))
            self.files = dict(loaded.get("files", {}))
        except (OSError, ValueError, AttributeError):
            pass  # no state yet, or an unreadable one: start over

    def seen(self, entry: DirectoryEntry, out_dir: Path) -> bool:
        record = self.files.get(_key(entry))
        if record is None:
            return False
        # A dive whose file has since been deleted is fetched again.
        saved = record.get("file")
        return saved is None or (out_dir / saved).exists()

    def remember(self, entry: DirectoryEntry, outcome: str, saved: Optional[Path]) -> None:
        self.files[_key(entry)] = {
            "outcome": outcome,
            "file": saved.as_posix() if saved is not None else None,
            "recorded": entry.modified.isoformat(),
            "size": entry.size,
        }
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "files": self.files}, indent=1), encoding="utf-8"
        )
        os.replace(temporary, self._path)


def file_name(entry: DirectoryEntry, summary: Optional[FitSummary]) -> str:
    """Name a dive the way the watch names it on USB storage.

    That is ``YYYY-MM-DD-HH-MM-SS.fit`` in the watch's local time. Subsurface
    sorts on these names to find the newest dive and remembers the last one it
    imported, so they have to sort chronologically.
    """
    when = summary.local_start if summary is not None else None
    if when is None:
        when = entry.modified.replace(tzinfo=None)
    return f"{when:%Y-%m-%d-%H-%M-%S}.fit"


def _save(directory: Path, name: str, data: bytes, entry: DirectoryEntry) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    temporary = path.with_suffix(".part")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    stamp = entry.modified.timestamp()
    os.utime(path, (stamp, stamp))
    return path


async def sync_dives(
    watch: Watch,
    out_dir: Union[str, Path],
    newest: Optional[int] = None,
    redownload: bool = False,
    keep_others: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    on_result: Optional[ResultCallback] = None,
) -> SyncReport:
    """Fetch every dive on *watch* that is not already in *out_dir*.

    Activities are taken newest first, so an interrupted run still leaves the
    most recent dives in place, and the next run picks up where it stopped.

    *newest* limits the run to the N most recent activities on the watch,
    which keeps a first sync short. *redownload* ignores what earlier runs
    recorded. *keep_others* saves activities that are not dives to
    ``other-activities/`` instead of discarding them.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = SyncReport(out_dir=out_dir)
    state = _State(out_dir)

    activities = [entry for entry in await watch.files() if entry.is_activity]
    activities.sort(key=lambda entry: entry.modified, reverse=True)
    report.on_watch = len(activities)
    if newest is not None:
        activities = activities[:newest]

    for entry in activities:
        if not redownload and state.seen(entry, out_dir):
            report.already_synced += 1
            continue

        def progress(received: int, total: int, entry: DirectoryEntry = entry) -> None:
            if on_progress is not None:
                on_progress(entry, received, total)

        try:
            data = await watch.download(entry, progress=progress)
        except TransferError as exc:
            result = SyncedFile(entry, "failed", error=str(exc))
        except (ConnectionFailed, NotConnected) as exc:
            result = SyncedFile(entry, "failed", error=str(exc))
            report.files.append(result)
            if on_result is not None:
                on_result(result)
            report.interrupted = str(exc)
            break
        else:
            result = _store(out_dir, entry, data, keep_others)
            saved = result.path.relative_to(out_dir) if result.path is not None else None
            state.remember(entry, result.outcome, saved)

        report.files.append(result)
        if on_result is not None:
            on_result(result)

    return report


def _store(out_dir: Path, entry: DirectoryEntry, data: bytes, keep_others: bool) -> SyncedFile:
    """Decide what a downloaded activity is and put it where it belongs."""
    try:
        summary: Optional[FitSummary] = inspect(data)
    except FitError:
        summary = None

    if summary is None:
        # The transfer itself is checksummed, so the bytes are what the watch
        # holds. If we cannot read them, let Subsurface try rather than drop
        # what might be a dive.
        path = _save(out_dir / ACTIVITY_DIR, file_name(entry, None), data, entry)
        return SyncedFile(entry, "unreadable", path=path)

    if summary.is_dive:
        path = _save(out_dir / ACTIVITY_DIR, file_name(entry, summary), data, entry)
        return SyncedFile(entry, "dive", path=path, summary=summary)

    path = None
    if keep_others:
        path = _save(out_dir / OTHER_DIR, file_name(entry, summary), data, entry)
    return SyncedFile(entry, "not-a-dive", path=path, summary=summary)


__all__ = [
    "ACTIVITY_DIR",
    "OTHER_DIR",
    "STATE_FILE",
    "SyncedFile",
    "SyncReport",
    "sync_dives",
    "file_name",
]
