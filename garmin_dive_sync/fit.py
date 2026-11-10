"""Just enough FIT to tell a dive from a run.

The watch's file directory says a file is an activity but not which sport, so
the only way to pick the dives out is to look inside. This reads the handful of
messages that answer "is this a dive, when, how deep, how long" and ignores the
rest. Turning a FIT file into a dive profile is Subsurface's job, not ours.

There is also a small writer. It exists so the simulator and the tests have
realistic dive files to move around; it is not a general FIT encoder.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from garmin_ble.protocol.crc import compute_crc

#: FIT timestamps count seconds from 1989-12-31 00:00:00 UTC.
FIT_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)

# Global message numbers
FILE_ID = 0
SPORT = 12
SESSION = 18
RECORD = 20
ACTIVITY = 34
DIVE_SETTINGS = 258
DIVE_GAS = 259
DIVE_SUMMARY = 268

#: FIT ``sport`` value for diving.
SPORT_DIVING = 53

#: FIT ``sub_sport`` values that are dives. The same list libdivecomputer uses
#: to decide whether Subsurface should import a file.
DIVE_SUB_SPORTS = {
    53: "single-gas",
    54: "multi-gas",
    55: "gauge",
    56: "apnea",
    57: "apnea hunt",
    63: "CCR",
}

# Base types used by the writer: (type byte, struct code)
ENUM = (0x00, "B")
SINT8 = (0x01, "b")
UINT8 = (0x02, "B")
UINT16 = (0x84, "H")
UINT32 = (0x86, "I")
UINT32Z = (0x8C, "I")


class FitError(ValueError):
    """The data is not a readable FIT file."""


@dataclass(frozen=True)
class FitSummary:
    """What a FIT file says about itself, as far as dive syncing cares."""

    is_dive: bool
    sport: Optional[int] = None
    sub_sport: Optional[int] = None
    #: When the activity started, in UTC.
    start: Optional[datetime] = None
    #: Seconds to add to UTC for the watch's local time when it was recorded.
    local_offset: Optional[int] = None
    duration: Optional[timedelta] = None
    max_depth_m: Optional[float] = None
    #: False if the file's own checksum does not match its contents.
    crc_ok: bool = True

    @property
    def local_start(self) -> Optional[datetime]:
        """Start time on the watch's own clock, as a naive datetime."""
        if self.start is None:
            return None
        local = self.start + timedelta(seconds=self.local_offset or 0)
        return local.replace(tzinfo=None)

    @property
    def dive_mode(self) -> Optional[str]:
        return DIVE_SUB_SPORTS.get(self.sub_sport) if self.sub_sport is not None else None

    def describe(self) -> str:
        """One line for a human: ``dive 2026-11-02 10:14, 42 min, max 28.3 m``."""
        parts = ["dive" if self.is_dive else "not a dive"]
        if self.is_dive and self.dive_mode:
            parts[0] = f"dive ({self.dive_mode})"
        elif not self.is_dive and self.sport is not None:
            parts[0] = f"not a dive (sport {self.sport})"
        when = self.local_start
        if when is not None:
            parts[0] += f" {when:%Y-%m-%d %H:%M}"
        if self.duration is not None:
            parts.append(f"{round(self.duration.total_seconds() / 60)} min")
        if self.max_depth_m is not None:
            parts.append(f"max {self.max_depth_m:.1f} m")
        return ", ".join(parts)


# Reading


def _messages(data: bytes) -> Iterator[Tuple[int, Dict[int, int]]]:
    """Yield ``(global_message_number, {field_number: value})`` for every data
    message whose fields are plain integers. Other fields are skipped.
    """
    position = 0
    while position == 0 or len(data) - position >= 12:
        if len(data) - position < 12:
            raise FitError("too short to be a FIT file")
        header_size = data[position]
        if header_size < 12 or data[position + 8 : position + 12] != b".FIT":
            raise FitError("not a FIT file")
        (data_size,) = struct.unpack_from("<I", data, position + 4)
        start = position + header_size
        end = start + data_size
        if end + 2 > len(data):
            raise FitError("FIT file is shorter than its header says")
        yield from _records(data, start, end)
        position = end + 2  # a FIT file can be several files end to end


def _records(data: bytes, position: int, end: int) -> Iterator[Tuple[int, Dict[int, int]]]:
    # local message type -> (global number, byte order, [(field, size)], total size)
    definitions: Dict[int, Tuple[int, str, List[Tuple[int, int]], int]] = {}

    while position < end:
        header = data[position]
        position += 1

        if header & 0x80:
            local_type = (header >> 5) & 0x03  # compressed timestamp header
        elif header & 0x40:
            local_type = header & 0x0F
            if position + 5 > end:
                raise FitError("truncated FIT definition")
            order = ">" if data[position + 1] else "<"
            (global_number,) = struct.unpack_from(order + "H", data, position + 2)
            field_count = data[position + 4]
            position += 5
            fields = []
            for _ in range(field_count):
                fields.append((data[position], data[position + 1]))
                position += 3
            total = sum(size for _, size in fields)
            if header & 0x20:  # developer fields: sized, but not ours to read
                developer_count = data[position]
                position += 1
                for _ in range(developer_count):
                    total += data[position + 1]
                    position += 3
            if position > end:
                raise FitError("truncated FIT definition")
            definitions[local_type] = (global_number, order, fields, total)
            continue
        else:
            local_type = header & 0x0F

        definition = definitions.get(local_type)
        if definition is None:
            raise FitError("FIT data message without a definition")
        global_number, order, fields, total = definition
        if position + total > end:
            raise FitError("truncated FIT data message")

        values: Dict[int, int] = {}
        offset = position
        for number, size in fields:
            if size in (1, 2, 4):
                code = {1: "B", 2: "H", 4: "I"}[size]
                (value,) = struct.unpack_from(order + code, data, offset)
                if value != (1 << (8 * size)) - 1:  # all ones means "no value"
                    values[number] = value
            offset += size
        position += total
        yield global_number, values


def _file_crc_ok(data: bytes) -> bool:
    """Check the checksum of the first FIT file in *data*."""
    if len(data) < 14:
        return False
    header_size = data[0]
    (data_size,) = struct.unpack_from("<I", data, 4)
    end = header_size + data_size
    if end + 2 > len(data):
        return False
    (stored,) = struct.unpack_from("<H", data, end)
    return compute_crc(data[:end]) == stored


def inspect(data: bytes) -> FitSummary:
    """Summarise a FIT file. Raises :class:`FitError` if it cannot be read."""
    sport = sub_sport = None
    session_sport = session_sub_sport = None
    start = created = first_record = None
    elapsed_ms = None
    max_depth_mm = bottom_time_ms = None
    local_offset = None
    saw_dive_summary = False

    try:
        for number, fields in _messages(data):
            if number == FILE_ID:
                created = fields.get(4, created)
            elif number == SPORT:
                sport = fields.get(0, sport)
                sub_sport = fields.get(1, sub_sport)
            elif number == SESSION:
                session_sport = fields.get(5, session_sport)
                session_sub_sport = fields.get(6, session_sub_sport)
                start = fields.get(2, start) if start is None else start
                if 7 in fields:
                    elapsed_ms = (elapsed_ms or 0) + fields[7]
            elif number == RECORD:
                if first_record is None:
                    first_record = fields.get(253)
            elif number == ACTIVITY:
                if 5 in fields and 253 in fields:
                    local_offset = fields[5] - fields[253]
            elif number == DIVE_SUMMARY:
                if 2 in fields or 3 in fields:
                    saw_dive_summary = True
                if 3 in fields:
                    max_depth_mm = max(max_depth_mm or 0, fields[3])
                if 11 in fields:
                    bottom_time_ms = max(bottom_time_ms or 0, fields[11])
    except (struct.error, IndexError) as exc:
        raise FitError(f"malformed FIT file: {exc}") from None

    sport = sport if sport is not None else session_sport
    sub_sport = sub_sport if sub_sport is not None else session_sub_sport
    is_dive = sport == SPORT_DIVING or sub_sport in DIVE_SUB_SPORTS or saw_dive_summary

    begin = next((t for t in (start, created, first_record) if t is not None), None)
    duration_ms = bottom_time_ms if bottom_time_ms is not None else elapsed_ms

    return FitSummary(
        is_dive=is_dive,
        sport=sport,
        sub_sport=sub_sport,
        start=FIT_EPOCH + timedelta(seconds=begin) if begin is not None else None,
        local_offset=local_offset,
        duration=timedelta(milliseconds=duration_ms) if duration_ms is not None else None,
        max_depth_m=max_depth_mm / 1000 if max_depth_mm is not None else None,
        crc_ok=_file_crc_ok(data),
    )


# Writing (simulator and tests only)


class _Writer:
    def __init__(self) -> None:
        self._body = bytearray()
        self._layouts: Dict[int, Tuple[int, str]] = {}

    def message(self, local: int, number: int, fields: Sequence[Tuple[int, tuple, int]]) -> None:
        """Write a definition and one data message: ``(field, base type, value)``."""
        layout = "<" + "".join(base[1] for _, base, _ in fields)
        if self._layouts.get(local) != (number, layout):
            self._layouts[local] = (number, layout)
            self._body += struct.pack("<BBBHB", 0x40 | local, 0, 0, number, len(fields))
            for field, base, _ in fields:
                self._body += bytes([field, struct.calcsize("<" + base[1]), base[0]])
        self._body += bytes([local]) + struct.pack(layout, *(v for _, _, v in fields))

    def finish(self) -> bytes:
        header = struct.pack("<BBHI4s", 14, 0x20, 2132, len(self._body), b".FIT")
        header += struct.pack("<H", compute_crc(header))
        content = header + bytes(self._body)
        return content + struct.pack("<H", compute_crc(content))


def _fit_time(when: datetime) -> int:
    return int((when - FIT_EPOCH).total_seconds())


def make_activity(
    start: datetime,
    minutes: int = 40,
    max_depth_m: Optional[float] = 18.0,
    utc_offset_hours: float = 0,
    serial: int = 3_400_000_001,
    product: int = 4223,
) -> bytes:
    """Build a small but well-formed activity FIT file.

    With a *max_depth_m* it is a single-gas dive with a depth profile; with
    ``None`` it is a run, for exercising the "not a dive" path.
    """
    is_dive = max_depth_m is not None
    sport, sub_sport = (SPORT_DIVING, 53) if is_dive else (1, 0)
    begin = _fit_time(start)
    seconds = minutes * 60
    writer = _Writer()

    writer.message(0, FILE_ID, [
        (0, ENUM, 4), (1, UINT16, 1), (2, UINT16, product),
        (3, UINT32Z, serial), (4, UINT32, begin),
    ])
    writer.message(1, SPORT, [(0, ENUM, sport), (1, ENUM, sub_sport)])

    if is_dive:
        writer.message(2, DIVE_SETTINGS, [(2, UINT8, 40), (3, UINT8, 85), (4, ENUM, 1)])
        writer.message(3, DIVE_GAS, [(254, UINT16, 0), (0, UINT8, 0), (1, UINT8, 32), (2, ENUM, 1)])
        max_mm = int(max_depth_m * 1000)
        total_mm = 0
        samples = 0
        for t in range(0, seconds + 1, 10):
            # Down in a tenth of the dive, level, up in the last fifth.
            down, up = seconds * 0.1, seconds * 0.8
            if t < down:
                depth = max_mm * t / down
            elif t < up:
                depth = max_mm
            else:
                depth = max_mm * (seconds - t) / (seconds - up)
            depth = int(depth)
            total_mm += depth
            samples += 1
            writer.message(4, RECORD, [
                (253, UINT32, begin + t), (92, UINT32, depth), (13, SINT8, 24),
            ])
        writer.message(5, DIVE_SUMMARY, [
            (253, UINT32, begin + seconds), (0, UINT16, SESSION), (1, UINT16, 0),
            (2, UINT32, total_mm // samples), (3, UINT32, max_mm),
            (10, UINT32, 1), (11, UINT32, seconds * 1000),
        ])
    else:
        for t in range(0, seconds + 1, 10):
            writer.message(4, RECORD, [(253, UINT32, begin + t), (3, UINT8, 150)])

    writer.message(6, SESSION, [
        (253, UINT32, begin + seconds), (2, UINT32, begin),
        (7, UINT32, seconds * 1000), (5, ENUM, sport), (6, ENUM, sub_sport),
    ])
    writer.message(7, ACTIVITY, [
        (253, UINT32, begin + seconds), (0, UINT32, seconds * 1000),
        (5, UINT32, begin + seconds + int(utc_offset_hours * 3600)), (1, UINT16, 1),
    ])
    return writer.finish()


__all__ = [
    "FitError",
    "FitSummary",
    "DIVE_SUB_SPORTS",
    "SPORT_DIVING",
    "inspect",
    "make_activity",
]
