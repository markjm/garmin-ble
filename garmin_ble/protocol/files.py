"""GFDI file transfer: the directory listing and file download messages.

A watch stores everything it records (activities, dives, monitoring, sleep) as
files, addressed by a 16-bit *file index*. Index 0 is the directory: a flat
table of 16-byte records describing every other file. Downloading is the same
exchange for the directory and for a file:

1. host sends ``DOWNLOAD_REQUEST`` for an index;
2. watch answers with a status carrying the file size;
3. watch sends ``FILE_TRANSFER_DATA`` chunks, each with a running CRC, and
   waits for the host to acknowledge one before sending the next.

Both directions are covered: the host uses the request builders and the
parsers, the simulator uses the watch-side builders. Message layouts are ported
from Gadgetbridge's ``DownloadRequestMessage``, ``FileTransferDataMessage`` and
their status messages.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import List, Optional, Sequence

from ..constants import GarminMessage
from .gfdi import GARMIN_EPOCH_OFFSET, GfdiMessage, _build_frame

#: The file index of the directory listing itself.
DIRECTORY_INDEX = 0

#: ``data_type`` of every FIT file. The sub type says which kind.
FIT_DATA_TYPE = 128

#: Bit in :attr:`DirectoryEntry.flags` set once a sync client has taken the file.
ARCHIVE_FLAG = 0x10

_ENTRY = struct.Struct("<HBBHBBII")

#: Names for the FIT sub types seen on real watches, for display only.
FIT_SUB_TYPES = {
    1: "device",
    2: "settings",
    3: "sports",
    4: "activity",
    5: "workouts",
    6: "courses",
    7: "schedules",
    8: "location",
    10: "totals",
    11: "goals",
    20: "summary",
    29: "records",
    32: "monitor",
    33: "multisport",
    34: "segments",
    35: "segment_list",
    37: "clubs",
    38: "score",
    39: "adjustments",
    40: "hmd",
    41: "changelog",
    44: "metrics",
    49: "sleep",
    56: "pace_bands",
    58: "device_58",
    59: "muscle_map",
    60: "running_track",
    61: "ecg",
    62: "benchmark",
    63: "power_guidance",
    65: "calendar",
    68: "hrv_status",
    70: "hsa",
    71: "com_act",
    72: "fbt_backup",
    73: "skin_temp",
    74: "fbt_ptd_backup",
}

#: The sub type of a recorded activity. Dives are activities.
ACTIVITY_SUB_TYPE = 4


class DownloadStatus(IntEnum):
    """The watch's verdict on a ``DOWNLOAD_REQUEST``."""

    OK = 0
    INDEX_UNKNOWN = 1
    INDEX_NOT_READABLE = 2
    NO_SPACE_LEFT = 3
    INVALID = 4
    NOT_READY = 5
    CRC_INCORRECT = 6


class TransferStatus(IntEnum):
    """The receiver's verdict on one ``FILE_TRANSFER_DATA`` chunk."""

    OK = 0
    RESEND = 1
    ABORT = 2
    CRC_MISMATCH = 3
    OFFSET_MISMATCH = 4
    SYNC_PAUSED = 5


@dataclass(frozen=True)
class DirectoryEntry:
    """One file on the watch, as described by its directory record."""

    index: int
    data_type: int
    sub_type: int
    number: int
    specific_flags: int
    flags: int
    size: int
    modified: datetime

    @property
    def is_fit(self) -> bool:
        return self.data_type == FIT_DATA_TYPE

    @property
    def is_activity(self) -> bool:
        """Whether this is a recorded activity, which is what a dive is."""
        return self.is_fit and self.sub_type == ACTIVITY_SUB_TYPE

    @property
    def is_archived(self) -> bool:
        """Whether a sync client (Garmin Connect, usually) already took it."""
        return bool(self.flags & ARCHIVE_FLAG)

    @property
    def kind(self) -> str:
        """A short name for the file type, e.g. ``activity`` or ``128/77``."""
        if self.is_fit and self.sub_type in FIT_SUB_TYPES:
            return FIT_SUB_TYPES[self.sub_type]
        return f"{self.data_type}/{self.sub_type}"

    def __str__(self) -> str:
        return (
            f"#{self.index} {self.kind} {self.size:,} bytes "
            f"{self.modified:%Y-%m-%d %H:%M:%S}"
        )


def _from_garmin_time(seconds: int) -> datetime:
    return datetime.fromtimestamp(seconds + GARMIN_EPOCH_OFFSET, tz=timezone.utc)


def _to_garmin_time(when: datetime) -> int:
    return max(int(when.timestamp()) - GARMIN_EPOCH_OFFSET, 0)


def parse_directory(data: bytes) -> List[DirectoryEntry]:
    """Decode the contents of file index 0 into its entries.

    Raises ``ValueError`` if the length is not a whole number of records, which
    means the download itself went wrong.
    """
    if len(data) % _ENTRY.size:
        raise ValueError(
            f"directory is {len(data)} bytes, not a multiple of {_ENTRY.size}"
        )
    entries = []
    for offset in range(0, len(data), _ENTRY.size):
        index, data_type, sub_type, number, specific, flags, size, stamp = (
            _ENTRY.unpack_from(data, offset)
        )
        entries.append(
            DirectoryEntry(
                index=index,
                data_type=data_type,
                sub_type=sub_type,
                number=number,
                specific_flags=specific,
                flags=flags,
                size=size,
                modified=_from_garmin_time(stamp),
            )
        )
    return entries


def encode_directory(entries: Sequence[DirectoryEntry]) -> bytes:
    """Build directory contents from entries. Used by the simulator."""
    return b"".join(
        _ENTRY.pack(
            e.index, e.data_type, e.sub_type, e.number, e.specific_flags,
            e.flags, e.size, _to_garmin_time(e.modified),
        )
        for e in entries
    )


# Builders: host side


def build_download_request(
    file_index: int,
    offset: int = 0,
    crc_seed: int = 0,
    max_size: int = 0,
) -> bytes:
    """Ask the watch to send a file. Index 0 is the directory.

    A non-zero *offset* continues an interrupted download, in which case
    *crc_seed* must be the running CRC of the bytes already held. *max_size* of
    zero means the whole file.
    """
    request_type = 1 if offset == 0 else 0  # NEW / CONTINUE
    payload = struct.pack("<HIBHI", file_index, offset, request_type, crc_seed, max_size)
    return _build_frame(int(GarminMessage.DOWNLOAD_REQUEST), payload)


def build_file_transfer_status(
    offset: int, status: TransferStatus = TransferStatus.OK
) -> bytes:
    """Acknowledge (or reject) one ``FILE_TRANSFER_DATA`` chunk."""
    payload = struct.pack(
        "<HBBI", int(GarminMessage.FILE_TRANSFER_DATA), 0, int(status), offset
    )
    return _build_frame(int(GarminMessage.RESPONSE), payload)


# Builders: watch side (used by the simulator and by tests)


def build_download_request_status(status: DownloadStatus, size: int = 0) -> bytes:
    """The watch's answer to a ``DOWNLOAD_REQUEST``."""
    payload = struct.pack(
        "<HBBI", int(GarminMessage.DOWNLOAD_REQUEST), 0, int(status), size
    )
    return _build_frame(int(GarminMessage.RESPONSE), payload)


def build_file_transfer_data(data: bytes, offset: int, crc: int) -> bytes:
    """One chunk of file content. *crc* is the running CRC including *data*."""
    payload = struct.pack("<BHI", 0, crc, offset) + data
    return _build_frame(int(GarminMessage.FILE_TRANSFER_DATA), payload)


# Parsers


@dataclass(frozen=True)
class StatusMessage:
    """A decoded ``RESPONSE`` (5000): which message it answers, and how."""

    ref_type: int
    status: int
    #: Whatever follows the status byte; its layout depends on ``ref_type``.
    body: bytes

    @property
    def acknowledged(self) -> bool:
        return self.status == 0


@dataclass(frozen=True)
class DownloadRequestStatus:
    status: DownloadStatus
    size: int

    @property
    def ok(self) -> bool:
        return self.status == DownloadStatus.OK


@dataclass(frozen=True)
class FileTransferData:
    flags: int
    crc: int
    offset: int
    data: bytes


@dataclass(frozen=True)
class FileTransferStatus:
    status: int
    offset: int

    @property
    def ok(self) -> bool:
        return self.status == TransferStatus.OK


@dataclass(frozen=True)
class DownloadRequest:
    file_index: int
    offset: int
    is_new: bool
    crc_seed: int
    max_size: int


def _body(message: GfdiMessage) -> bytes:
    """The payload without the trailing CRC16."""
    return message.payload[:-2] if len(message.payload) >= 2 else b""


def parse_status(message: GfdiMessage) -> Optional[StatusMessage]:
    """Decode the common head of a ``RESPONSE`` message."""
    body = _body(message)
    if message.type != GarminMessage.RESPONSE or len(body) < 3:
        return None
    ref_type, status = struct.unpack_from("<HB", body)
    return StatusMessage(ref_type=ref_type, status=status, body=body[3:])


def parse_download_request_status(status: StatusMessage) -> Optional[DownloadRequestStatus]:
    """Decode the answer to a ``DOWNLOAD_REQUEST``.

    A refusal at the GFDI level (anything but ACK) carries no detail, and is
    reported as ``INVALID``.
    """
    if status.ref_type != GarminMessage.DOWNLOAD_REQUEST:
        return None
    if not status.acknowledged or len(status.body) < 5:
        return DownloadRequestStatus(status=DownloadStatus.INVALID, size=0)
    code, size = struct.unpack_from("<BI", status.body)
    try:
        download_status = DownloadStatus(code)
    except ValueError:
        download_status = DownloadStatus.INVALID
    return DownloadRequestStatus(status=download_status, size=size)


def parse_file_transfer_status(status: StatusMessage) -> Optional[FileTransferStatus]:
    """Decode the receiver's answer to a chunk. Used by the simulator."""
    if status.ref_type != GarminMessage.FILE_TRANSFER_DATA or len(status.body) < 5:
        return None
    code, offset = struct.unpack_from("<BI", status.body)
    return FileTransferStatus(status=code, offset=offset)


def parse_file_transfer_data(message: GfdiMessage) -> Optional[FileTransferData]:
    """Decode one chunk of file content."""
    body = _body(message)
    if message.type != GarminMessage.FILE_TRANSFER_DATA or len(body) < 7:
        return None
    flags, crc, offset = struct.unpack_from("<BHI", body)
    return FileTransferData(flags=flags, crc=crc, offset=offset, data=body[7:])


def parse_download_request(message: GfdiMessage) -> Optional[DownloadRequest]:
    """Decode a ``DOWNLOAD_REQUEST``. Used by the simulator."""
    body = _body(message)
    if message.type != GarminMessage.DOWNLOAD_REQUEST or len(body) < 13:
        return None
    index, offset, request_type, crc_seed, max_size = struct.unpack_from("<HIBHI", body)
    return DownloadRequest(
        file_index=index,
        offset=offset,
        is_new=request_type == 1,
        crc_seed=crc_seed,
        max_size=max_size,
    )


__all__ = [
    "DIRECTORY_INDEX",
    "FIT_DATA_TYPE",
    "ACTIVITY_SUB_TYPE",
    "ARCHIVE_FLAG",
    "DownloadStatus",
    "TransferStatus",
    "DirectoryEntry",
    "parse_directory",
    "encode_directory",
    "build_download_request",
    "build_file_transfer_status",
    "build_download_request_status",
    "build_file_transfer_data",
    "StatusMessage",
    "DownloadRequestStatus",
    "FileTransferData",
    "FileTransferStatus",
    "DownloadRequest",
    "parse_status",
    "parse_download_request_status",
    "parse_file_transfer_status",
    "parse_file_transfer_data",
    "parse_download_request",
]
