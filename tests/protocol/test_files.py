"""Tests for the GFDI file transfer messages.

The byte layouts are checked against hand-built expectations taken from
Gadgetbridge's message classes, since a watch is the only other thing that can
say whether they are right.
"""
import struct
from datetime import datetime, timezone

import pytest

from garmin_ble.protocol import files as ft
from garmin_ble.protocol.crc import compute_crc
from garmin_ble.protocol.gfdi import GARMIN_EPOCH_OFFSET, parse_message


def body(frame: bytes) -> bytes:
    """A frame's payload: no size, type, or CRC."""
    return frame[4:-2]


def entry(index=7, sub_type=4, size=1234, flags=0, when=datetime(2026, 11, 2, 10, 30, tzinfo=timezone.utc)):
    return ft.DirectoryEntry(
        index=index, data_type=128, sub_type=sub_type, number=index,
        specific_flags=0, flags=flags, size=size, modified=when,
    )


class TestDirectory:
    def test_round_trip(self):
        entries = [entry(1), entry(2, sub_type=32, size=99), entry(3, flags=ft.ARCHIVE_FLAG)]
        assert ft.parse_directory(ft.encode_directory(entries)) == entries

    def test_record_layout(self):
        """index, type, sub type, number, flags, flags, size, Garmin timestamp."""
        stamp = 1_000_000
        raw = struct.pack("<HBBHBBII", 12, 128, 4, 3, 0, 0x10, 54321, stamp)
        (parsed,) = ft.parse_directory(raw)
        assert (parsed.index, parsed.data_type, parsed.sub_type, parsed.number) == (12, 128, 4, 3)
        assert parsed.size == 54321
        assert parsed.modified.timestamp() == stamp + GARMIN_EPOCH_OFFSET
        assert parsed.is_archived

    def test_empty_directory(self):
        assert ft.parse_directory(b"") == []

    def test_rejects_partial_record(self):
        with pytest.raises(ValueError):
            ft.parse_directory(b"\x00" * 17)

    def test_activity_is_what_a_dive_is_filed_as(self):
        assert entry(sub_type=4).is_activity
        assert not entry(sub_type=32).is_activity

    def test_kind_names_known_and_unknown_types(self):
        assert entry(sub_type=4).kind == "activity"
        assert entry(sub_type=200).kind == "128/200"


class TestHostMessages:
    def test_download_request_for_a_new_file(self):
        frame = ft.build_download_request(5)
        assert struct.unpack("<HH", frame[:4]) == (len(frame), 5002)
        # file index, data offset, request type (1 = new), CRC seed, data size
        assert body(frame) == struct.pack("<HIBHI", 5, 0, 1, 0, 0)
        assert struct.unpack("<H", frame[-2:])[0] == compute_crc(frame[:-2])

    def test_download_request_continuing_a_file(self):
        frame = ft.build_download_request(5, offset=4096, crc_seed=0xBEEF)
        assert body(frame) == struct.pack("<HIBHI", 5, 4096, 0, 0xBEEF, 0)

    def test_chunk_acknowledgement(self):
        frame = ft.build_file_transfer_status(362)
        assert struct.unpack("<H", frame[2:4])[0] == 5000
        # answered message, ACK, transfer status, data offset
        assert body(frame) == struct.pack("<HBBI", 5004, 0, 0, 362)

    def test_chunk_rejection_names_the_status(self):
        frame = ft.build_file_transfer_status(0, ft.TransferStatus.CRC_MISMATCH)
        assert body(frame)[3] == 3


class TestParsing:
    def test_download_request_round_trip(self):
        request = ft.parse_download_request(parse_message(ft.build_download_request(9)))
        assert (request.file_index, request.offset, request.is_new) == (9, 0, True)

    def test_download_status_ok(self):
        message = parse_message(ft.build_download_request_status(ft.DownloadStatus.OK, 70_000))
        answer = ft.parse_download_request_status(ft.parse_status(message))
        assert answer.ok and answer.size == 70_000

    def test_download_status_refusal(self):
        message = parse_message(ft.build_download_request_status(ft.DownloadStatus.INDEX_UNKNOWN))
        answer = ft.parse_download_request_status(ft.parse_status(message))
        assert not answer.ok
        assert answer.status == ft.DownloadStatus.INDEX_UNKNOWN

    def test_download_status_without_detail_is_a_refusal(self):
        """A bare NAK carries no download status at all."""
        status = ft.StatusMessage(ref_type=5002, status=1, body=b"")
        assert not ft.parse_download_request_status(status).ok

    def test_file_data_round_trip(self):
        data = bytes(range(200))
        message = parse_message(ft.build_file_transfer_data(data, 400, 0x1234))
        chunk = ft.parse_file_transfer_data(message)
        assert (chunk.offset, chunk.crc, chunk.data) == (400, 0x1234, data)

    def test_file_data_excludes_the_frame_checksum(self):
        message = parse_message(ft.build_file_transfer_data(b"abc", 0, 0))
        assert ft.parse_file_transfer_data(message).data == b"abc"

    def test_transfer_status_round_trip(self):
        message = parse_message(ft.build_file_transfer_status(77, ft.TransferStatus.OFFSET_MISMATCH))
        answer = ft.parse_file_transfer_status(ft.parse_status(message))
        assert (answer.status, answer.offset, answer.ok) == (4, 77, False)

    def test_status_ignores_other_message_types(self):
        assert ft.parse_status(parse_message(ft.build_download_request(1))) is None
