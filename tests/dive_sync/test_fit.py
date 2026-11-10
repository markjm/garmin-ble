"""Tests for telling a dive from any other activity."""
from datetime import datetime, timedelta, timezone

import pytest

from garmin_dive_sync.fit import FitError, inspect, make_activity

START = datetime(2026, 11, 2, 10, 14, 33, tzinfo=timezone.utc)


class TestInspect:
    def test_recognises_a_dive(self):
        summary = inspect(make_activity(START, minutes=42, max_depth_m=28.3))
        assert summary.is_dive
        assert summary.dive_mode == "single-gas"
        assert summary.start == START
        assert summary.duration == timedelta(minutes=42)
        assert summary.max_depth_m == pytest.approx(28.3)
        assert summary.crc_ok

    def test_recognises_a_run_as_not_a_dive(self):
        summary = inspect(make_activity(START, minutes=30, max_depth_m=None))
        assert not summary.is_dive
        assert summary.sport == 1
        assert summary.max_depth_m is None

    def test_local_start_uses_the_watch_clock(self):
        summary = inspect(make_activity(START, utc_offset_hours=-10))
        assert summary.local_offset == -36000
        assert summary.local_start == datetime(2026, 11, 2, 0, 14, 33)

    def test_describe_reads_like_a_log_entry(self):
        summary = inspect(make_activity(START, minutes=42, max_depth_m=28.3))
        assert summary.describe() == "dive (single-gas) 2026-11-02 10:14, 42 min, max 28.3 m"

    def test_detects_a_damaged_file(self):
        data = bytearray(make_activity(START))
        data[40] ^= 0xFF
        assert not inspect(bytes(data)).crc_ok

    @pytest.mark.parametrize("junk", [b"", b"not a fit file at all", b"\x0e" + b"\x00" * 30])
    def test_rejects_non_fit_data(self, junk):
        with pytest.raises(FitError):
            inspect(junk)

    def test_rejects_a_truncated_file(self):
        with pytest.raises(FitError):
            inspect(make_activity(START)[:200])

    def test_unknown_dive_mode_with_a_dive_summary_is_still_a_dive(self):
        """Same fallback libdivecomputer uses for sub sports it has not seen."""
        import struct
        from garmin_dive_sync import fit

        writer = fit._Writer()
        writer.message(0, fit.SPORT, [(0, fit.ENUM, 0), (1, fit.ENUM, 99)])
        writer.message(1, fit.DIVE_SUMMARY, [(2, fit.UINT32, 9000), (3, fit.UINT32, 15000)])
        assert inspect(writer.finish()).is_dive

    def test_reads_big_endian_and_compressed_timestamp_records(self):
        """Layouts the writer never produces but real watches do."""
        import struct
        from garmin_ble.protocol.crc import compute_crc

        body = bytes([0x40, 0, 1]) + struct.pack(">HB", 12, 2) + bytes([0, 1, 0, 1, 1, 0])
        body += bytes([0x00, 53, 54])                       # sport: diving, multi-gas
        body += bytes([0x41, 0, 0]) + struct.pack("<HB", 20, 1) + bytes([3, 1, 2])
        body += bytes([0x80 | (1 << 5) | 5, 150])           # compressed timestamp header
        header = struct.pack("<BBHI4s", 12, 0x20, 2132, len(body), b".FIT")
        data = header + body
        data += struct.pack("<H", compute_crc(data))
        summary = inspect(data)
        assert summary.is_dive and summary.dive_mode == "multi-gas" and summary.crc_ok
