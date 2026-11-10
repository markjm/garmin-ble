"""The dive sync, end to end against the simulated watch."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from garmin_ble import SimulatedFile, Watch
from garmin_dive_sync import sync_dives
from garmin_dive_sync.cli import main
from garmin_dive_sync.fit import inspect, make_activity
from garmin_dive_sync.sync import STATE_FILE, file_name

DAY = datetime(2026, 11, 1, 9, 30, tzinfo=timezone.utc)
DIVE_1 = make_activity(DAY, 46, 28.4, utc_offset_hours=2)
DIVE_2 = make_activity(DAY + timedelta(hours=4), 52, 18.1, utc_offset_hours=2)
RUN = make_activity(DAY + timedelta(days=1), 35, None, utc_offset_hours=2)

FILES = [
    SimulatedFile(DIVE_1, modified=DAY),
    SimulatedFile(RUN, modified=DAY + timedelta(days=1)),
    SimulatedFile(b"\x00" * 500, modified=DAY, sub_type=32),   # monitoring, not an activity
    SimulatedFile(DIVE_2, modified=DAY + timedelta(hours=4)),
]


@pytest.fixture
async def watch():
    session = Watch.simulated(profile="descentmk3", files=FILES)
    connected = await session.open()
    try:
        yield connected
    finally:
        await session.aclose()


def fit_files(folder):
    return sorted(p.name for p in (folder / "Garmin" / "Activity").glob("*.fit"))


class TestSync:
    async def test_saves_dives_where_subsurface_looks(self, watch, tmp_path):
        report = await sync_dives(watch, tmp_path)

        assert fit_files(tmp_path) == ["2026-11-01-11-30-00.fit", "2026-11-01-15-30-00.fit"]
        assert (tmp_path / "Garmin" / "Activity" / "2026-11-01-11-30-00.fit").read_bytes() == DIVE_1
        assert len(report.dives) == 2 and len(report.not_dives) == 1 and not report.failed
        assert report.on_watch == 3

    async def test_only_activities_are_downloaded(self, watch, tmp_path):
        await sync_dives(watch, tmp_path)
        assert 3 not in watch._transport.download_requests

    async def test_newest_first(self, watch, tmp_path):
        await sync_dives(watch, tmp_path)
        assert watch._transport.download_requests == [0, 2, 4, 1]

    async def test_second_run_downloads_nothing(self, watch, tmp_path):
        await sync_dives(watch, tmp_path)
        watch._transport.download_requests.clear()

        report = await sync_dives(watch, tmp_path)

        assert watch._transport.download_requests == [0]  # the directory only
        assert report.already_synced == 3 and not report.files

    async def test_non_dives_are_remembered_but_not_kept(self, watch, tmp_path):
        await sync_dives(watch, tmp_path)
        assert not (tmp_path / "other-activities").exists()

    async def test_keep_others(self, watch, tmp_path):
        await sync_dives(watch, tmp_path, keep_others=True)
        (kept,) = (tmp_path / "other-activities").glob("*.fit")
        assert kept.read_bytes() == RUN
        assert len(fit_files(tmp_path)) == 2  # still not mixed in with the dives

    async def test_newest_limits_the_run(self, watch, tmp_path):
        report = await sync_dives(watch, tmp_path, newest=2)
        assert fit_files(tmp_path) == ["2026-11-01-15-30-00.fit"]
        assert len(report.files) == 2

    async def test_redownload(self, watch, tmp_path):
        await sync_dives(watch, tmp_path)
        report = await sync_dives(watch, tmp_path, redownload=True)
        assert len(report.dives) == 2 and report.already_synced == 0

    async def test_deleted_dive_is_fetched_again(self, watch, tmp_path):
        await sync_dives(watch, tmp_path)
        (tmp_path / "Garmin" / "Activity" / "2026-11-01-11-30-00.fit").unlink()

        report = await sync_dives(watch, tmp_path)

        assert [f.path.name for f in report.dives] == ["2026-11-01-11-30-00.fit"]

    async def test_failed_download_is_retried_next_run(self, watch, tmp_path):
        armed = []

        def corrupt_the_first_file(entry, received, total):
            if not armed:
                armed.append(entry.index)
                watch._transport.corrupt_chunks(6)  # enough to sink this file only

        report = await sync_dives(watch, tmp_path, on_progress=corrupt_the_first_file)
        assert len(report.failed) == 1 and len(report.dives) == 2

        report = await sync_dives(watch, tmp_path)
        assert len(report.not_dives) == 1 and report.already_synced == 2

    async def test_link_drop_stops_the_run_and_keeps_what_was_saved(self, watch, tmp_path):
        def drop_on_second_file(entry, received, total):
            if entry.index == 4:
                watch._transport.simulate_drop()

        report = await sync_dives(watch, tmp_path, on_progress=drop_on_second_file)

        assert report.interrupted
        assert len(report.failed) == 1
        assert json.loads((tmp_path / STATE_FILE).read_text())["files"]  # the run before the drop

    async def test_unreadable_activity_is_kept_for_subsurface(self, tmp_path):
        """The transfer is checksummed, so if we cannot parse a file that is
        our parser's problem, and dropping what might be a dive is worse."""
        session = Watch.simulated(files=[SimulatedFile(b"??" * 50, modified=DAY)])
        async with session as watch:
            report = await sync_dives(watch, tmp_path)
        assert fit_files(tmp_path) == ["2026-11-01-09-30-00.fit"]
        assert report.dives[0].outcome == "unreadable"

    async def test_reports_results_as_they_happen(self, watch, tmp_path):
        seen = []
        await sync_dives(watch, tmp_path, on_result=seen.append)
        assert [r.outcome for r in seen] == ["not-a-dive", "dive", "dive"]


class TestFileNames:
    """Subsurface sorts on these and stores the newest as its bookmark."""

    def entry(self):
        from garmin_ble import DirectoryEntry

        return DirectoryEntry(1, 128, 4, 1, 0, 0, 10, DAY)

    def test_named_like_the_watch_names_them_on_usb(self):
        assert file_name(self.entry(), inspect(DIVE_1)) == "2026-11-01-11-30-00.fit"

    def test_falls_back_to_the_directory_date(self):
        assert file_name(self.entry(), None) == "2026-11-01-09-30-00.fit"

    def test_fits_subsurface_limits(self):
        name = file_name(self.entry(), inspect(DIVE_1))
        assert len(name) < 24 and name.endswith(".fit")

    def test_names_sort_chronologically(self):
        names = [file_name(self.entry(), inspect(d)) for d in (DIVE_2, DIVE_1)]
        assert sorted(names) == names[::-1]


class TestCli:
    def test_simulated_sync(self, tmp_path, capsys):
        assert main(["--simulate", "--out", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "3 new dive(s)" in out
        assert "Device or mount point" in out
        assert len(fit_files(tmp_path)) == 3

    def test_second_run_reports_nothing_new(self, tmp_path, capsys):
        main(["--simulate", "--out", str(tmp_path)])
        capsys.readouterr()
        assert main(["--simulate", "--out", str(tmp_path)]) == 0
        assert "0 new dive(s)" in capsys.readouterr().out

    def test_list(self, tmp_path, capsys):
        assert main(["--simulate", "--list", "--out", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "activity" in out and "monitor" in out
        assert not (tmp_path / "Garmin").exists()

    def test_record_writes_a_capture(self, tmp_path):
        capture = tmp_path / "session.gble"
        main(["--simulate", "--out", str(tmp_path / "dives"), "--record", str(capture)])
        assert capture.stat().st_size > 0
