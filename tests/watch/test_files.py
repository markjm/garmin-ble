"""File listing and download, end to end against the simulated watch."""
import asyncio
import dataclasses
import os
from datetime import datetime, timezone

import pytest

from garmin_ble import ConnectionFailed, SimulatedFile, TransferError, Watch
from garmin_ble.constants import GarminMessage
from garmin_ble.protocol import gfdi
from garmin_ble.transport.simulated import SimulatedTransport, get_profile
from garmin_ble.watch import WatchSession

BIG = os.urandom(40_000)
SMALL = os.urandom(300)


@pytest.fixture
async def dive_watch():
    files = [
        SimulatedFile(BIG, modified=datetime(2026, 11, 1, tzinfo=timezone.utc)),
        SimulatedFile(SMALL, modified=datetime(2026, 11, 2, tzinfo=timezone.utc), sub_type=32),
        SimulatedFile(b""),
    ]
    session = Watch.simulated(profile="descentmk3", files=files)
    watch = await session.open()
    try:
        yield watch
    finally:
        await session.aclose()


class TestListing:
    async def test_lists_every_file(self, dive_watch):
        entries = await dive_watch.files()
        assert [e.index for e in entries] == [1, 2, 3]
        assert [e.size for e in entries] == [len(BIG), len(SMALL), 0]
        assert [e.is_activity for e in entries] == [True, False, True]
        assert entries[0].modified == datetime(2026, 11, 1, tzinfo=timezone.utc)

    async def test_watch_with_no_files(self, watch):
        assert await watch.files() == []


class TestDownload:
    async def test_downloads_exact_contents(self, dive_watch):
        entries = await dive_watch.files()
        assert await dive_watch.download(entries[0]) == BIG
        assert await dive_watch.download(entries[1]) == SMALL

    async def test_accepts_a_bare_index(self, dive_watch):
        assert await dive_watch.download(2) == SMALL

    async def test_empty_file(self, dive_watch):
        assert await dive_watch.download(3) == b""

    async def test_reports_progress(self, dive_watch):
        seen = []
        await dive_watch.download(1, progress=lambda got, total: seen.append((got, total)))
        assert seen[-1] == (len(BIG), len(BIG))
        assert [got for got, _ in seen] == sorted(got for got, _ in seen)
        assert len(seen) > 50  # arrives in many chunks, not one

    async def test_unknown_file_is_refused(self, dive_watch):
        with pytest.raises(TransferError, match="index_unknown") as caught:
            await dive_watch.download(99)
        assert caught.value.file_index == 99

    async def test_concurrent_downloads_are_serialised(self, dive_watch):
        first, second = await asyncio.gather(dive_watch.download(1), dive_watch.download(2))
        assert (first, second) == (BIG, SMALL)

    async def test_survives_a_heartbeat_mid_transfer(self, dive_watch):
        """Time syncs share the channel with file data and must not corrupt it."""
        async def chatter():
            for _ in range(20):
                await dive_watch.sync_time()
                await asyncio.sleep(0)

        data, _ = await asyncio.gather(dive_watch.download(1), chatter())
        assert data == BIG

    async def test_small_mtu(self):
        """At the BLE default MTU every chunk spans many notifications."""
        profile = dataclasses.replace(get_profile("descentmk3"), mtu=23)
        transport = SimulatedTransport(profile=profile, files=[SimulatedFile(BIG[:5000])])
        async with WatchSession(lambda: transport, heartbeat=None, reconnect="off") as watch:
            assert await watch.download(1) == BIG[:5000]


class TestFailures:
    async def test_recovers_from_a_corrupted_chunk(self, dive_watch):
        dive_watch._transport.corrupt_chunks(2)
        assert await dive_watch.download(1) == BIG

    async def test_gives_up_on_persistent_corruption(self, dive_watch):
        dive_watch._transport.corrupt_chunks(1000)
        with pytest.raises(TransferError, match="crc_mismatch"):
            await dive_watch.download(1)

    async def test_next_download_works_after_a_failure(self, dive_watch):
        dive_watch._transport.corrupt_chunks(6)
        with pytest.raises(TransferError):
            await dive_watch.download(1)
        assert await dive_watch.download(2) == SMALL

    async def test_stalled_transfer_times_out(self, dive_watch):
        dive_watch._transport.stall_downloads()
        with pytest.raises(TransferError, match="stalled at byte 0"):
            await dive_watch.download(1, timeout=0.2)

    async def test_link_drop_mid_download(self, dive_watch):
        def drop(received, total):
            if received > 5000:
                dive_watch._transport.simulate_drop()

        with pytest.raises(ConnectionFailed, match="link lost"):
            await dive_watch.download(1, progress=drop)

    async def test_unrequested_data_is_aborted_not_fatal(self, dive_watch):
        from garmin_ble.protocol import files as ft

        dive_watch._on_gfdi_message(
            gfdi.parse_message(ft.build_file_transfer_data(b"stray", 0, 0))
        )
        await asyncio.sleep(0.05)
        assert await dive_watch.download(2) == SMALL


class TestIntroduction:
    async def test_host_introduces_itself_before_syncing(self, dive_watch):
        """The watch only sends its capabilities once the host has replied to
        its device information, so reaching `configured` proves both replies."""
        await asyncio.wait_for(dive_watch._configured.wait(), 2)

    async def test_watch_identity_comes_from_device_information(self, dive_watch):
        await asyncio.wait_for(dive_watch._configured.wait(), 2)
        assert dive_watch.info.model == "Descent Mk3i - 51mm"
        assert dive_watch.info.firmware == "16.21"

    async def test_download_does_not_wait_forever_for_a_quiet_watch(self, dive_watch, monkeypatch):
        """A watch that never sends capabilities still gets asked for the file."""
        dive_watch._configured.clear()
        monkeypatch.setattr(Watch, "CONFIGURATION_TIMEOUT", 0.05)
        assert await dive_watch.download(2) == SMALL

    async def test_file_data_gets_its_own_acknowledgement(self, dive_watch):
        """File chunks are answered with a transfer status, never a bare ACK."""
        sent = []
        original = dive_watch._send_gfdi

        async def spy(frame):
            sent.append(gfdi.parse_message(frame))
            await original(frame)

        dive_watch._send_gfdi = spy
        await dive_watch.download(2)
        answers = [m for m in sent if m.type == GarminMessage.RESPONSE]
        assert answers and all(len(m.payload) - 2 == 8 for m in answers)
