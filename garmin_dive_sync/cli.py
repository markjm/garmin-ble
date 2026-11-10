"""``garmin-dive-sync``: the command-line front end."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from garmin_ble import (
    DirectoryEntry,
    GarminBleError,
    HandshakeError,
    SimulatedFile,
    Watch,
    WatchNotFound,
    WatchSession,
)
from garmin_ble.logging import configure as configure_logging

from .fit import make_activity
from .sync import SyncedFile, SyncReport, sync_dives

DEFAULT_OUT = Path("~/GarminDives")

CONNECTION_HELP = """\
Things to check:
  * Garmin watches talk to one Bluetooth client at a time. Turn Bluetooth off
    on your phone, or force-quit Garmin Connect and Garmin Dive, then retry.
  * The first time, the watch has to be paired with this computer. On the
    watch, open Settings > Connectivity > Phone > Pair Phone, then pair it
    from your computer's Bluetooth settings while that screen is showing.
  * `garmin-dive-sync --scan` lists every Bluetooth device in range. If the
    watch shows up under an unexpected name, pass --name or --address.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garmin-dive-sync",
        description="Download dives from a Garmin Descent over Bluetooth into "
        "a folder that Subsurface can import.",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT, metavar="DIR",
        help=f"where to put the dives (default: {DEFAULT_OUT})",
    )
    where = parser.add_argument_group("finding the watch")
    where.add_argument("--address", help="connect to this Bluetooth address, skipping the scan")
    where.add_argument("--name", metavar="GLOB", help="connect to the device whose name matches, e.g. 'Descent*'")
    where.add_argument("--scan-timeout", type=float, default=30.0, metavar="SECONDS", help="how long to look for the watch (default: 30)")
    where.add_argument("--scan", action="store_true", help="list Bluetooth devices in range and exit")

    what = parser.add_argument_group("what to sync")
    what.add_argument("--list", action="store_true", help="show the files on the watch and exit")
    what.add_argument("--newest", type=int, metavar="N", help="only consider the N most recent activities on the watch")
    what.add_argument("--redownload", action="store_true", help="fetch activities again even if an earlier run handled them")
    what.add_argument("--keep-others", action="store_true", help="also keep activities that are not dives, in other-activities/")

    debug = parser.add_argument_group("troubleshooting")
    debug.add_argument("--simulate", action="store_true", help="use a built-in pretend watch instead of Bluetooth")
    debug.add_argument("--record", type=Path, metavar="FILE", help="write every Bluetooth packet to FILE, to attach to a bug report")
    debug.add_argument("-v", "--verbose", action="store_true", help="log protocol details")
    return parser


def _simulated_files() -> List[SimulatedFile]:
    """A few dives and a run, for trying the tool without a watch."""
    day = datetime(2026, 11, 1, 9, 30, tzinfo=timezone.utc)
    activities = [
        (day, 46, 28.4),
        (day + timedelta(hours=4), 52, 18.1),
        (day + timedelta(days=1, hours=-1), 35, None),  # a morning run
        (day + timedelta(days=1, hours=1), 61, 32.7),
    ]
    files = [
        SimulatedFile(
            make_activity(start, minutes, depth, utc_offset_hours=-10),
            modified=start,
        )
        for start, minutes, depth in activities
    ]
    # A watch holds plenty that is not an activity; the sync must ignore it.
    files.append(SimulatedFile(b"\x00" * 800, modified=day, sub_type=32))
    files.append(SimulatedFile(b"\x00" * 300, modified=day, sub_type=2))
    return files


def _session(args: argparse.Namespace) -> WatchSession:
    if args.simulate:
        return Watch.simulated(profile="descentmk3", files=_simulated_files())
    if args.address:
        return Watch.at(args.address, reconnect="off")
    if args.name:
        return Watch.named(args.name, timeout=args.scan_timeout, reconnect="off")
    return Watch.discover(timeout=args.scan_timeout, reconnect="off")


class _Progress:
    """A one-line download meter, drawn only when someone is watching."""

    def __init__(self) -> None:
        self._live = sys.stderr.isatty()
        self._started = 0.0
        self._last_draw = 0.0
        self._index: Optional[int] = None

    def update(self, entry: DirectoryEntry, received: int, total: int) -> None:
        if not self._live:
            return
        now = time.monotonic()
        if entry.index != self._index:
            self._index, self._started, self._last_draw = entry.index, now, 0.0
        if now - self._last_draw < 0.2 and received < total:
            return
        self._last_draw = now
        rate = received / max(now - self._started, 0.001) / 1024
        percent = 100 * received // max(total, 1)
        sys.stderr.write(
            f"\r  {entry.modified:%Y-%m-%d %H:%M}  {percent:3d}%  "
            f"{received // 1024:,}/{total // 1024:,} KB  {rate:.1f} KB/s "
        )
        sys.stderr.flush()

    def clear(self) -> None:
        if self._live:
            sys.stderr.write("\r" + " " * 72 + "\r")
            sys.stderr.flush()


def _print_result(result: SyncedFile, progress: _Progress) -> None:
    progress.clear()
    if result.outcome == "dive":
        print(f"  + {result.summary.describe()}  ->  {result.path.name}")
    elif result.outcome == "unreadable":
        print(f"  ? could not read {result.path.name}; kept it for Subsurface to try")
    elif result.outcome == "not-a-dive":
        kept = f"  ->  {result.path}" if result.path else ""
        print(f"  - {result.summary.describe()}{kept}")
    else:
        print(f"  ! file #{result.entry.index} ({result.entry.modified:%Y-%m-%d %H:%M}) failed: {result.error}")


def _print_summary(report: SyncReport) -> int:
    dives, failed = report.dives, report.failed
    print()
    print(
        f"{len(dives)} new dive(s), {len(report.not_dives)} other activit(ies) skipped, "
        f"{report.already_synced} already synced, {len(failed)} failed."
    )
    if report.interrupted:
        print(f"Stopped early: {report.interrupted}")
        print("Run it again to carry on from here; finished files are not fetched twice.")
    if dives:
        print()
        print("To import into Subsurface:")
        print("  1. Import > Import from dive computer")
        print("  2. Vendor: Garmin    Product: Descent Mk3(i)")
        print(f"  3. Device or mount point: {report.out_dir}")
        print("  4. Download")
    return 1 if failed or report.interrupted else 0


def _print_listing(entries: List[DirectoryEntry]) -> None:
    print(f"{'index':>5}  {'type':<14} {'size':>10}  {'recorded (UTC)':<19}  flags")
    for entry in sorted(entries, key=lambda e: e.modified):
        flags = "archived" if entry.is_archived else ""
        print(
            f"{entry.index:>5}  {entry.kind:<14} {entry.size:>10,}  "
            f"{entry.modified:%Y-%m-%d %H:%M:%S}  {flags}"
        )
    activities = sum(1 for entry in entries if entry.is_activity)
    print(f"\n{len(entries)} file(s), {activities} of them activities (dives are activities).")


async def _scan(timeout: float) -> int:
    from bleak import BleakScanner

    from garmin_ble.transport.ble import looks_like_garmin

    print(f"Scanning for {timeout:.0f}s...")
    found = await BleakScanner.discover(timeout=timeout, return_adv=True)
    for address, (device, adv) in sorted(found.items()):
        name = device.name or getattr(adv, "local_name", None)
        mark = "*" if looks_like_garmin(name) else " "
        print(f" {mark} {name or '<unnamed>':<32} {address}")
    print(f"\n{len(found)} device(s). Ones marked * look like a Garmin watch.")
    return 0


async def run(args: argparse.Namespace) -> int:
    if args.scan:
        return await _scan(args.scan_timeout)

    if not args.simulate:
        print("Looking for the watch...")
    async with _session(args) as watch:
        if args.record:
            watch.record(args.record)
        print(f"Connected to {watch.info.name}.")

        if args.list:
            _print_listing(await watch.files())
            return 0

        progress = _Progress()
        print("Reading the list of files on the watch...")
        report = await sync_dives(
            watch,
            args.out,
            newest=args.newest,
            redownload=args.redownload,
            keep_others=args.keep_others,
            on_progress=progress.update,
            on_result=lambda result: _print_result(result, progress),
        )
        return _print_summary(report)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=logging.DEBUG if args.verbose else logging.WARNING)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted. Run it again to carry on; finished files are not fetched twice.")
        return 130
    except (WatchNotFound, HandshakeError) as exc:
        print(f"Could not connect: {exc}\n\n{CONNECTION_HELP}", file=sys.stderr)
        return 2
    except GarminBleError as exc:
        print(f"Sync failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
