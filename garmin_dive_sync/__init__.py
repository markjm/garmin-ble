"""Sync dives from a Garmin Descent to Subsurface over Bluetooth.

Garmin dive computers only hand their logs to Garmin's own apps over
Bluetooth, and Subsurface can only read them from a USB cable. This package
closes the gap: it downloads the dive files from the watch over BLE and lays
them out on disk the way the watch's USB storage looks, which is a layout
Subsurface already knows how to import.

    garmin-dive-sync                # download new dives to ~/GarminDives
    garmin-dive-sync --list         # just show what is on the watch

It is built on :mod:`garmin_ble` and kept out of it on purpose: the library
speaks the wire protocol, and this is one thing to do with it.
"""

from .fit import FitError, FitSummary, inspect
from .sync import SyncedFile, SyncReport, sync_dives

__all__ = [
    "sync_dives",
    "SyncReport",
    "SyncedFile",
    "inspect",
    "FitSummary",
    "FitError",
]
