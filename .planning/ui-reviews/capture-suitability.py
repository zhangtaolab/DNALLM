"""Headless visual capture of the suitability listing at 80x24 and 120x40.

Uses the REAL registry listing (all_entries -> _build_listing) with fake
probe/local helpers so nothing touches the network. Exports SVG screenshots
and dumps per-column widths + first rows' rendered cell text for the
legibility audit (Phase 27.2 D-16 / D2).
"""

from __future__ import annotations

import asyncio
import sys
import threading

sys.path.insert(0, "/home/forrest/Github/DNALLM")

import dnallm.preflight as preflight_facade
from dnallm.models.registry import all_entries
from dnallm.preflight import AvailabilityReport, AvailabilityState, Hub
from dnallm.preflight.compatibility import _build_listing
from dnallm.preflight.report import CompatibilityReport, ModelVerdict
from dnallm.tui.app import DNALLMPreflightApp
from dnallm.tui.screens.suitability import SuitabilityScreen


class FakeCollector:
    def collect_versions(self):
        return []

    def collect_device(self):
        return []

    def collect_disk(self):
        return []


class FakeProber:
    def __init__(self):
        self.calls = []

    def __call__(self, model):
        self.calls.append(model)
        return AvailabilityReport(
            state=AvailabilityState.REMOTE_CONFIRMED,
            size_bytes=1,
            revision="rev-1",
            source_hub=Hub.MODELSCOPE,
        )


def _real_report():
    listing = _build_listing(all_entries())
    print(f"listing rows: {len(listing)}")
    # Show the worst-case support words present
    words = {}
    for e in listing:
        words[e.support] = words.get(e.support, 0) + 1
    for w, n in sorted(words.items(), key=lambda kv: -kv[1]):
        print(f"  support word: {w!r} x{n}")
    return CompatibilityReport(verdicts=[], listing=listing)


REPORT = _real_report()


class FakeCore:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def __call__(self, data=None, *, task_type=None, models=None, intent=None,
                 prober=None, probe_shortlist=None):
        self.calls.append(intent)
        return self.report


def fake_local(model):
    return AvailabilityReport(state=AvailabilityState.UNKNOWN, reason="RATE_LIMITED")


async def main(outdir):
    preflight_facade.check_compatibility = FakeCore(REPORT)
    preflight_facade.local_availability = fake_local
    for w, h, name in ((80, 24, "suitability-80x24"), (120, 40, "suitability-120x40")):
        app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())
        async with app.run_test(size=(w, h)) as pilot:
            app.push_screen(SuitabilityScreen())
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SuitabilityScreen)
            table = screen.query_one("#suitability-listing")
            svg = app.export_screenshot()
            with open(f"{outdir}/{name}.svg", "w") as fh:
                fh.write(svg)
            print(f"\n== {name} (terminal {w}x{h}) ==")
            print(f"table region: {table.region}")
            print(f"table total width: {table.virtual_size.width}")
            for col_key, col in table.columns.items():
                label = str(col.label)
                print(f"  column {label!r}: width={col.width}")
            for i in range(min(4, table.row_count)):
                print(f"  row {i}: {[c[:40] for c in map(str, table.get_row_at(i))]}")
            # Longest support words in the rendered rows
            sup_idx = 5
            words = [str(table.get_row_at(i)[sup_idx]) for i in range(table.row_count)]
            print(f"  distinct support cells visible: {sorted(set(words))}")
            await pilot.pause()


asyncio.run(main(sys.argv[1]))
