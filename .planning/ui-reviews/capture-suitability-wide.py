"""Capture the suitability listing at a wide size so ALL six columns fit.

Validation evidence for Phase 27.2 (validate-phase): the shipped Support
column (plan 27.2-05) is real but sits ~170 chars of table width out, past
the 74-114 char viewports measured in 27.2-UI-REVIEW.md (D2 WARNING). This
capture renders the same real registry listing at 176x42 so every column --
including Support -- is visible in one frame. Same offline stubs as the other
captures: no network.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/home/forrest/Github/DNALLM")

import dnallm.preflight as preflight_facade
from dnallm.models.registry import all_entries
from dnallm.preflight import AvailabilityReport, AvailabilityState
from dnallm.preflight.compatibility import _build_listing
from dnallm.preflight.report import CompatibilityReport
from dnallm.tui.app import DNALLMPreflightApp
from dnallm.tui.screens.suitability import SuitabilityScreen

OUT = "/home/forrest/Github/DNALLM/.planning/ui-reviews/suitability-176x42-all-columns.svg"


class FakeCollector:
    def collect_versions(self):
        return []

    def collect_device(self):
        return []

    def collect_disk(self):
        return []


class FakeProber:
    def __call__(self, model):
        return AvailabilityReport(state=AvailabilityState.UNKNOWN, reason="capture")


def fake_local(model):
    return AvailabilityReport(state=AvailabilityState.UNKNOWN, reason="capture")


async def main():
    listing = _build_listing(all_entries())
    words: dict[str, int] = {}
    for e in listing:
        words[e.support] = words.get(e.support, 0) + 1
    print("listing rows:", len(listing))
    for w, n in sorted(words.items(), key=lambda kv: -kv[1]):
        print(f"  support word {w!r} x{n}")
    report = CompatibilityReport(verdicts=[], listing=listing)

    class FakeCore:
        def __call__(self, data=None, *, task_type=None, models=None, intent=None,
                     prober=None, probe_shortlist=None):
            return report

    preflight_facade.check_compatibility = FakeCore()
    preflight_facade.local_availability = fake_local
    app = DNALLMPreflightApp(collector=FakeCollector(), prober=FakeProber())
    async with app.run_test(size=(176, 42)) as pilot:
        app.push_screen(SuitabilityScreen())
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SuitabilityScreen)
        table = screen.query_one("#suitability-listing")
        print("table region:", table.region, "virtual width:", table.virtual_size.width)
        svg = app.export_screenshot(title="dnallm preflight - suitability (176x42, all six columns)")
        with open(OUT, "w") as fh:
            fh.write(svg)
        header = [str(c.label) for c in table.columns.values()]
        print("columns:", header)
        assert header[-1] == "Support", header
        # prove support cells rendered on-screen in the export
        assert "Support" in svg and "family-tested" in svg
        print(f"wrote {OUT} ({len(svg)} bytes)")


asyncio.run(main())
