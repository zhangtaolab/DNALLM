# Deferred Items

- `.planning/WINDOWS.md` fenced-JSON vs rendered-table mismatch (row id 28) blocks `gsd_run windows append`
  status: resolved
  **Status:** resolved (2026-10-22, v1.4 milestone close)
  **What:** Discovered 2026-10-20 during 22-02 close-out when appending a deviation entry; the ledger tool refuses writes until the JSON block and table agree. Pre-existing planning-artifact state, unrelated to phase code; the 22-02 deviation context is fully recorded in 22-02-SUMMARY.md instead.
  **Resolution:** JSON entry 28 inserted mirroring the rendered table row field-for-field (28 entries on both sides, ids 1-28 aligned); ledger writes unblocked.
