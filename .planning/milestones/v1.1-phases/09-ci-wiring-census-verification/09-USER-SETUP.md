# Phase 09: User Setup Required

**Generated:** 2026-10-05
**Phase:** 09-ci-wiring-census-verification (plan 09-02)
**Status:** Incomplete

Complete these items for the D-06 runtime cut (and the T-09-03 loopback restore) to go LIVE. Claude automated everything possible (in-repo unit file, README, contract tests, CI wiring); applying a systemd unit on the runner host requires owner sudo — Claude has no such credentials.

## Service Configuration

- [ ] **Re-apply the in-repo ollama unit and restart the service (D-06 + T-09-03)**
  - Why: `OLLAMA_CONTEXT_LENGTH=8192` (the num_ctx 8k server-default cut) only takes effect at server START — the env is read when ollama starts, so copying the file alone changes nothing. The SAME re-apply also restores the loopback pin: the LIVE unit on the runner was probed at `OLLAMA_HOST=0.0.0.0:11434` on 2026-10-05 (real LAN exposure of an unauthenticated model server); the in-repo unit pins `127.0.0.1:11434`.
  - Location: runner host shell (owner; the runner shares `$HOME` with the dev box)
  - Run:
    ```bash
    sudo cp scripts/runner/ollama.service /etc/systemd/system/ollama.service
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    ```

## Verification

After the re-apply (read-only, no sudo):

```bash
systemctl show ollama -p Environment
# must show OLLAMA_CONTEXT_LENGTH=8192 and OLLAMA_HOST=127.0.0.1:11434 (NOT 0.0.0.0)

curl -s http://127.0.0.1:11434/api/tags
# must list qwen3.8:latest
```

Expected results:
- `Environment` shows both pins; no `0.0.0.0` anywhere in the live unit.
- The model list still contains `qwen3.8:latest` (num_ctx default changes behavior, not availability).

Plan 09-04 Task 2 asserts this read-only evidence (`systemctl show ollama -p Environment`) as a precondition before its baseline dispatch — the live service state is PENDING until this op runs.

---

**Once all items complete:** Mark status as "Complete" at top of file.
