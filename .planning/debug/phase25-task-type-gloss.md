# Debug: G-25-4 — task-type copy explains 3 of the 8 allowed values

- **Bug class:** Bohrbug (deterministic copy defect; reproducible by reading the strings)
- **Goal:** find root cause only — no source changes made
- **Repo:** /home/forrest/Github/DNALLM, branch v1.4

## Symptom

From `.planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-UAT.md` G-25-4 (Test 4):

- **Truth:** "Every task-type string names all eight allowed values in language a biologist can act on."
- **Tester:** "Eleven of the twelve hints, three of the four error sentences and both glossed headings read plainly. The task-type copy does not: every task-type string tells the reader to choose among eight values (embedding, mask, generation, binary, multiclass, multilabel, regression, token) and then explains only three of them, so a biologist whose model needs embedding, mask, generation, multilabel or regression is left to guess. Separately, the LoRA target-modules hint offers 'query, value' as its only example without saying what those are or where valid names come from, and the unknown-model sentence ends 'go back and choose a model there' without naming the screen."
- **Status:** failed, severity minor, artifacts: [], missing: [] (the UAT left the diagnosis to this session)

## Investigation

### 1. Core allows eight canonical values — CONFIRMED

`dnallm/configuration/configs.py:109` — `TaskConfig.task_type` pattern:

```
^(embedding|mask|generation|binary|binary_classification|multiclass|multi_class_classification|multilabel|multi_label_classification|regression|token|token_classification)$
```

Eight canonical values plus four long-form aliases, canonicalised in `model_post_init` (configs.py:118-130). There is **no public constant or enum** of these values; the canonical set exists only as this regex literal, duplicated privately in `dnallm/preflight/compatibility.py:97-114` (`_SUPERVISED_TASK_TYPES`, `_TASK_TYPE_ALIASES`) and `dnallm/models/model.py:1578` (`_TASK_TYPE_ALIASES`). The TUI cannot import a single source of truth today.

### 2. Four copy sites, each glosses exactly three — CONFIRMED

All in `dnallm/tui/copy.py`:

| Key | Lines | What it says |
|---|---|---|
| `SUITABILITY_TASK_PROMPT` | 252-256 | "Binary is a yes/no answer, multiclass is one of several classes, and token is a label for each DNA letter." |
| `CONFIG_FIELD_HINTS[("task","task_type")]` | 452-455 | same three glosses, prefixed "What the model should do." |
| `CONFIG_ERROR_MESSAGES[("task","task_type","not_an_allowed_value")]` | 528-533 | enumerates all eight, then glosses the same three — the worst offender |
| `CONFIG_ERROR_MESSAGES[("task","task_type","task_type_required")]` | 534-537 | glosses the same three |

The tester's phrase "every task-type string tells the reader to choose among eight values" is slightly loose: only the `not_an_allowed_value` sentence enumerates all eight; the other three never mention embedding/mask/generation/multilabel/regression at all. The actionable failure is identical — the five are never defined anywhere.

### 3. No other gloss exists in the shipped TUI — CONFIRMED (hypothesis refuted)

Searched every rendered path, not just copy.py:

- `grep -rn "embedding|mask|generation|multilabel|regression" dnallm/tui/` returns only `copy.py:492` (a comment) and `copy.py:529-530` (the enumeration itself). Nothing else.
- All user-facing text flows through `copy.py` constants: screens render `copy.*` and the shared renderer `app.py:214` (`Static(text, markup=False)`). `state.py`, `messages.py`, and `workers.py` contain docstrings only (the two failure sentences in workers.py:67,72 are unrelated).
- No help screen or `?` binding: `app.py:74` (`q` quit), `config.py:230` (escape back), `suitability.py:98` (`p` probe). No tooltip API use anywhere in `dnallm/tui/`.
- Placeholders are the only free-standing text: `entry.py:57` (config path), `data.py:72` (data path), `suitability.py:135` (`SUITABILITY_TASK_PROMPT`), `suitability.py:136` (model names). Config form inputs carry **no** placeholders (`config.py:286-289`).
- The suitability listing shows raw values in a "Task type" column (`copy.py:271-277` columns, `suitability.py:356` cell) with no legend. Values, not strings — the truth is about strings, but it is the one other place task-type words reach the screen.
- Core field descriptions never reach the TUI: no `model_fields` / `.description` use in `dnallm/tui/`.

**Conclusion:** embedding, mask, generation, multilabel and regression are explained nowhere in the shipped TUI.

### 4. Why only three — the 25-11 contract scoped it that way

`.planning/phases/25-terminal-ui-pre-flight-wizard-tui-extra/25-11-PLAN.md` wrote the three-value scope into the contract at every level:

- must-have (line 28): "the suitability task-type hint explains binary/multiclass/token"
- Task 2 contract (line 180): the exact `CONFIG_FIELD_HINTS` sentence; (line 194): the exact `SUITABILITY_TASK_PROMPT` sentence
- done (line 218): "the task-type hint explains its examples"; success criteria (line 261): "the task-type hint explains binary, multiclass and token"

The shipped strings are **verbatim** the plan's contract, and `tests/tui/test_copy.py:18-31` pins A1/A2 byte-for-byte. The plan's own GAP A contract (line 120) likewise specifies the enumerate-eight-then-gloss-three sentence. So the copy is faithful to a contract narrower than UAT truth G-25-4: it implemented exactly what was specified, and the specification was incomplete.

**Structural amplifier:** the vocabulary is duplicated as four independent prose literals with no shared source; a fix must touch four places or introduce one. Nothing in the test suite enforced completeness — `tests/tui/test_copy.py:327-330` is named "explains its examples" and asserts only the three substrings that exist; `tests/tui/test_screens.py:1001` asserts only "Binary is a yes/no answer".

**AND-gate check:** the observed failure requires only the under-scoped contract (process/config cause). Duplication is not a necessary condition for the defect (one under-scoped sentence would fail too); it is a contributing factor for the size and risk of the fix. Single root cause confirmed.

### 5. Secondary observation A — LoRA target-modules hint

`dnallm/tui/copy.py:483-485`:

> "Which parts of the model receive the add-on weights, for example query, value."

The hint glosses "modules" as "parts of the model" but leaves "query, value" undefined and never says where valid names come from. Facts established:

- Valid names are PEFT-style module-name suffixes matched against the model's real modules (`dnallm/finetune/trainer.py:218-250`).
- The only packaged source of per-family valid names is `dnallm/configuration/presets/lora_targets.yaml` (e.g. `[query, value]`, `[q_proj, v_proj]`). The TUI never reads or shows it.
- Leaving `target_modules` empty is meaningful: the trainer auto-selects the preset for the model's family (`trainer.py:476-500`; the YAML header documents the resolution). The hint does not say this.
- The inspection path is `finetune.peft_dry_run=true` (`configs.py:335`, `trainer.py:499-500`), but `peft_dry_run` is **not** one of the form fields (`dnallm/tui/screens/config.py:106-117`), so a TUI user cannot turn it on without hand-editing the file.
- The core's own field description repeats the same unexplained example (`configs.py:402-407`: "E.g., ['query', 'value'] or ['q_proj', 'v_proj']").
- The same "query, value" example is repeated in `CONFIG_ERROR_GENERIC["not_a_list"]` (`copy.py:554`) and asserted in `tests/tui/test_screens.py:2573` / `tests/tui/test_copy.py:117`.

### 6. Secondary observation B — unknown-model sentence

`dnallm/tui/copy.py:542-545` (`CONFIG_ERROR_MESSAGES[("model","model","model_not_in_registry")]`):

> "This model name is not in the model list. Check the spelling, or go back and choose a model there."

Navigation verified:

- `ConfigScreen.action_go_back` is `pop_screen()` (`config.py:452`); it returns to whatever pushed the screen.
- `model_not_in_registry` is emitted only by `generate_config` (`dnallm/preflight/config_gen.py:283-292`), i.e. only on the model route, which is pushed from the suitability panel (`suitability.py:523`). So "go back" does land on the model chooser — the sentence is factually right but names neither the screen nor its action.
- The destination has user-visible names to compose from: `SUITABILITY_PANEL_TITLE = "Which model should do the job?"` (`copy.py:246`) and `USE_MODEL_ACTION = "Use this model"` (`copy.py:335`).
- The same unanchored phrasing appears in `CONFIG_NO_SOURCE` (`copy.py:408-411`, "Go back a step and choose a model"). That one is worse: the config screen can also be pushed from the entry screen (`entry.py:86` → `app.py:415`), and the entry screen offers no model choice at all (only "Start from my data" / "Edit an existing config file", `copy.py:147-151`), so "there" may not be a place a model can be chosen.

## Root Cause

The task-type gloss was specified to three examples (binary/multiclass/token) by the 25-11 gap-closure plan's must-have and task contract, and the implementation matched that contract verbatim; UAT truth G-25-4 requires all eight allowed values, so the copy passes its own tests while failing the truth. The failure is amplified by the vocabulary being duplicated as four independent prose literals with no shared source (the core exposes no constant — the canonical eight are a regex literal at `configs.py:109` plus private sets in preflight), and by tests that assert the three present substrings rather than completeness over the eight.

The two secondary observations are the same *class* — copy in `dnallm/tui/copy.py` that leaves the reader to guess — but two distinct sub-defects: (A) an undefined identifier/example with no stated provenance (same mechanism as the task-type gap, same table `CONFIG_FIELD_HINTS`), and (B) an unanchored navigation reference that names no screen (different mechanism, same table family `CONFIG_ERROR_MESSAGES`, and repeated in `CONFIG_NO_SOURCE`). A single fix plan can cover all three because they share the file, the rendering path, and the UAT truth axis, but each needs its own content decision.

## Artifacts

Primary (the four task-type sites):

- `dnallm/tui/copy.py:252-256` — `SUITABILITY_TASK_PROMPT` glosses only binary/multiclass/token; also used as the suitability input's placeholder (`dnallm/tui/screens/suitability.py:135`), which disappears once the user types.
- `dnallm/tui/copy.py:452-455` — `CONFIG_FIELD_HINTS[("task","task_type")]` glosses only three.
- `dnallm/tui/copy.py:528-533` — `CONFIG_ERROR_MESSAGES[("task","task_type","not_an_allowed_value")]` enumerates eight and glosses three.
- `dnallm/tui/copy.py:534-537` — `CONFIG_ERROR_MESSAGES[("task","task_type","task_type_required")]` glosses three.

Structural:

- `dnallm/configuration/configs.py:109` — canonical eight live in a regex literal; no public constant.
- `dnallm/preflight/compatibility.py:97-114` — private task-type sets/aliases; not reusable.
- `dnallm/models/model.py:1578` — a third private alias table.
- `dnallm/tui/screens/config.py:98` — `_FieldSpec("task_type")` is free text, so the user must type one of the eight words.
- `dnallm/tui/screens/suitability.py:356` + `dnallm/tui/copy.py:271-277` — listing shows raw task_type values under "Task type" with no legend.

Secondary A (LoRA):

- `dnallm/tui/copy.py:483-485` — undefined "query, value".
- `dnallm/tui/copy.py:554` — `CONFIG_ERROR_GENERIC["not_a_list"]` repeats the same unexplained example.
- `dnallm/configuration/configs.py:402-407` — core description uses the same example with no provenance.
- `dnallm/configuration/presets/lora_targets.yaml` — the real valid-name table; never surfaced in the TUI.
- `dnallm/finetune/trainer.py:218-250, 476-500` — dry-run inspection and empty-means-preset behaviour that the hint omits.
- `dnallm/configuration/configs.py:335` + `dnallm/tui/screens/config.py:106-117` — `peft_dry_run` exists but is not editable in the form.

Secondary B (navigation):

- `dnallm/tui/copy.py:542-545` — "go back and choose a model there" names no screen.
- `dnallm/tui/copy.py:408-411` — `CONFIG_NO_SOURCE` repeats the unanchored phrasing.
- `dnallm/tui/copy.py:246` (`SUITABILITY_PANEL_TITLE`), `:335` (`USE_MODEL_ACTION`), `:140` (`ENTRY_PANEL_TITLE`), `:147-151` — the user-visible names available for composing an anchored sentence.

Tests that pin the current wording (must change with any fix):

- `tests/tui/test_copy.py:18-38` — A1-A4 byte-pinned; `:313-330` — hint inventory and the three-substring prompt assertion; `:117` — generic example.
- `tests/tui/test_screens.py:922` (A1 rendering), `:968-982` (hint inventory rendered), `:1000-1057` (placeholder and hint constants, literal "Binary is a yes/no answer"), `:2573` (parse example).

## Missing

- **One shared source of the eight task types and their plain glosses** in the copy layer (e.g. an ordered `TASK_TYPE_GLOSSES` mapping) used by all four task-type strings, so a future edit cannot update one site and miss three. Decide whether the canonical value list should be exported by the core (a core change) or pinned copy-side with a drift test against `TaskConfig.task_type`'s pattern.
- **A completeness guard**: a test asserting each of the eight canonical values is named and glossed in every task-type string, plus a drift test that the copy's set equals the core's allowed set (derived from `TaskConfig.model_fields["task_type"].metadata` / the pattern, with a comment pointing at `configs.py:109`). The current tests assert only that the three present glosses exist.
- **Rewritten sentences at all four sites**, still legible at 80x24 (the placeholder and hints render on the smallest supported screen; the 80x24 Pilot tests must be re-run).
- **A decision on where the eight-value legend lives in the suitability panel**: today it is an Input placeholder that vanishes the moment the user starts typing, and the config field is free text with only a one-line hint. A static legend/hint line may be required — a design decision, not only copy.
- **LoRA hint content**: what "query"/"value" are (attention projection names, not arbitrary words), and where valid names come from — leaving the field empty uses the packaged per-family preset (`lora_targets.yaml`, `trainer.py:476-500`), or `finetune.peft_dry_run=true` lists this model's names (`configs.py:335`, not editable in the form). Owner must pick the actionable advice; the current hint gives none.
- **Alignment decision for `CONFIG_ERROR_GENERIC["not_a_list"]`** (`copy.py:554`), which repeats the same unexplained "query, value" example.
- **Unknown-model sentence naming the destination**, composed from the existing constants (`SUITABILITY_PANEL_TITLE`, `USE_MODEL_ACTION`) so the name cannot drift; simultaneously fix `CONFIG_NO_SOURCE` (`copy.py:408-411`), which is wrong-or-unanchored on the entry route where no model can be chosen.
- **Update the byte-pinned tests** (`test_copy.py` A1-A4, the task-type hint assertion; `test_screens.py` literal placeholder assertion) in the same change; the TUI test suite holds every module at ≥96% (commit 075ce5a), so new copy tables need coverage.
- **Scope confirmation**: whether the truth also covers the suitability listing's raw "Task type" / "Trained outputs" column values (`copy.py:271-277`, `suitability.py:356`). They are values rather than strings, but they are the other place the eight words reach a biologist unglossed.

## Confidence

- **Q1 (no other gloss exists): high.** Exhaustive grep over `dnallm/tui/**` for all five missing terms returns only the enumeration itself; every rendered Static/Input placeholder is accounted for; no tooltip/help surface exists.
- **Q2 (root cause): high.** The plan contract lines, the byte-pinned tests, and the verbatim implementation line up; the duplication and missing completeness test explain why four sites drifted together.
- **Q3 (secondary observations): high on facts and line numbers; medium-high on the "same class" judgment.** Both live in `copy.py`, share the rendering path and the UAT axis, but are distinct mechanisms (undefined identifier vs unanchored destination). The LoRA fix also needs a product decision the current code does not answer on screen.
- **Q4 (missing list): high-level complete;** exact wording of the replacement sentences is a copy decision for the plan, not a diagnostic one.

No source files were modified. This note is the only file written.
