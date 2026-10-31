---
phase: 22-registry-foundation-runtime-loader-enrichment
reviewed: 2026-10-20T12:28:41Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - dnallm/models/__init__.py
  - dnallm/models/model_info.yaml
  - dnallm/models/registry.py
  - tests/models/test_model_info_registry.py
  - tests/models/test_registry.py
findings:
  critical: 0
  warning: 3
  info: 6
  total: 9
status: issues_found
---

# Phase 22: Code Review Report

**Reviewed:** 2026-10-20T12:28:41Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Reviewed the Phase 22 registry work: the new `dnallm/models/registry.py` loader (sole
parser of the packaged `model_info.yaml`, lru_cache single-parse, typed frozen Pydantic
entries, base-model inheritance resolution), the facade re-exports in
`dnallm/models/__init__.py`, the additive `tokenizer_type` / `max_length` enrichment of
`model_info.yaml` (49 pretrained entries enriched, 14 explicit finetuned overrides on the
external-based plantcad repos, end-of-file provenance block), and the two test files
(57 tests). The diff is purely additive (1369 insertions, 0 deletions), matching the
plan's claims.

Verified directly: all 57 phase tests pass; the full `tests/models/` suite passes
(527 passed); `ruff check` and `ruff format --check` pass on all five files; the
sole-parser grep guard allowlist matches reality (`dnallm/mcp/server.py:821` is the one
stale-comment mentioner); the committed YAML has no duplicate mapping keys, no
whitespace-polluted ids, and exactly 49/187 sections and 189 lab entries as pinned.

No critical issues. The loader is genuinely read-only (no write/serialize path, safe
loaders only, importlib.resources access is wheel-safe), validation is loud on the
paths it covers, and the inheritance resolution (own value -> base chain -> None, with
a visited-set cycle guard) is correct — traced through override, inheritance,
unresolvable-base, and cycle cases. The three warnings are robustness gaps in the
hard-validation contract the module itself documents (duplicate YAML keys silently
last-wins; internally inconsistent task metadata accepted; preview-only ruff
suppression that silently no-ops outside preview mode).

## Warnings

### WR-01: S506 suppression uses a preview-only directive that silently no-ops without `preview = true`

**File:** `dnallm/models/registry.py:218`
**Issue:** The line

```python
data = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))  # ruff: ignore[unsafe-yaml-load]
```

is suppressed via `# ruff: ignore[unsafe-yaml-load]`, which ruff only honors when
preview mode is on (the project sets `preview = true` in `pyproject.toml`, so CI passes
today). Verified experimentally with ruff 0.16.10: with `--isolated --select S506`
(preview off) the error fires on this exact line, directive and all; with
`--preview` it is suppressed. The `S` band is selected project-wide, so any run of
ruff without the project config (editor LSP with isolated settings, `ruff check
--no-project`, or simply dropping `preview = true` once the feature stabilizes) makes
S506 fire and the lint gate go red — with nothing in the repo explaining the coupling.
The loader itself is safe (CSafeLoader/SafeLoader only), so this is a tooling-fragility
defect, not a security one.
**Fix:** Use the stable, always-honored suppression syntax:

```python
data = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))  # noqa: S506
```

(`# noqa: S506` works with and without preview; the comment above the line explaining
the getattr indirection can stay.)

### WR-02: Duplicate YAML mapping keys are silently last-wins — a repeated scripted insertion would override a field instead of failing loudly

**File:** `dnallm/models/registry.py:218` (and `tests/models/test_model_info_registry.py:93`)
**Issue:** The module's charter (docstring, D-01b) is that "a malformed entry ... fails
loudly as a descriptive ValueError naming the offender instead of corrupting downstream
behavior", and maintenance is "scripted text insertion, never a YAML round-trip". But
both the loader (`yaml.load` with CSafeLoader/SafeLoader) and the integrity tests
(`yaml.safe_load`) silently keep the *last* value of a duplicated mapping key. If the
insertion script ever runs twice over the same entry (e.g. re-running the 22-02
enrichment after a partial revert), the second `max_length:`/`tokenizer_type:` line
would silently replace the first — schema validation (`extra="forbid"`, `gt=0`) never
sees it, and the spot pins only cover 5 repos. Verified the committed file currently
has zero duplicate keys (custom duplicate-rejecting loader), so this is a detection
gap, not active corruption — but it is precisely the class of corruption the loud-
validation contract exists to catch.
**Fix:** Reject duplicate keys at parse time in the loader (belt) and/or the integrity
tests (suspenders):

```python
class _NoDuplicateKeyLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
    pass

def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.YAMLError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping

_NoDuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)
```

A `yaml.YAMLError` raised here already lands in the existing except clause and surfaces
as the documented descriptive `ValueError`.

### WR-03: TaskInfo accepts internally inconsistent entries — `num_labels: 92` with `label_names: []` validates cleanly

**File:** `dnallm/models/model_info.yaml:1810-1811` (small/medium/large variants; validation gap in `dnallm/models/registry.py:53-89`)
**Issue:** The three `plantcad/cell_type_specific_acr_plantcad2_{small,medium,large}`
entries declare `num_labels: 92` together with `label_names: []`. The new typed layer
(`TaskInfo`) validates `label_names: list[str] | None` and accepts the empty list, so
this phase's validation rubber-stamps an entry that implies 92 label names but ships
zero. The data is pre-existing (present verbatim at the diff base), but this phase is
what blesses it as a typed, published contract for Phase 24+ consumers, and it
contradicts the file's own enrichment discipline (D-03: "a gap is an absent key, never
null/empty/zero" — `[]` is the degenerate form of an absent key). A cross-field check
would surface it loudly instead.
**Fix:** Add a `TaskInfo.model_validator` enforcing that when both are present,
`len(label_names) in (num_labels, 1)` (1 covers the regression bare-string wrap), or
min-length-1 on `label_names`; commit `label_names: []` -> remove the key (absent) for
the three plantcad entries in the same change, since the value carries no information.

## Info

### IN-01: A stray committed `kind:` key is silently overridden instead of rejected

**File:** `dnallm/models/registry.py:168`
**Issue:** `RegistryEntry.model_validate({**raw_entry, "kind": section})` overwrites
any `kind` value already present in the file. Because `kind` is a valid schema field,
`extra="forbid"` cannot fire, so a hand-edited entry carrying `kind: pretrained` inside
the `finetuned:` section would be silently coerced rather than rejected — a small hole
in the loud-validation contract.
**Fix:** Check and reject explicitly before merging:

```python
if "kind" in raw_entry:
    raise ValueError(f"... entry {raw_entry.get('model')!r} carries a reserved 'kind' key ...")
```

### IN-02: Unknown top-level YAML sections are silently ignored

**File:** `dnallm/models/registry.py:149-156`
**Issue:** Only `pretrained` and `finetuned` are read; any additional top-level key (a
typo'd third section, or entries accidentally pasted under a new heading) is dropped
without error. The 49/187 count pins in the integrity tests partially mitigate this,
but the loader itself stays silent.
**Fix:** After section extraction, `if set(data) - {"pretrained", "finetuned"}: raise ValueError(...)` naming the unexpected sections.

### IN-03: `UnicodeDecodeError` / generic `OSError` escape the documented ValueError contract

**File:** `dnallm/models/registry.py:219-220`
**Issue:** `_load_registry` documents "Raises ValueError: If the packaged resource is
missing, unparseable, or malformed" but the except tuple is
`(FileNotFoundError, ModuleNotFoundError, yaml.YAMLError)`. A non-UTF-8 file raises
`UnicodeDecodeError`; permission/IO troubles raise `OSError` subclasses — both leak
raw. Practically unlikely for a committed packaged file; contract nit.
**Fix:** Broaden the tuple to `(OSError, UnicodeDecodeError, yaml.YAMLError)` (`FileNotFoundError` is an `OSError`).

### IN-04: Stale docstring claim: "The registry YAML has no runtime consumer inside the package"

**File:** `tests/models/test_model_info_registry.py:7-9`
**Issue:** The module docstring's rationale ("wrong values mislead without any runtime
error — these structure tests are the enforcement layer") predates Phase 22-01:
`dnallm.models.registry` is now a runtime consumer, and a wrong `max_length` reaches
runtime behavior via `effective_max_length`. The claim misleads future maintainers
about what the enforcement layer actually is.
**Fix:** Reword to "the registry YAML had no runtime consumer when these tests landed
(quick task 261020-kwt); since Phase 22 the runtime loader parses it, and these
structure tests remain the data-level enforcement layer above the schema."

### IN-05: 120-character line exceeds the documented 100-char style

**File:** `dnallm/models/registry.py:218`
**Issue:** The `yaml.load` line with its trailing suppression comment is 120 characters
(86 + 34). Nothing fails (`E501` is not in the selected rule set; the formatter cannot
split trailing comments), but it violates the enforced-conventions text in
CLAUDE/CONTRIBUTING (line-length 100) that all other new code in this phase follows.
**Fix:** Move the suppression onto the preceding comment block lines, or split after switching to `# noqa: S506` (shorter directive likely brings the line under 100 anyway).

### IN-06: REG-03 file-mode guard misses append modes

**File:** `tests/models/test_registry.py:538`
**Issue:** `FILE_MODE_PATTERN = re.compile(r"^[rwx][rwx+bt]*$")` does not match strings
starting with `a`, so an `open(path, "a")`/`"ab"`/`"a+"` literal in registry.py would
slip past `test_registry_module_source_has_no_yaml_emitter`'s mode check, weakening the
born-read-only guard.
**Fix:** `re.compile(r"^[rwxa][rwxabt+]*$")` (append is a write-side mode; keep `r` the
only allowed literal in the assertion).

---

_Reviewed: 2026-10-20T12:28:41Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
