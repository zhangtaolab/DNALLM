#!/usr/bin/env python3
"""
DNALLM Documentation Verification Script

Automatically extracts code blocks from Markdown files and Jupyter notebooks,
executes them in isolated subprocesses with timeout protection, and generates
structured verification reports.

Usage:
    python scripts/verify_docs.py
    python scripts/verify_docs.py --verbose
    python scripts/verify_docs.py --timeout-base 60 --timeout-heavy 300
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from dnallm.utils import get_logger

try:
    import nbformat
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False
# Notebook execution uses subprocess (jupyter nbconvert CLI) to avoid ZMQ issues

logger = get_logger("dnallm.verify_docs")

# YAML config prefixes to skip
YAML_PREFIXES = (
    "task:",
    "finetune:",
    "inference:",
    "server:",
    "mcp:",
    "lora:",
    "models:",
    "multi_model:",
    "logging:",
    "training_args:",
)

# AST-detected heavyweight operations → extended timeout
HEAVYWEIGHT_NAMES = {
    "load_model_and_tokenizer",
    "from_pretrained",
    "train",
    "fit",
    "run",
}
HEAVYWEIGHT_ATTRS = {
    "train",
    "fit",
    "run",
    "infer",
}
HEAVYWEIGHT_MODULE_CALLS = {
    ("wandb", "init"),
    ("accelerate", "launch"),
}


@dataclass
class BlockResult:
    """Result of verifying a single code block."""

    file: str
    line: int
    language: str
    status: str  # PASS, FAIL, SKIP, TIMEOUT
    error: str = ""
    execution_time_ms: int = 0
    skip_reason: str = ""
    cell_index: int | None = None


@dataclass
class NotebookResult:
    """Result of verifying a notebook."""

    file: str
    cell_results: list[dict[str, Any]] = field(default_factory=list)
    overall_status: str = "PASS"  # PASS, FAIL, SKIP, TIMEOUT


@dataclass
class VerificationReport:
    """Complete verification report."""

    timestamp: str
    summary: dict[str, int]
    results: list[dict[str, Any]]
    notebooks: list[dict[str, Any]]


class DocVerifier:
    """Documentation code verifier."""

    def __init__(
        self,
        project_root: Path | None = None,
        docs_dir: Path | None = None,
        notebooks_dir: Path | None = None,
        timeout_base: int = 30,
        timeout_heavy: int = 120,
        verbose: bool = False,
    ) -> None:
        """Initialize verifier.

        Args:
            project_root: Repository root (default: parent of scripts/).
            docs_dir: Docs directory override.
            notebooks_dir: Notebooks directory override.
            timeout_base: Base timeout in seconds.
            timeout_heavy: Extended timeout for heavyweight operations.
            verbose: Print detailed progress.
        """
        self.verbose = verbose
        self.timeout_base = timeout_base
        self.timeout_heavy = timeout_heavy

        if project_root is None:
            # This script lives in scripts/ under repo root
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = project_root.resolve()

        self.docs_dir = (docs_dir or self.project_root / "docs").resolve()
        self.notebooks_dir = (
            notebooks_dir or self.project_root / "example" / "notebooks"
        ).resolve()

        self.block_results: list[BlockResult] = []
        self.notebook_results: list[NotebookResult] = []

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------
    def _is_inside_project(self, path: Path) -> bool:
        """Validate that a resolved path is within the project root."""
        try:
            resolved = path.resolve()
            root = self.project_root.resolve()
            return str(resolved).startswith(str(root))
        except (OSError, ValueError):
            return False

    def discover_markdown_files(self) -> list[Path]:
        """Discover all Markdown files to verify."""
        files: list[Path] = []

        # docs/ recursively
        if self.docs_dir.exists():
            for p in self.docs_dir.rglob("*.md"):
                if self._is_inside_project(p):
                    files.append(p)

        # Root-level .md files
        for name in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md"):
            p = self.project_root / name
            if p.exists() and self._is_inside_project(p):
                files.append(p)

        # Sort for deterministic order
        files.sort()
        return files

    def discover_notebooks(self) -> list[Path]:
        """Discover all Jupyter notebooks to verify."""
        files: list[Path] = []
        if self.notebooks_dir.exists():
            for p in self.notebooks_dir.rglob("*.ipynb"):
                if self._is_inside_project(p):
                    files.append(p)
        files.sort()
        return files

    # ------------------------------------------------------------------
    # Code block extraction
    # ------------------------------------------------------------------
    def extract_code_blocks(self, file_path: Path) -> list[tuple[str, str, int, str | None]]:
        """Extract code blocks from a Markdown file.

        Returns:
            List of tuples: (language, code_content, start_line, skip_reason).
            skip_reason is None if the block should be executed.
        """
        blocks: list[tuple[str, str, int, str | None]] = []

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return blocks

        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        in_block = False
        current_lang = ""
        current_code: list[str] = []
        start_line = 0
        skip_reason: str | None = None

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            if stripped.startswith("```"):
                if not in_block:
                    in_block = True
                    lang_part = stripped[3:].strip()
                    current_lang = lang_part.split()[0] if lang_part else ""
                    current_code = []
                    start_line = i
                    # Check preceding line for skip-verify comment
                    skip_reason = None
                    if i >= 2:
                        prev = lines[i - 2].strip()
                        m = re.match(r"<!--\s*skip-verify:\s*(.+?)\s*-->", prev)
                        if m:
                            reason = m.group(1).strip()
                            if len(reason) < 10:
                                logger.warning(
                                    f"Skip-verify reason too short ({len(reason)} chars) at {file_path}:{i - 1}"
                                )
                            skip_reason = reason
                else:
                    in_block = False
                    if current_code:
                        code = "\n".join(current_code)
                        blocks.append((current_lang, code, start_line, skip_reason))
                    skip_reason = None
                continue

            if in_block:
                current_code.append(line)

        return blocks

    # ------------------------------------------------------------------
    # Timeout heuristic
    # ------------------------------------------------------------------
    def _detect_heavyweight(self, code: str) -> bool:
        """Use AST to detect if code contains heavyweight operations."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Direct function call: foo(...)
                if isinstance(node.func, ast.Name):
                    if node.func.id in HEAVYWEIGHT_NAMES:
                        return True
                # Method call: obj.method(...)
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in HEAVYWEIGHT_ATTRS:
                        return True
                    # module.function call
                    if isinstance(node.func.value, ast.Name):
                        mod_name = node.func.value.id
                        func_name = node.func.attr
                        if (mod_name, func_name) in HEAVYWEIGHT_MODULE_CALLS:
                            return True
        return False

    def _get_timeout(self, code: str) -> int:
        """Return appropriate timeout for a code block."""
        if self._detect_heavyweight(code):
            return self.timeout_heavy
        return self.timeout_base

    # ------------------------------------------------------------------
    # Python block execution (sequential within a file)
    # ------------------------------------------------------------------
    def _execute_python_blocks(
        self, file_path: Path, blocks: list[tuple[str, str, int, str | None]]
    ) -> list[BlockResult]:
        """Execute all Python blocks from a file sequentially in one subprocess."""
        results: list[BlockResult] = []

        python_blocks = [
            (lang, code, line, skip)
            for lang, code, line, skip in blocks
            if lang in ("python", "py")
        ]

        if not python_blocks:
            return results

        # Separate executable blocks from skipped ones
        executable_blocks: list[tuple[str, int, int]] = []  # (code, line, timeout)
        for lang, code, line, skip in python_blocks:
            if skip is not None:
                results.append(
                    BlockResult(
                        file=str(file_path.relative_to(self.project_root)),
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason=skip,
                    )
                )
                continue

            # Skip YAML configs disguised as Python
            stripped = code.strip()
            if stripped.startswith(YAML_PREFIXES):
                results.append(
                    BlockResult(
                        file=str(file_path.relative_to(self.project_root)),
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason="yaml configuration",
                    )
                )
                continue

            # Skip empty or comment-only blocks
            code_without_comments = "\n".join(
                line for line in code.split("\n")
                if line.strip() and not line.strip().startswith("#")
            )
            if not stripped or not code_without_comments.strip():
                results.append(
                    BlockResult(
                        file=str(file_path.relative_to(self.project_root)),
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason="empty code block",
                    )
                )
                continue

            timeout = self._get_timeout(code)
            executable_blocks.append((code, line, timeout))

        if not executable_blocks:
            return results

        # Build a single script that executes each block and reports results
        script_lines: list[str] = [
            "import sys, json, time",
            "_doc_verify_results = []",
            "",
            "# --- auto-injected mocks for doc verification ---",
            f'exec(open(r"{self.project_root / "scripts" / "doc_mocks.py"}").read())',
            "# --- end mocks ---",
        ]

        for idx, (code, line, timeout) in enumerate(executable_blocks):
            script_lines.append(f"# --- block {idx} from line {line} ---")
            script_lines.append("_t0 = time.time()")
            script_lines.append("_exc = None")
            script_lines.append("try:")
            # Indent code
            for c_line in code.split("\n"):
                script_lines.append(f"    {c_line}")
            script_lines.append("except Exception as _e:")
            script_lines.append("    _exc = str(_e)")
            script_lines.append("_elapsed = int((time.time() - _t0) * 1000)")
            script_lines.append(
                f"_doc_verify_results.append({{'line': {line}, 'exc': _exc, 'elapsed': _elapsed}})"
            )
            script_lines.append("")

        script_lines.append("print(json.dumps(_doc_verify_results))")
        script_lines.append("sys.stdout.flush()")

        script_content = "\n".join(script_lines)

        # Determine max timeout for the combined script
        max_timeout = max(timeout for _, _, timeout in executable_blocks)
        # Give some buffer for overhead
        max_timeout = max_timeout + 10

        rel_path = str(file_path.relative_to(self.project_root))
        if self.verbose:
            print(f"  Executing {len(executable_blocks)} Python block(s) from {rel_path}")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=str(self.project_root)
        ) as f:
            f.write(script_content)
            temp_path = f.name

        try:
            start = time.time()
            proc = subprocess.run(
                [sys.executable, temp_path],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=max_timeout,
            )
            overall_elapsed = int((time.time() - start) * 1000)

            output = proc.stdout.strip()
            # Try to parse the JSON results array from the last non-empty line
            block_results: list[dict] = []
            for line in reversed(output.split("\n")):
                line = line.strip()
                if line:
                    try:
                        block_results = json.loads(line)
                        if isinstance(block_results, list):
                            break
                    except json.JSONDecodeError:
                        continue

            if proc.returncode != 0:
                # Script itself crashed — try to use any parsed results, then mark rest
                for idx, (_, line, _) in enumerate(executable_blocks):
                    if idx < len(block_results):
                        br = block_results[idx]
                        status = "PASS" if br.get("exc") is None else "FAIL"
                        err = br.get("exc", "") or ""
                        elapsed = br.get("elapsed", 0)
                    else:
                        status = "FAIL"
                        err = f"Script crashed: {proc.stderr[:500]}"
                        elapsed = 0
                    results.append(
                        BlockResult(
                            file=rel_path,
                            line=line,
                            language="python",
                            status=status,
                            error=err[:500] if err else "",
                            execution_time_ms=elapsed,
                        )
                    )
                return results

            for idx, (_, line, _) in enumerate(executable_blocks):
                if idx < len(block_results):
                    br = block_results[idx]
                    status = "PASS" if br.get("exc") is None else "FAIL"
                    err = br.get("exc", "") or ""
                    elapsed = br.get("elapsed", 0)
                else:
                    status = "FAIL"
                    err = "No result returned from execution"
                    elapsed = 0
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language="python",
                        status=status,
                        error=err[:500] if err else "",
                        execution_time_ms=elapsed,
                    )
                )

        except subprocess.TimeoutExpired:
            for _, line, _ in executable_blocks:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language="python",
                        status="TIMEOUT",
                        error=f"Execution exceeded {max_timeout}s timeout",
                    )
                )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        return results

    # ------------------------------------------------------------------
    # Bash block validation
    # ------------------------------------------------------------------
    def _validate_bash_blocks(
        self, file_path: Path, blocks: list[tuple[str, str, int, str | None]]
    ) -> list[BlockResult]:
        """Validate bash blocks with bash -n syntax check."""
        results: list[BlockResult] = []
        rel_path = str(file_path.relative_to(self.project_root))

        for lang, code, line, skip in blocks:
            if lang != "bash":
                continue

            if skip is not None:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language="bash",
                        status="SKIP",
                        skip_reason=skip,
                    )
                )
                continue

            stripped = code.strip()
            if not stripped:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language="bash",
                        status="SKIP",
                        skip_reason="empty code block",
                    )
                )
                continue

            # Write to temp file and run bash -n
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False, dir=str(self.project_root)
            ) as f:
                f.write(code)
                temp_path = f.name

            try:
                start = time.time()
                proc = subprocess.run(
                    ["bash", "-n", temp_path],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                elapsed = int((time.time() - start) * 1000)

                if proc.returncode == 0:
                    results.append(
                        BlockResult(
                            file=rel_path,
                            line=line,
                            language="bash",
                            status="PASS",
                            execution_time_ms=elapsed,
                        )
                    )
                else:
                    err = proc.stderr.strip()[:500]
                    results.append(
                        BlockResult(
                            file=rel_path,
                            line=line,
                            language="bash",
                            status="FAIL",
                            error=err,
                            execution_time_ms=elapsed,
                        )
                    )
            except subprocess.TimeoutExpired:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language="bash",
                        status="TIMEOUT",
                        error="bash -n exceeded 10s timeout",
                    )
                )
            except FileNotFoundError:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language="bash",
                        status="SKIP",
                        skip_reason="bash not available",
                    )
                )
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        return results

    # ------------------------------------------------------------------
    # YAML block validation
    # ------------------------------------------------------------------
    def _validate_yaml_blocks(
        self, file_path: Path, blocks: list[tuple[str, str, int, str | None]]
    ) -> list[BlockResult]:
        """Validate YAML blocks with yaml.safe_load."""
        results: list[BlockResult] = []
        rel_path = str(file_path.relative_to(self.project_root))

        for lang, code, line, skip in blocks:
            if lang != "yaml":
                continue

            if skip is not None:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason=skip,
                    )
                )
                continue

            stripped = code.strip()
            if not stripped:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason="empty code block",
                    )
                )
                continue

            try:
                import yaml
                yaml.safe_load(code)
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="PASS",
                    )
                )
            except Exception as e:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="FAIL",
                        error=str(e)[:500],
                    )
                )

        return results

    # ------------------------------------------------------------------
    # JSON block validation
    # ------------------------------------------------------------------
    def _validate_json_blocks(
        self, file_path: Path, blocks: list[tuple[str, str, int, str | None]]
    ) -> list[BlockResult]:
        """Validate JSON blocks with json.loads."""
        results: list[BlockResult] = []
        rel_path = str(file_path.relative_to(self.project_root))

        for lang, code, line, skip in blocks:
            if lang != "json":
                continue

            if skip is not None:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason=skip,
                    )
                )
                continue

            stripped = code.strip()
            if not stripped:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason="empty code block",
                    )
                )
                continue

            try:
                # Support both single JSON object and JSONL (one object per line)
                # Strip JSON comments (lines starting with //) before parsing
                lines = [l for l in code.split("\n") if l.strip() and not l.strip().startswith("//")]
                if not lines:
                    raise ValueError("Empty JSON block after removing comments")
                for l in lines:
                    json.loads(l.strip())
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="PASS",
                    )
                )
            except Exception as e:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="FAIL",
                        error=str(e)[:500],
                    )
                )

        return results

    # ------------------------------------------------------------------
    # CSV block validation
    # ------------------------------------------------------------------
    def _validate_csv_blocks(
        self, file_path: Path, blocks: list[tuple[str, str, int, str | None]]
    ) -> list[BlockResult]:
        """Validate CSV blocks with csv.reader."""
        results: list[BlockResult] = []
        rel_path = str(file_path.relative_to(self.project_root))

        for lang, code, line, skip in blocks:
            if lang != "csv":
                continue

            if skip is not None:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason=skip,
                    )
                )
                continue

            stripped = code.strip()
            if not stripped:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason="empty code block",
                    )
                )
                continue

            try:
                list(csv.reader(StringIO(code)))
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="PASS",
                    )
                )
            except Exception as e:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="FAIL",
                        error=str(e)[:500],
                    )
                )

        return results

    # ------------------------------------------------------------------
    # FASTA block validation
    # ------------------------------------------------------------------
    def _validate_fasta_blocks(
        self, file_path: Path, blocks: list[tuple[str, str, int, str | None]]
    ) -> list[BlockResult]:
        """Validate FASTA format blocks."""
        results: list[BlockResult] = []
        rel_path = str(file_path.relative_to(self.project_root))

        for lang, code, line, skip in blocks:
            if lang != "fasta":
                continue

            if skip is not None:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason=skip,
                    )
                )
                continue

            stripped = code.strip()
            if not stripped:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason="empty code block",
                    )
                )
                continue

            try:
                lines = stripped.split("\n")
                in_sequence = False
                for l in lines:
                    l = l.strip()
                    if not l:
                        continue
                    if l.startswith(">"):
                        in_sequence = True
                        continue
                    if in_sequence:
                        valid_chars = set("ACGTNacgtn")
                        if not all(c in valid_chars for c in l):
                            raise ValueError(f"Invalid FASTA character in line: {l[:50]}")
                if not any(l.startswith(">") for l in lines if l.strip()):
                    raise ValueError("FASTA must have at least one header line starting with >")
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="PASS",
                    )
                )
            except Exception as e:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="FAIL",
                        error=str(e)[:500],
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Dockerfile block validation
    # ------------------------------------------------------------------
    def _validate_dockerfile_blocks(
        self, file_path: Path, blocks: list[tuple[str, str, int, str | None]]
    ) -> list[BlockResult]:
        """Validate Dockerfile blocks with basic syntax checks."""
        results: list[BlockResult] = []
        rel_path = str(file_path.relative_to(self.project_root))

        for lang, code, line, skip in blocks:
            if lang != "dockerfile":
                continue

            if skip is not None:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason=skip,
                    )
                )
                continue

            stripped = code.strip()
            if not stripped:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="SKIP",
                        skip_reason="empty code block",
                    )
                )
                continue

            try:
                lines = stripped.split("\n")
                has_from = False
                i = 0
                while i < len(lines):
                    l = lines[i].strip()
                    # Handle line continuations (backslash at end)
                    while l.endswith("\\") and i + 1 < len(lines):
                        i += 1
                        l = l[:-1] + " " + lines[i].strip()
                    if not l or l.startswith("#"):
                        i += 1
                        continue
                    upper = l.upper()
                    valid_directives = (
                        "FROM", "RUN", "CMD", "LABEL", "MAINTAINER", "EXPOSE",
                        "ENV", "ADD", "COPY", "ENTRYPOINT", "VOLUME", "USER",
                        "WORKDIR", "ARG", "ONBUILD", "STOPSIGNAL", "HEALTHCHECK",
                        "SHELL",
                    )
                    if not any(upper.startswith(d) for d in valid_directives):
                        raise ValueError(f"Invalid Dockerfile instruction: {l[:50]}")
                    if upper.startswith("FROM"):
                        has_from = True
                    i += 1
                if not has_from:
                    raise ValueError("Dockerfile must have a FROM instruction")
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="PASS",
                    )
                )
            except Exception as e:
                results.append(
                    BlockResult(
                        file=rel_path,
                        line=line,
                        language=lang,
                        status="FAIL",
                        error=str(e)[:500],
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Notebook execution
    # ------------------------------------------------------------------
    def _execute_notebook(self, nb_path: Path) -> NotebookResult:
        """Execute a single Jupyter notebook."""
        rel_path = str(nb_path.relative_to(self.project_root))
        result = NotebookResult(file=rel_path)

        if not HAS_NBFORMAT:
            result.overall_status = "SKIP"
            result.cell_results.append({
                "index": 0,
                "source": "",
                "status": "SKIP",
                "error": "nbconvert not installed",
            })
            return result

        try:
            nb = nbformat.read(str(nb_path), as_version=4)
        except Exception as e:
            result.overall_status = "FAIL"
            result.cell_results.append({
                "index": 0,
                "source": "",
                "status": "FAIL",
                "error": f"Failed to read notebook: {e}",
            })
            return result

        # Check for skip-verify tag at notebook level
        if nb.metadata.get("tags") and "skip-verify" in nb.metadata.tags:
            reason = nb.metadata.get("skip_reason", "notebook marked skip-verify")
            result.overall_status = "SKIP"
            result.cell_results.append({
                "index": 0,
                "source": "",
                "status": "SKIP",
                "error": reason,
            })
            return result

        # Filter out magic-only cells before execution
        code_cells = []
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            source = cell.source.strip()
            if not source:
                continue
            # Check cell-level skip tag
            tags = cell.metadata.get("tags", [])
            if "skip-verify" in tags:
                result.cell_results.append({
                    "index": idx,
                    "source": source[:200],
                    "status": "SKIP",
                    "error": cell.metadata.get("skip_reason", "cell marked skip-verify"),
                })
                continue
            # Skip cells with only IPython magic
            non_magic_lines = [
                line for line in source.split("\n")
                if line.strip() and not line.strip().startswith(("%", "!"))
            ]
            if not non_magic_lines:
                result.cell_results.append({
                    "index": idx,
                    "source": source[:200],
                    "status": "SKIP",
                    "error": "IPython magic only",
                })
                continue
            code_cells.append((idx, cell))

        if not code_cells:
            result.overall_status = "SKIP"
            return result

        # Execute via nbconvert CLI in subprocess (avoids ZMQ issues in nested contexts)
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ipynb", delete=False, dir=str(self.project_root)
        ) as f:
            nbformat.write(nb, f)
            temp_nb = f.name

        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "jupyter", "nbconvert",
                    "--to", "notebook",
                    "--execute",
                    "--ExecutePreprocessor.timeout=120",
                    "--ExecutePreprocessor.kernel_name=python3",
                    "--output", temp_nb + ".out.ipynb",
                    temp_nb,
                ],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=180,
            )
            if proc.returncode == 0:
                for idx, _ in code_cells:
                    result.cell_results.append({
                        "index": idx,
                        "source": nb.cells[idx].source[:200],
                        "status": "PASS",
                        "error": "",
                    })
                result.overall_status = "PASS"
            else:
                error_str = proc.stderr[:500] if proc.stderr else proc.stdout[:500]
                failed_idx = code_cells[0][0] if code_cells else None
                for idx, _ in code_cells:
                    if idx == failed_idx:
                        result.cell_results.append({
                            "index": idx,
                            "source": nb.cells[idx].source[:200],
                            "status": "FAIL",
                            "error": error_str,
                        })
                    elif idx < (failed_idx or 0):
                        result.cell_results.append({
                            "index": idx,
                            "source": nb.cells[idx].source[:200],
                            "status": "PASS",
                            "error": "",
                        })
                    else:
                        result.cell_results.append({
                            "index": idx,
                            "source": nb.cells[idx].source[:200],
                            "status": "FAIL",
                            "error": "Execution halted due to earlier cell failure",
                        })
                result.overall_status = "FAIL"
        except subprocess.TimeoutExpired:
            for idx, _ in code_cells:
                result.cell_results.append({
                    "index": idx,
                    "source": nb.cells[idx].source[:200],
                    "status": "TIMEOUT",
                    "error": "Notebook execution timed out after 180s",
                })
            result.overall_status = "TIMEOUT"
        finally:
            try:
                os.unlink(temp_nb)
                if os.path.exists(temp_nb + ".out.ipynb"):
                    os.unlink(temp_nb + ".out.ipynb")
            except OSError:
                pass

        return result

    # ------------------------------------------------------------------
    # Markdown verification
    # ------------------------------------------------------------------
    def verify_markdown(self) -> None:
        """Verify all discovered Markdown files."""
        md_files = self.discover_markdown_files()
        logger.info(f"Discovered {len(md_files)} Markdown files")
        if self.verbose:
            print(f"Discovered {len(md_files)} Markdown files")

        for md_file in md_files:
            rel_path = str(md_file.relative_to(self.project_root))
            if self.verbose:
                print(f"Checking {rel_path}...")

            blocks = self.extract_code_blocks(md_file)
            if not blocks:
                continue

            # Python blocks
            py_results = self._execute_python_blocks(md_file, blocks)
            self.block_results.extend(py_results)

            # Bash blocks
            bash_results = self._validate_bash_blocks(md_file, blocks)
            self.block_results.extend(bash_results)

            # Other language blocks — validate supported formats, SKIP the rest
            for lang, code, line, skip in blocks:
                if lang not in ("python", "py", "bash"):
                    # If it has an explicit skip marker, honour it
                    if skip is not None:
                        self.block_results.append(
                            BlockResult(
                                file=rel_path,
                                line=line,
                                language=lang,
                                status="SKIP",
                                skip_reason=skip,
                            )
                        )
                        continue

            yaml_results = self._validate_yaml_blocks(md_file, blocks)
            self.block_results.extend(yaml_results)

            json_results = self._validate_json_blocks(md_file, blocks)
            self.block_results.extend(json_results)

            csv_results = self._validate_csv_blocks(md_file, blocks)
            self.block_results.extend(csv_results)

            fasta_results = self._validate_fasta_blocks(md_file, blocks)
            self.block_results.extend(fasta_results)

            dockerfile_results = self._validate_dockerfile_blocks(md_file, blocks)
            self.block_results.extend(dockerfile_results)

            # Remaining unsupported languages
            validated_langs = {"python", "py", "bash", "yaml", "json", "csv", "fasta", "dockerfile"}
            for lang, code, line, skip in blocks:
                if lang not in validated_langs:
                    reason = skip or f"unsupported language: {lang}"
                    self.block_results.append(
                        BlockResult(
                            file=rel_path,
                            line=line,
                            language=lang,
                            status="SKIP",
                            skip_reason=reason,
                        )
                    )

    # ------------------------------------------------------------------
    # Notebook verification
    # ------------------------------------------------------------------
    def verify_notebooks(self) -> None:
        """Verify all discovered notebooks."""
        nb_files = self.discover_notebooks()
        logger.info(f"Discovered {len(nb_files)} notebooks")
        if self.verbose:
            print(f"Discovered {len(nb_files)} notebooks")

        for nb_path in nb_files:
            rel_path = str(nb_path.relative_to(self.project_root))
            if self.verbose:
                print(f"Checking notebook {rel_path}...")
            result = self._execute_notebook(nb_path)
            self.notebook_results.append(result)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    def generate_markdown_report(self, output_path: Path) -> None:
        """Generate human-readable Markdown report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = len(self.block_results)
        pass_count = sum(1 for r in self.block_results if r.status == "PASS")
        fail_count = sum(1 for r in self.block_results if r.status == "FAIL")
        skip_count = sum(1 for r in self.block_results if r.status == "SKIP")
        timeout_count = sum(1 for r in self.block_results if r.status == "TIMEOUT")

        lines = [
            "# DNALLM Documentation Verification Report",
            "",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}Z",
            f"**Total code blocks:** {total}",
            f"**Files scanned:** {len(self.discover_markdown_files())} Markdown + {len(self.discover_notebooks())} notebooks",
            "",
            "## Summary",
            "",
            f"| Status  | Count |",
            f"|---------|-------|",
            f"| PASS    | {pass_count} |",
            f"| FAIL    | {fail_count} |",
            f"| SKIP    | {skip_count} |",
            f"| TIMEOUT | {timeout_count} |",
            "",
        ]

        # Per-file details
        lines.extend(["## Details", ""])
        current_file: str | None = None
        for r in self.block_results:
            if r.file != current_file:
                current_file = r.file
                lines.extend([f"### {current_file}", ""])
                lines.append("| Line | Lang | Status | Error |")
                lines.append("|------|------|--------|-------|")
            err = r.error.replace("|", "\\|") if r.error else ""
            reason = r.skip_reason.replace("|", "\\|") if r.skip_reason else ""
            display_err = err or reason
            lines.append(f"| {r.line} | {r.language} | {r.status} | {display_err} |")
        lines.append("")

        # Notebook details
        if self.notebook_results:
            lines.extend(["## Notebooks", ""])
            for nb in self.notebook_results:
                lines.extend([f"### {nb.file}", ""])
                lines.append("| Cell | Status | Error |")
                lines.append("|------|--------|-------|")
                for cell in nb.cell_results:
                    err = cell.get("error", "").replace("|", "\\|")
                    lines.append(f"| {cell['index']} | {cell['status']} | {err} |")
                lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Markdown report written to {output_path}")
        if self.verbose:
            print(f"Markdown report written to {output_path}")

    def generate_json_report(self, output_path: Path) -> None:
        """Generate machine-readable JSON report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = len(self.block_results)
        pass_count = sum(1 for r in self.block_results if r.status == "PASS")
        fail_count = sum(1 for r in self.block_results if r.status == "FAIL")
        skip_count = sum(1 for r in self.block_results if r.status == "SKIP")
        timeout_count = sum(1 for r in self.block_results if r.status == "TIMEOUT")

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_blocks": total,
                "pass": pass_count,
                "fail": fail_count,
                "skip": skip_count,
                "timeout": timeout_count,
            },
            "results": [
                {
                    "file": r.file,
                    "line": r.line,
                    "language": r.language,
                    "status": r.status,
                    "error": r.error,
                    "execution_time_ms": r.execution_time_ms,
                    "skip_reason": r.skip_reason,
                }
                for r in self.block_results
            ],
            "notebooks": [
                {
                    "file": nb.file,
                    "overall_status": nb.overall_status,
                    "cell_results": nb.cell_results,
                }
                for nb in self.notebook_results
            ],
        }

        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"JSON report written to {output_path}")
        if self.verbose:
            print(f"JSON report written to {output_path}")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        output_md: Path | None = None,
        output_json: Path | None = None,
    ) -> int:
        """Run full verification pipeline.

        Returns:
            0 if no FAIL or TIMEOUT blocks, 1 otherwise.
        """
        self.verify_markdown()
        self.verify_notebooks()

        if output_md:
            self.generate_markdown_report(output_md)
        if output_json:
            self.generate_json_report(output_json)

        fail_count = sum(1 for r in self.block_results if r.status == "FAIL")
        timeout_count = sum(1 for r in self.block_results if r.status == "TIMEOUT")
        nb_fail = sum(1 for nb in self.notebook_results if nb.overall_status == "FAIL")
        nb_timeout = sum(1 for nb in self.notebook_results if nb.overall_status == "TIMEOUT")

        total_issues = fail_count + timeout_count + nb_fail + nb_timeout
        if total_issues > 0:
            logger.warning(f"Verification found {total_issues} issues")
            if self.verbose:
                print(f"Verification found {total_issues} issues")
            return 1

        logger.info("Verification passed — no failures or timeouts")
        if self.verbose:
            print("Verification passed — no failures or timeouts")
        return 0


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_docs.py",
        description="Verify documentation code examples by executing them.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Override docs directory (default: docs/)",
    )
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=None,
        help="Override notebooks directory (default: example/notebooks/)",
    )
    parser.add_argument(
        "--timeout-base",
        type=int,
        default=30,
        help="Base timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--timeout-heavy",
        type=int,
        default=120,
        help="Heavyweight timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Markdown report path (default: docs/VERIFICATION_REPORT.md)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="JSON report path (default: .planning/phases/06-fix-documentation-code-examples-and-verify-all-docs-are-exec/verification_results.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent

    docs_dir = args.docs_dir
    notebooks_dir = args.notebooks_dir

    default_md = project_root / "docs" / "VERIFICATION_REPORT.md"
    default_json = (
        project_root
        / ".planning"
        / "phases"
        / "06-fix-documentation-code-examples-and-verify-all-docs-are-exec"
        / "verification_results.json"
    )

    output_md = args.output_md or default_md
    output_json = args.output_json or default_json

    verifier = DocVerifier(
        project_root=project_root,
        docs_dir=docs_dir,
        notebooks_dir=notebooks_dir,
        timeout_base=args.timeout_base,
        timeout_heavy=args.timeout_heavy,
        verbose=args.verbose,
    )

    return verifier.run(output_md=output_md, output_json=output_json)


if __name__ == "__main__":
    sys.exit(main())
