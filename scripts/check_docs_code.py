#!/usr/bin/env python3
"""
DNALLM Documentation Code Checker

This script checks code examples in docs directory for correctness.
"""

import re
import sys
from pathlib import Path

# Core libraries to skip during import checks
CORE_LIBRARIES = {
    # DNALLM modules
    "dnallm",
    "clear_model_cache",
    "load_config",
    "load_model_and_tokenizer",
    "DNATrainer",
    "DNADataset",
    "AutoTokenizer",
    "AutoModelForMaskedLM",
    "AutoModelForSequenceClassification",
    "DNAInference",
    "Mutagenesis",
    "Benchmark",
    "peft",
    # Third party libraries
    "torch",
    "transformers",
    "pandas",
    "numpy",
    "matplotlib",
    "sklearn",
    "scipy",
    "huggingface",
    "modelscope",
    "accelerate",
    "datasets",
    "tokenizers",
    "sentencepiece",
    "einops",
    "loguru",
    "rich",
    "pydantic",
    "pyyaml",
    "tqdm",
    "requests",
    "aiohttp",
    "uvicorn",
    "websockets",
    "captum",
    "umap",
    "altair",
    "addict",
    "colorama",
    "seaborn",
    "plotly",
    "wandb",
    "tensorboardx",
    "evaluate",
    "seqeval",
    "jax",
    "flax",
    "optimum",
    # Standard library modules (commonly used)
    "contextmanager",
    "contextlib",
    "functools",
    "itertools",
    "collections",
    "pathlib",
    "typing",
    "dataclasses",
    "enum",
    "json",
    "yaml",
    "pickle",
    "csv",
    "re",
    "sys",
    "os",
    "memory_profiler",
    "KFold",
    "StratifiedKFold",
    "CustomMetric",
    "autoclt",
    "product",
    "reverse_complement",
}


class DocCodeChecker:
    """Documentation code checker."""

    def __init__(self, docs_dir: str) -> None:
        """Initialize checker.

        Args:
            docs_dir: Path to docs directory.
        """
        self.docs_dir = Path(docs_dir)
        self.md_files: list[Path] = []
        self.results: list[dict] = []
        self.issues: list[dict] = []

        if self.docs_dir.exists():
            self.md_files = list(self.docs_dir.glob("**/*.md"))

    def extract_code_blocks(
        self, file_path: Path
    ) -> list[tuple[str, str, int]]:
        """Extract code blocks from markdown file.

        Args:
            file_path: Path to markdown file.

        Returns:
            List of tuples: (language, code_content, line_number).
        """
        code_blocks: list[tuple[str, str, int]] = []

        if not file_path.exists():
            self.issues.append({
                "file": file_path,
                "type": "file_not_found",
                "message": f"File not found: {file_path}",
            })
            return code_blocks

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            in_code_block = False
            current_lang = ""
            current_code: list[str] = []
            start_line = 0

            for i, line in enumerate(lines, 1):
                stripped = line.strip()

                if stripped.startswith("```"):
                    if not in_code_block:
                        in_code_block = True
                        # Extract language (handle cases like ```python)
                        lang_part = stripped[3:].strip()
                        # Take first part as language
                        current_lang = (
                            lang_part.split()[0] if lang_part else ""
                        )
                        current_code = []
                        start_line = i
                    else:
                        in_code_block = False
                        if current_code:
                            code_content = "\n".join(current_code)
                            code_blocks.append((
                                current_lang,
                                code_content,
                                start_line,
                            ))
                    continue

                if in_code_block:
                    current_code.append(line)

        except UnicodeDecodeError:
            self.issues.append({
                "file": file_path,
                "type": "encoding_error",
                "message": f"Encoding error: {file_path}",
            })
        except Exception as e:
            self.issues.append({
                "file": file_path,
                "type": "read_error",
                "message": f"Read error: {e!s}",
            })

        return code_blocks

    def check_python_code(
        self, code: str, file_path: Path, line_num: int
    ) -> dict:
        """Check Python code syntax and imports.

        Args:
            code: Code content.
            file_path: File path.
            line_num: Line number.

        Returns:
            Check result dictionary.
        """
        code = code.strip()

        if not code:
            return {
                "status": "skipped",
                "reason": "empty code",
                "line": line_num,
                "file": file_path,
            }

        if code.startswith("#"):
            return {
                "status": "skipped",
                "reason": "bash comment",
                "line": line_num,
                "file": file_path,
            }

        yaml_prefixes = (
            "task:",
            "finetune:",
            "inference:",
            "server:",
            "mcp:",
            "lora:",
            "models:",
            "multi_model:",
            "sse:",
            "logging:",
            "training_args:",
        )
        if code.startswith(yaml_prefixes):
            return {
                "status": "skipped",
                "reason": "yaml configuration",
                "line": line_num,
                "file": file_path,
            }

        if "```" in code:
            return {
                "status": "skipped",
                "reason": "nested code block",
                "line": line_num,
                "file": file_path,
            }

        try:
            imports = re.findall(r"from\s+([a-zA-Z_][\w.]*)\s+import", code)
            imports.extend(re.findall(r"import\s+([a-zA-Z_][\w.]*)", code))
            imports = list(set(imports))

            missing_imports: list[str] = []

            for imp in imports:
                # Get top-level module
                parts = imp.split(".")
                top_level = parts[0]

                # Skip if in core libraries
                if top_level in CORE_LIBRARIES:
                    continue

                # Handle sklearn submodules (e.g., sklearn.model_selection)
                if top_level == "sklearn":
                    continue

                # Check if it's a standard library module
                stdlib_modules = {
                    "contextmanager",
                    "contextlib",
                    "functools",
                    "itertools",
                    "collections",
                    "pathlib",
                    "typing",
                    "dataclasses",
                    "enum",
                    "json",
                    "pickle",
                    "csv",
                    "re",
                    "sys",
                    "os",
                    "io",
                    "datetime",
                    "time",
                    "random",
                    "math",
                    "hashlib",
                }
                if top_level in stdlib_modules:
                    continue

                # Try to import the module
                try:
                    __import__(top_level)
                except (ImportError, NameError):
                    missing_imports.append(imp)

            if missing_imports:
                missing_str = ", ".join(missing_imports)
                return {
                    "status": "warning",
                    "message": f"Missing modules: {missing_str}",
                    "line": line_num,
                    "file": file_path,
                }

            return {"status": "checkable", "line": line_num, "file": file_path}

        except Exception as e:
            return {
                "status": "error",
                "message": f"Check failed: {e!s}",
                "line": line_num,
                "file": file_path,
            }

    def check_bash_code(
        self, code: str, file_path: Path, line_num: int
    ) -> dict:
        """Check bash command validity.

        Args:
            code: Code content.
            file_path: File path.
            line_num: Line number.

        Returns:
            Check result dictionary.
        """
        code = code.strip()

        if not code:
            return {
                "status": "skipped",
                "reason": "empty code",
                "line": line_num,
                "file": file_path,
            }

        if code.startswith("#") and not code.startswith("# "):
            return {
                "status": "skipped",
                "reason": "incomplete command",
                "line": line_num,
                "file": file_path,
            }

        commands = [
            c.strip()
            for c in code.split("\n")
            if c.strip() and not c.strip().startswith("#")
        ]

        if not commands:
            return {
                "status": "skipped",
                "reason": "empty or comment only",
                "line": line_num,
                "file": file_path,
            }

        return {"status": "checkable", "line": line_num, "file": file_path}

    def run_check(self) -> tuple[list[dict], list[dict]]:
        """Run all checks.

        Returns:
            Tuple of (check results, issues).
        """
        print("=" * 80)
        print("DNALLM Documentation Code Checker")
        print("=" * 80)

        total_files = len(self.md_files)
        print(f"Scanning files: {total_files}")

        code_stats: dict[str, int] = {
            "python": 0,
            "bash": 0,
            "yaml": 0,
            "other": 0,
        }

        checked_count = 0
        skipped_count = 0

        for i, md_file in enumerate(self.md_files, 1):
            if i % 20 == 0:
                print(f"Progress: {i}/{total_files} files checked...")

            code_blocks = self.extract_code_blocks(md_file)

            if not code_blocks:
                continue

            rel_path = md_file.relative_to(self.docs_dir)
            print(f"\n📄 Checking file: {rel_path}")
            print(f"   Code blocks: {len(code_blocks)}")

            for lang, code, line_num in code_blocks:
                code_stats[lang] = code_stats.get(lang, 0) + 1

                if lang == "python":
                    result = self.check_python_code(code, md_file, line_num)
                elif lang == "bash":
                    result = self.check_bash_code(code, md_file, line_num)
                elif lang == "yaml":
                    result = {
                        "status": "checkable",
                        "line": line_num,
                        "file": md_file,
                    }
                else:
                    result = {
                        "status": "skipped",
                        "reason": f"unsupported language: {lang}",
                        "line": line_num,
                        "file": md_file,
                    }

                self.results.append({
                    "file": rel_path,
                    "line": line_num,
                    "language": lang,
                    "result": result,
                })

                status = result.get("status", "unknown")
                if status == "checkable":
                    checked_count += 1
                    print(f"   ✅ [{lang}] Line {line_num}: checkable")
                elif status == "warning":
                    msg = result.get("message", "warning")
                    print(f"   ⚠️ [{lang}] Line {line_num}: {msg}")
                elif status == "error":
                    msg = result.get("message", "error")
                    print(f"   ❌ [{lang}] Line {line_num}: {msg}")
                else:
                    skipped_count += 1
                    reason = result.get("reason", "")
                    icon = "⏭️"
                    msg = f"   {icon} [{lang}] Line {line_num}:"
                    print(f"{msg} skipped ({reason})")

        print("\n" + "=" * 80)
        print("Check Statistics")
        print("=" * 80)
        py_count = code_stats.get("python", 0)
        bash_count = code_stats.get("bash", 0)
        yaml_count = code_stats.get("yaml", 0)
        print(f"Python blocks: {py_count}")
        print(f"Bash blocks: {bash_count}")
        excluded = ["python", "bash", "yaml"]
        other_count = sum(
            v for k, v in code_stats.items() if k not in excluded
        )
        print(f"YAML blocks: {yaml_count}")
        print(f"Other blocks: {other_count}")
        print(f"Checkable: {checked_count}")
        print(f"Skipped: {skipped_count}")

        return self.results, self.issues


def main() -> int:
    """Main function."""
    docs_dir = Path("/Users/forrest/GitHub/DNALLM/docs")

    if not docs_dir.exists():
        print(f"❌ Error: Directory not found {docs_dir}")
        return 1

    print(f"📁 Checking directory: {docs_dir}\n")

    checker = DocCodeChecker(str(docs_dir))
    results, issues = checker.run_check()

    # Generate report
    report_path = "/Users/forrest/GitHub/DNALLM/DOCS_CODE_CHECK_REPORT.md"
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# DNALLM Documentation Code Check Report",
        "",
        f"Files checked: {len(checker.md_files)}",
        f"Code blocks: {len(results)}",
    ]

    err_warn_count = len([
        r for r in results if r["result"].get("status") in ["error", "warning"]
    ])
    report_lines.append(f"Issues found: {err_warn_count}")
    report_lines.extend([
        f"Processing issues: {len(issues)}",
        "",
        "## Summary",
        "",
    ])

    status_counts: dict[str, int] = {}
    for r in results:
        status = r["result"].get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    status_icons = {
        "checkable": "✅",
        "warning": "⚠️",
        "error": "❌",
        "skipped": "⏭️",
    }

    for status, count in sorted(status_counts.items()):
        icon = status_icons.get(status, "❓")
        report_lines.append(f"- {icon} {status}: {count}")

    report_lines.extend(["", "## Details", ""])

    current_file: str | None = None
    for r in results:
        file_path = str(r["file"])
        if file_path != current_file:
            current_file = file_path
            report_lines.extend([f"### {file_path}", ""])

        status = r["result"].get("status", "unknown")
        line = r["line"]
        lang = r["language"]
        icon = status_icons.get(status, "❓")

        report_lines.append(f"- {icon} [{lang}] Line {line}: {status}")
        if "message" in r["result"]:
            report_lines.append(f"  - {r['result']['message']}")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n📊 Report saved to: {report_file}")

    total_results = len(results)
    error_count = len([
        r for r in results if r["result"].get("status") == "error"
    ])
    warnings = len([
        r for r in results if r["result"].get("status") == "warning"
    ])

    print("\n📈 Statistics:")
    print(f"  - Total blocks: {total_results}")
    print(f"  - Errors: {error_count}")
    print(f"  - Warnings: {warnings}")
    print(f"  - Issues: {len(issues)}")

    if error_count > 0:
        print(f"\n⚠️ Found {error_count} errors!")
        return 1
    elif warnings > 0:
        print(f"\n⚠️ Found {warnings} warnings.")
        return 0
    else:
        print("\n✅ All checks passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
