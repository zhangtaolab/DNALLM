#!/usr/bin/env python3
"""Run GB10 feasibility spikes for the environment-gated example model families (FEAS-01)."""

import argparse
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# The five families the Phase 5 verdict matrix (05-FEASIBILITY.md) covers.
FAMILIES = ("evo1", "evo2", "megadna", "pybigwig", "marimo")

# Exact notebook variants (D-05): verdicts must be taken against THESE ids first.
EVO1_NOTEBOOK_VARIANT = "togethercomputer/evo-1-131k-base"
EVO1_FALLBACK_VARIANT = "togethercomputer/evo-1-8k-base"
EVO2_NOTEBOOK_VARIANT = "arcinstitute/evo2_1b_base"
MEGADNA_NOTEBOOK_VARIANT = "lingxusb/megaDNA_updated"

# D-06 fallback for megaDNA: the unpickle may need the cloned repo's classes.
# Pinned + hash-verified clone only — never the notebook's unpinned `git clone`
# (threat T-05-07).
MEGADNA_REPO_URL = "https://github.com/lingxusb/megaDNA"
MEGADNA_COMMIT_PIN = "cb2f5ab4cc88dc0effe05c5f23358862c837014a"

# Notebook-faithful configs and the marimo app under spike.
EVO_CONFIG = "example/notebooks/generation_evo_models/inference_evo_config.yaml"
MEGADNA_CONFIG = "example/notebooks/generation_megaDNA/inference_megaDNA_config.yaml"
MARIMO_APP_DIR = "example/marimo/inference"
MARIMO_APP = "inference_demo.py"
MARIMO_FLAVOR_TIMEOUT_S = 1200

REPO_ROOT = Path(__file__).resolve().parents[2]


def _one_line(text: str) -> str:
    """Collapse a traceback/message onto one greppable ``failure_text=`` line."""
    return " ".join(text.split())


def _failure_text(exc: BaseException) -> str:
    """Return the exact failure text (type + message + last traceback frames)."""
    import traceback

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return _one_line(tb[-1500:])


def _emit_evidence(
    family: str,
    variant: str,
    *,
    load_s: float,
    forward_s: float,
    peak_vram_gb: float,
    disk_gb: float,
    result: str,
    failure_text: str | None = None,
    extra: tuple[str, ...] = (),
) -> None:
    """Print the D-05 machine-greppable evidence block for one family run.

    Args:
        family: Family key (one of ``FAMILIES``).
        variant: Exact model variant / spike target this run used.
        load_s: Wall seconds spent reaching a loaded model (or failing).
        forward_s: Wall seconds of the real forward/generate (0.0 when unused).
        peak_vram_gb: ``torch.cuda.max_memory_allocated()`` in GiB (0.0 unused).
        disk_gb: ``du`` footprint of the model snapshot in GB (0.0 unused).
        result: ``OK`` or ``FAIL``.
        failure_text: Exact failure text on FAIL.
        extra: Additional family-specific ``key=value`` evidence lines.
    """
    print(f"family={family}")
    print(f"variant={variant}")
    for line in extra:
        print(line)
    print(f"load_s={load_s:.1f}")
    print(f"forward_s={forward_s:.1f}")
    print(f"peak_vram_gb={peak_vram_gb:.2f}")
    print(f"disk_gb={disk_gb:.2f}")
    print(f"result={result}")
    if failure_text:
        print(f"failure_text={failure_text}")
    print()


def _gpu_guard() -> bool:
    """Print the device identity and fail closed when CUDA is unavailable.

    Returns:
        True when the box carries a usable CUDA device, False otherwise
        (``main`` turns False into exit code 2 — distinguishable from a
        pre-evidence crash, which is exit code 1).
    """
    import torch

    if not torch.cuda.is_available():
        print("gpu_guard=CUDA_UNAVAILABLE")
        print(
            "failure_text=torch.cuda.is_available() is False; this spike requires "
            "the GB10 GPU (same hardware class as the dnallm-nightly runner, D-04)"
        )
        return False
    print(f"gpu_device={torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"gpu_compute_capability={capability[0]}.{capability[1]}")
    print(f"torch_version={torch.__version__}")
    return True


def _peak_vram_gb() -> float:
    """Return ``torch.cuda.max_memory_allocated()`` in GiB (0.0 without CUDA)."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.max_memory_allocated() / 1e9
    except Exception:  # pragma: no cover - defensive, CUDA gone mid-run
        return 0.0


def _snapshot_disk_gb(repo_id: str, revision: str | None = None) -> tuple[float, str]:
    """Measure the HF snapshot footprint in bytes (D-05 disk evidence).

    Sums ``st_size`` (dereferenced, deduped by inode) over the
    ``models--<repo>`` cache root: snapshot entries are symlinks into
    ``blobs/``, so lstat would measure link lengths and naive sums would
    double-count blob + link.

    Args:
        repo_id: HuggingFace repo id that was downloaded.
        revision: Revision the dnallm route fetched (None = default branch).

    Returns:
        Tuple of (numeric GB, human string).
    """
    path = ""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True)
    except Exception:
        hub_root = (
            Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
        )
        matches = sorted(hub_root.glob("models--" + repo_id.replace("/", "--") + "*"))
        if matches:
            path = str(matches[-1])
    if not path or not os.path.exists(path):
        return 0.0, "not-found"
    # HF cache layout: snapshots/<rev> entries are symlinks into blobs/ — walk
    # up to the models--<repo> root (fallback: keep the given path) and sum it.
    target = Path(path)
    root = target
    while root.name and not root.name.startswith("models--"):
        root = root.parent
    if root.name.startswith("models--"):
        target = root
    total = 0
    seen_inodes: set[tuple[int, int]] = set()
    for dirpath, _dirnames, filenames in os.walk(target):
        for name in filenames:
            try:
                stat_result = os.stat(os.path.join(dirpath, name))  # follow HF links
            except OSError:
                continue
            inode = (stat_result.st_dev, stat_result.st_ino)
            if inode in seen_inodes:
                continue
            seen_inodes.add(inode)
            total += stat_result.st_size
    gb = total / 1e9
    human = f"{gb:.2f}GB" if gb >= 1 else f"{total / 1e6:.1f}MB"
    return gb, human


def _forward_pass(model: Any, tokenizer: Any, prompt: str = "ACGT" * 64) -> tuple[float, str]:
    """Run one real forward pass through the loaded model and time it.

    Args:
        model: Model (or dnallm wrapper) returned by ``load_model_and_tokenizer``.
        tokenizer: Its tokenizer.
        prompt: DNA prompt; 256 nt by default.

    Returns:
        Tuple of (wall seconds, logits shape string).
    """
    import torch

    encoded = tokenizer([prompt], return_tensors="pt")
    # transformers BatchEncoding is a UserDict, NOT a dict instance — probe for
    # the key instead of isinstance-checking, then coerce lists to a tensor.
    ids = encoded
    if hasattr(encoded, "__getitem__"):
        try:
            ids = encoded["input_ids"]
        except (KeyError, IndexError):
            ids = encoded
    ids = torch.as_tensor(ids)
    core = getattr(model, "model", model)  # CustomEvo wrappers keep the core on .model
    device = next(core.parameters()).device
    ids = ids.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    # no_grad: evo2 loads weights under inference_mode, and vortex's compute_filter
    # saves activations for backward — a plain tracked forward then raises
    # "Inference tensors cannot be saved for backward" (inference is no-grad).
    with torch.no_grad():
        out = core(ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_s = time.perf_counter() - start
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return forward_s, str(tuple(logits.shape))


def _load_notebook_config(config_rel: str) -> dict:
    """Load the notebook's exact YAML config (D-05 fidelity) with a tmp output_dir.

    Args:
        config_rel: Repo-relative path to the notebook's inference config.

    Returns:
        The validated config dict (``task`` / ``inference`` sections).
    """
    from dnallm.configuration.configs import load_config

    configs = load_config(str(REPO_ROOT / config_rel))
    # Keep every write under /tmp — the repo tree stays clean during the spike.
    configs["inference"].output_dir = tempfile.mkdtemp(prefix="spike-results-")
    return configs


def _yaml_flag(path: Path, key: str) -> bool:
    """Read a boolean flag from a dnallm-packaged evo config YAML."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return bool(data.get(key, False))


def _checkout_pinned_megadna(clone_dir: Path) -> str:
    """Clone megaDNA at the pinned commit and verify the hash (T-05-07).

    Args:
        clone_dir: Destination for the pinned clone.

    Returns:
        The verified commit hash.
    """
    if not clone_dir.exists():
        # ruff: ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
        subprocess.run(
            ["git", "clone", "--quiet", MEGADNA_REPO_URL, str(clone_dir)],
            check=True,
            capture_output=True,
        )
    # ruff: ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
    subprocess.run(
        ["git", "-C", str(clone_dir), "checkout", "--quiet", MEGADNA_COMMIT_PIN],
        check=True,
        capture_output=True,
    )
    # ruff: ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
    got = subprocess.run(
        ["git", "-C", str(clone_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if got != MEGADNA_COMMIT_PIN:
        raise RuntimeError(f"pinned clone hash mismatch: {got} != {MEGADNA_COMMIT_PIN}")
    return got


def _numpy_fromstring_shim() -> None:
    """Restore ``np.fromstring`` binary mode for stripedhyena on numpy >= 2.

    stripedhyena 0.2.2's CharLevelTokenizer calls ``np.fromstring(text,
    dtype=np.uint8)``, whose binary mode numpy 2.x removed ("use frombuffer
    instead"). Shimming it here mirrors the repo's ``transformers_compat``
    monkeypatch pattern and is the Phase 8 recipe for the same break.
    """
    import numpy as np

    if int(np.__version__.split(".")[0]) < 2:
        return

    def _fromstring(text: Any, dtype: Any = np.uint8, **_kwargs: Any):
        data = text.encode("utf-8") if isinstance(text, str) else text
        return np.frombuffer(data, dtype=dtype)

    np.fromstring = _fromstring  # type: ignore[attr-defined]


def spike_evo1(variant: str) -> int:
    """Spike the EVO-1 family through the dnallm route.

    Args:
        variant: Exact HF variant (notebook first, ``evo-1-8k-base`` fallback).

    Returns:
        0 on OK, 1 on FAIL (evidence printed either way).
    """
    family = "evo1"
    start = time.perf_counter()
    # dnallm handler revision pin (evo.py:371): "1.1_fix" only for dotted names.
    revision = "1.1_fix" if "." in variant else "main"
    print("prerequisite=evo-model + stripedhyena installed in the throwaway venv only")
    print(f"hf_revision={revision} (auto-selected by the dnallm handler)")
    try:
        _numpy_fromstring_shim()
        from dnallm.models.special.evo import evo_models
        from dnallm.utils.support import is_flash_attention_capable

        stem = next(value for key, value in evo_models.items() if key in variant.lower())
        flash = is_flash_attention_capable()
        config_name = f"{stem}{'' if flash else '-noFA'}.yml"
        print(f"flash_attention_installed={flash}")
        print(f"selected_config={config_name}")

        configs = _load_notebook_config(EVO_CONFIG)
        from dnallm.models import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(variant, configs["task"], source="huggingface")
        load_s = time.perf_counter() - start

        # Notebook path: DNAInference moves the model onto the auto device.
        from dnallm.inference import DNAInference

        engine = DNAInference(model=model, tokenizer=tokenizer, config=configs)
        forward_s, logits_shape = _forward_pass(engine.model, tokenizer)
        generate_s, generate_note = _notebook_generate(engine)
        peak = _peak_vram_gb()
        disk_gb, disk_human = _snapshot_disk_gb(variant, revision=revision)
        _emit_evidence(
            family,
            variant,
            load_s=load_s,
            forward_s=forward_s,
            peak_vram_gb=peak,
            disk_gb=disk_gb,
            result="OK",
            extra=(
                f"logits_shape={logits_shape}",
                f"hf_revision={revision}",
                f"selected_config={config_name}",
                f"generate_s={generate_s:.1f}",
                f"generate_note={generate_note}",
                f"disk_human={disk_human}",
            ),
        )
        return 0
    except Exception as exc:
        _emit_evidence(
            family,
            variant,
            load_s=time.perf_counter() - start,
            forward_s=0.0,
            peak_vram_gb=_peak_vram_gb(),
            disk_gb=0.0,
            result="FAIL",
            failure_text=_failure_text(exc),
            extra=(f"hf_revision={revision}",),
        )
        return 1


def _notebook_generate(engine: Any) -> tuple[float, str]:
    """Run the notebook's exact generate call and describe the outcome.

    A generate failure is recorded as a note, not a family FAIL: the raw
    forward above is the D-05 feasibility evidence; a generate-only failure
    is a repair finding for Phase 8, not hardware infeasibility.

    Args:
        engine: Constructed ``DNAInference`` engine.

    Returns:
        Tuple of (wall seconds, outcome note).
    """
    start = time.perf_counter()
    try:
        output = engine.generate(["@", "ACGT"])
        wall = time.perf_counter() - start
        summary = str(output)[:200]
        return wall, f"OK output[:200]={summary}"
    except Exception as exc:
        wall = time.perf_counter() - start
        return wall, f"GENERATE_FAILED {_failure_text(exc)[:600]}"


def spike_evo2(variant: str, config_override: str | None = None) -> int:
    """Spike the EVO2 family through the dnallm route.

    Args:
        variant: Exact HF variant (notebook: ``arcinstitute/evo2_1b_base``).
        config_override: ``"noFA-noFP8"`` forces the shipped no-FA/no-FP8
            config (the D-06 fallback against the GB10 FP8 trap); None uses
            the handler's auto-selection.

    Returns:
        0 on OK, 1 on FAIL (evidence printed either way).
    """
    family = "evo2"
    start = time.perf_counter()
    print("prerequisite=evo2 package installed in the throwaway venv only (pulls vortex)")
    print(
        "gb10_trap=is_fp8_capable() is True on GB10 (CC >= 9.0), so the auto-selected "
        "config is the FP8 variant; --fallback forces the noFA-noFP8 config"
    )
    patched = False
    original_fp8: Any = None
    try:
        from dnallm.models.special import evo as evo_module
        from dnallm.models.special.evo import evo2_models
        from dnallm.utils.support import is_flash_attention_capable

        stem = next(value for key, value in evo2_models.items() if key in variant.lower())
        flash = is_flash_attention_capable()
        fp8 = evo_module.is_fp8_capable()
        if config_override == "noFA-noFP8":
            original_fp8 = evo_module.is_fp8_capable

            def _fp8_disabled() -> bool:
                return False

            evo_module.is_fp8_capable = _fp8_disabled
            patched = True
            fp8 = False
            print(
                "config_override=noFA-noFP8 (is_fp8_capable patched to False inside the "
                "dnallm handler for this load — smallest documented deviation, D-06)"
            )
        config_name = f"{stem}{'' if flash else '-noFA'}{'' if fp8 else '-noFP8'}.yml"
        config_path = REPO_ROOT / "dnallm" / "configuration" / "evo" / config_name
        fp8_flag = _yaml_flag(config_path, "use_fp8_input_projections")
        print(f"flash_attention_installed={flash}")
        print(f"selected_config={config_name}")
        print(f"config_use_fp8_input_projections={fp8_flag}")

        configs = _load_notebook_config(EVO_CONFIG)
        from dnallm.models import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(variant, configs["task"], source="huggingface")
        load_s = time.perf_counter() - start

        from dnallm.inference import DNAInference

        engine = DNAInference(model=model, tokenizer=tokenizer, config=configs)
        forward_s, logits_shape = _forward_pass(engine.model, tokenizer)
        generate_s, generate_note = _notebook_generate(engine)
        peak = _peak_vram_gb()
        disk_gb, disk_human = _snapshot_disk_gb(variant)
        _emit_evidence(
            family,
            variant,
            load_s=load_s,
            forward_s=forward_s,
            peak_vram_gb=peak,
            disk_gb=disk_gb,
            result="OK",
            extra=(
                f"logits_shape={logits_shape}",
                f"selected_config={config_name}",
                f"config_use_fp8_input_projections={fp8_flag}",
                f"generate_s={generate_s:.1f}",
                f"generate_note={generate_note}",
                f"disk_human={disk_human}",
            ),
        )
        return 0
    except Exception as exc:
        _emit_evidence(
            family,
            variant,
            load_s=time.perf_counter() - start,
            forward_s=0.0,
            peak_vram_gb=_peak_vram_gb(),
            disk_gb=0.0,
            result="FAIL",
            failure_text=_failure_text(exc),
        )
        return 1
    finally:
        if patched and original_fp8 is not None:
            evo_module.is_fp8_capable = original_fp8


def spike_megadna(variant: str, pinned_clone: bool = False) -> int:
    """Spike the megaDNA family through the dnallm route.

    Args:
        variant: Exact HF variant (notebook: ``lingxusb/megaDNA_updated``).
        pinned_clone: D-06 fallback — put a pinned, hash-verified clone of
            ``lingxusb/megaDNA`` on ``sys.path`` before the load so the
            ``weights_only=False`` unpickle can resolve the repo's classes.

    Returns:
        0 on OK, 1 on FAIL (evidence printed either way).
    """
    family = "megadna"
    start = time.perf_counter()
    print(
        "prerequisite=none beyond dnallm base (handler torch.loads "
        "megaDNA_phage_145M.pt with weights_only=False)"
    )
    extra: list[str] = []
    try:
        if pinned_clone:
            clone_dir = Path(tempfile.gettempdir()) / "megadna-pinned-clone"
            commit = _checkout_pinned_megadna(clone_dir)
            sys.path.insert(0, str(clone_dir))
            print(f"megadna_clone_commit={commit} (pinned + hash-verified)")
            extra.append(f"megadna_clone_commit={commit}")

        configs = _load_notebook_config(MEGADNA_CONFIG)
        from dnallm.models import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(variant, configs["task"], source="huggingface")
        load_s = time.perf_counter() - start

        from dnallm.inference import DNAInference

        engine = DNAInference(model=model, tokenizer=tokenizer, config=configs)
        forward_s, logits_shape = _forward_pass(engine.model, tokenizer)
        generate_s, generate_note = _notebook_generate(engine)
        peak = _peak_vram_gb()
        disk_gb, disk_human = _snapshot_disk_gb(variant)
        _emit_evidence(
            family,
            variant,
            load_s=load_s,
            forward_s=forward_s,
            peak_vram_gb=peak,
            disk_gb=disk_gb,
            result="OK",
            extra=(
                f"logits_shape={logits_shape}",
                f"generate_s={generate_s:.1f}",
                f"generate_note={generate_note}",
                f"disk_human={disk_human}",
                *extra,
            ),
        )
        return 0
    except Exception as exc:
        _emit_evidence(
            family,
            variant,
            load_s=time.perf_counter() - start,
            forward_s=0.0,
            peak_vram_gb=_peak_vram_gb(),
            disk_gb=0.0,
            result="FAIL",
            failure_text=_failure_text(exc),
            extra=tuple(extra),
        )
        return 1


def spike_pybigwig() -> int:
    """Spike pyBigWig: import plus a small real BigWig write/read round-trip.

    Returns:
        0 on OK, 1 on FAIL (evidence printed either way).
    """
    family = "pybigwig"
    variant = "pyBigWig aarch64 sdist build + BigWig round-trip"
    start = time.perf_counter()
    print(
        "prerequisite=pip install pyBigWig in the throwaway venv (aarch64: sdist "
        "compile, needs gcc + zlib/libcurl headers)"
    )
    try:
        load_start = time.perf_counter()
        import pyBigWig

        load_s = time.perf_counter() - load_start
        version = str(
            getattr(pyBigWig, "__version__", None) or getattr(pyBigWig, "version", "unknown")
        )
        print(f"pybigwig_version={version}")

        forward_start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="spike-bw-") as tmp:
            path = os.path.join(tmp, "spike.bw")
            bw = pyBigWig.open(path, "w")
            bw.addHeader([("Chr1", 1000)])
            # Explicit-ends form needs chroms as a list (one per start); the
            # single-string form only pairs with span/step entries.
            bw.addEntries(["Chr1", "Chr1"], [0, 100], ends=[50, 200], values=[0.5, 1.0])
            bw.close()
            bw = pyBigWig.open(path)
            values = bw.values("Chr1", 0, 50)
            bw.close()
        forward_s = time.perf_counter() - forward_start
        if not values or abs(values[0] - 0.5) > 1e-6:
            raise ValueError(f"round-trip mismatch: values[0:3]={list(values[:3])}, expected 0.5")
        _emit_evidence(
            family,
            variant,
            load_s=load_s,
            forward_s=forward_s,
            peak_vram_gb=0.0,
            disk_gb=0.0,
            result="OK",
            extra=(f"pybigwig_version={version}", "round_trip=write 2 entries + read 50 values OK"),
        )
        return 0
    except Exception as exc:
        _emit_evidence(
            family,
            variant,
            load_s=time.perf_counter() - start,
            forward_s=0.0,
            peak_vram_gb=0.0,
            disk_gb=0.0,
            result="FAIL",
            failure_text=_failure_text(exc),
        )
        return 1


def _listening_ports() -> set[int]:
    """Return the set of TCP ports currently listening (empty when `ss` is absent)."""
    try:
        # ruff: ignore[start-process-with-partial-path]
        out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return set()
    ports: set[int] = set()
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4 and ":" in parts[3]:
            try:
                ports.add(int(parts[3].rsplit(":", 1)[1]))
            except ValueError:
                continue
    return ports


def _marimo_bin() -> str:
    """Resolve the marimo CLI binary that belongs to the active interpreter."""
    sibling = Path(sys.executable).parent / "marimo"
    if sibling.is_file():
        return str(sibling)
    found = shutil.which("marimo")
    if found:
        return found
    return str(sibling)  # absent: let the subprocess fail honestly


def _run_flavor(cmd: list[str], cwd: Path, timeout_s: int) -> dict:
    """Run one marimo flavor with a wall budget and port sampling.

    Args:
        cmd: Command vector to run.
        cwd: Working directory (the tmp copy of the app dir).
        timeout_s: Wall budget before the process is killed.

    Returns:
        Dict with exit code, wall seconds, timed_out flag, ports bound while
        running, and the tail of the combined output.
    """
    before = _listening_ports()
    env = {**os.environ, "MPLBACKEND": "Agg", "TOKENIZERS_PARALLELISM": "true"}
    # ruff: ignore[subprocess-without-shell-equals-true]
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    bound: set[int] = set()
    start = time.perf_counter()
    deadline = start + timeout_s
    while proc.poll() is None and time.perf_counter() < deadline:
        time.sleep(10)
        bound |= _listening_ports() - before
    timed_out = proc.poll() is None
    if timed_out:
        proc.kill()
    out, _ = proc.communicate(timeout=60)
    return {
        "exit": proc.returncode,
        "wall_s": time.perf_counter() - start,
        "timed_out": timed_out,
        "ports": sorted(bound),
        "tail": _one_line((out or "")[-400:]),
    }


def spike_marimo() -> int:
    """Spike the marimo execution flavor: export-html vs script-mode (A4).

    Runs both flavors on a tmp copy of ``example/marimo/inference`` (defaults:
    task 'open chromatin', model 'Plant DNABERT', tokenizer 'BPE', source
    'modelscope' — the warm-cached zhangtaolab model).

    Returns:
        0 on OK (flavor A produced an artifact), 1 on FAIL.
    """
    family = "marimo"
    variant = f"example/marimo/inference/{MARIMO_APP} (flavor A export-html, flavor B script-mode)"
    start = time.perf_counter()
    print(
        "prerequisite=marimo + notebook extras in the active venv; warm ModelScope "
        "cache for zhangtaolab/plant-dnamamba-BPE-open_chromatin"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="spike-marimo-") as tmp:
            work = Path(tmp) / "inference"
            shutil.copytree(
                REPO_ROOT / MARIMO_APP_DIR,
                work,
                ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"),
            )
            html_out = Path(tmp) / "out.html"

            flavor_a = _run_flavor(
                [_marimo_bin(), "export", "html", MARIMO_APP, "-o", str(html_out)],
                cwd=work,
                timeout_s=MARIMO_FLAVOR_TIMEOUT_S,
            )
            artifact_bytes = html_out.stat().st_size if html_out.exists() else 0
            print(
                f"flavor_a_exit={flavor_a['exit']} export_s={flavor_a['wall_s']:.1f} "
                f"timed_out={flavor_a['timed_out']} artifact_bytes={artifact_bytes}"
            )
            print(f"flavor_a_tail={flavor_a['tail']}")

            flavor_b = _run_flavor(
                [sys.executable, MARIMO_APP],
                cwd=work,
                timeout_s=MARIMO_FLAVOR_TIMEOUT_S,
            )
            ports = ",".join(str(p) for p in flavor_b["ports"]) or "none"
            print(
                f"flavor_b_exit={flavor_b['exit']} script_s={flavor_b['wall_s']:.1f} "
                f"timed_out={flavor_b['timed_out']} ports_bound={ports}"
            )
            print(f"flavor_b_tail={flavor_b['tail']}")

            if flavor_a["exit"] != 0 or artifact_bytes == 0:
                raise RuntimeError(
                    f"flavor A failed: exit={flavor_a['exit']} "
                    f"artifact_bytes={artifact_bytes} tail={flavor_a['tail']}"
                )
            _emit_evidence(
                family,
                variant,
                load_s=0.0,
                forward_s=0.0,  # marimo flavors are timed via export_s/script_s below
                peak_vram_gb=0.0,  # child-process VRAM is not parent-measurable
                disk_gb=0.0,
                result="OK",
                extra=(
                    f"export_s={flavor_a['wall_s']:.1f}",
                    f"flavor_a_exit={flavor_a['exit']}",
                    f"artifact_bytes={artifact_bytes}",
                    f"script_s={flavor_b['wall_s']:.1f}",
                    f"flavor_b_exit={flavor_b['exit']}",
                    f"flavor_b_timed_out={flavor_b['timed_out']}",
                    f"ports_bound={ports}",
                ),
            )
            return 0
    except Exception as exc:
        _emit_evidence(
            family,
            variant,
            load_s=time.perf_counter() - start,
            forward_s=0.0,
            peak_vram_gb=0.0,
            disk_gb=0.0,
            result="FAIL",
            failure_text=_failure_text(exc),
        )
        return 1


def _run_family(family: str, fallback: bool) -> int:
    """Dispatch one family, selecting notebook variant or D-06 fallback."""
    if family == "evo1":
        return spike_evo1(EVO1_FALLBACK_VARIANT if fallback else EVO1_NOTEBOOK_VARIANT)
    if family == "evo2":
        return spike_evo2(EVO2_NOTEBOOK_VARIANT, "noFA-noFP8" if fallback else None)
    if family == "megadna":
        return spike_megadna(MEGADNA_NOTEBOOK_VARIANT, pinned_clone=fallback)
    if family == "pybigwig":
        return spike_pybigwig()
    if family == "marimo":
        return spike_marimo()
    print(f"failure_text=unknown family: {family}")
    return 1


def main() -> int:
    """Parse args, guard the GPU, run the requested families fail-closed.

    Returns:
        0 when every requested family emitted a complete evidence block — an
        OK or FAIL verdict is matrix evidence, so FAIL does not turn the run
        red; 1 when any family crashed before emitting its block (the
        infrastructure-failure signal for the workflow step); 2 when the GPU
        guard failed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "GB10 feasibility spike runner (FEAS-01, D-04/D-05/D-06): loads each "
            "environment-gated family's exact notebook variant through the dnallm "
            "route and prints machine-greppable evidence lines."
        )
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=[*FAMILIES, "all"],
        help="family to spike, or all",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help=(
            "run the family's smallest-viable variant/config instead of the notebook "
            "variant (D-06: evo-1-8k-base / evo2 noFA-noFP8 config / pinned megaDNA clone)"
        ),
    )
    args = parser.parse_args()

    if not _gpu_guard():
        return 2

    families = list(FAMILIES) if args.family == "all" else [args.family]
    failed = []
    crashed = []
    for family in families:
        mode = "fallback variant" if args.fallback else "notebook variant"
        print(f"===== spike {family} ({mode}) =====")
        # Every spike_* emits a complete evidence block (OK or FAIL) on both
        # its success path and its except path, so a normal return means the
        # block exists. Only an exception escaping the family function — or
        # the process dying outright — leaves a family unevidenced.
        try:
            if _run_family(family, args.fallback) != 0:
                failed.append(family)
        except Exception as exc:
            crashed.append(family)
            print(f"family={family}")
            print("result=CRASH")
            print(f"failure_text={_failure_text(exc)}")
            print()
    if failed:
        print("failed_families=" + ",".join(failed))
    if crashed:
        print("crashed_families=" + ",".join(crashed))
    # Exit code = evidence completeness, not verdict (WR-04): FAIL verdicts
    # are matrix evidence and exit 0; only a pre-evidence crash exits 1 so
    # the workflow step can go red without masking family failures.
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
