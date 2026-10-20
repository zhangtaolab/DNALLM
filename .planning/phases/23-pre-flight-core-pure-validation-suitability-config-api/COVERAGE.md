# API Coverage — Hub metadata APIs (huggingface_hub + modelscope)

> Full coverage by default. Opt-outs are explicit, reasoned decisions.
> The phase-23 probe layer (`dnallm/preflight/availability.py`) integrates the hub metadata
> APIs of two external services through their pinned client libraries (`huggingface_hub` 1.33.0,
> `modelscope` 1.34.0). No raw REST is hand-rolled (research "Don't Hand-Roll").

| capability | decision | reason |
|---|---|---|
| HF `repo_exists` (remote existence) | INTEGRATE | |
| HF `model_info(files_metadata=True)` (size, revision `.sha`, file list) | INTEGRATE | |
| HF `scan_cache_dir()` (local cache census) | INTEGRATE | |
| HF `try_to_load_from_cache` (offline-safe snapshot probe) | INTEGRATE | |
| MS `HubApi.repo_exists` (remote existence) | INTEGRATE | ModelScope-first probe order (owner directive) |
| MS `HubApi.get_model_files` (file sizes, revision) | INTEGRATE | |
| MS `HubApi.get_model` (model metadata record) | INTEGRATE | |
| MS `cache_manager.scan_cache_dir` (local cache census) | INTEGRATE | optional bulk view; raises CacheNotFound when root absent — guarded |
| HF `snapshot_download` / MS `snapshot_download` (weights) | OPT-OUT | probe never downloads weights (PFC-03); MS snapshot_download on a miss also litters `.mdl` (research Pitfall 2); downloads are Phase 25's select-to-download |
| HF `hf_hub_download` (single file content, e.g. config.json) | OPT-OUT | runtime probe needs metadata only; file-content fetches were 22-02 one-time tooling |
| HF/MS auth & token management (`whoami`, token creation) | OPT-OUT | no credentials handled by preflight; ambient library token handling only |
| HF/MS upload / repo creation / commit | OPT-OUT | read-only phase; registry loader stays read-only forever |
