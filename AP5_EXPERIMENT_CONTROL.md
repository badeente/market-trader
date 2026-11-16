# AP5 — controlled local variants and training jobs

Andreas authorized AP5 after accepting AP4. Base: `615e5fa`. This stage implements local CLI experiment iteration and bounded manual searches, not a browser mutation service. The live service stays read-only. Andreas separately authorized commit, push and internal publication on 16 November 2026; the accepted AP5 wheel and additional AP5 registry are now published without browser mutation endpoints. No final-test/2025 evaluation, live trading, image models, GPU/paid services or unbounded optimization.

## Acceptance and scope

- Derive a new, strictly validated configuration from a verified stored parent, explicitly naming the hypothesis and changes. Preserve the parent and previous snapshots. Changes distinguish model/training settings, trading rules and data/cohort conditions.
- Add explicit experiment config v3 and configurable MLP JSON v2. Legacy experiment v1/v2 and MLP JSON v1 retain frozen AP2/AP3 semantics. CPU float64, one thread, ReLU, AdamW, train-only X/Y standardization, validation checkpoint selection and safe JSON inference remain fixed. Same four features and 2021–2023/2024 split; physically exclude2025 before features.
- MLP parameters: exactly two hidden layers, widths4–64; architecture `[4,H1,H2,1]`. Learning rate0.00001–0.01, weight decay0–0.1, batch16–512, max epochs1–100, patience1–20, min_delta0–0.01; finite strict values (no bool). Existing defaults unchanged. Model version2 carries the actual architecture/fit settings and checks parameter counts/shapes/history against them. v3 may vary causal lookback/horizon under existing session constraints; such changes are explicitly incompatible comparison conditions, not pure hyperparameter variants.
- New private local queue: queued/running/completed/failed/cancelled/timed_out/interrupted, immutable submitted snapshot, append-only events, parent/changes/budget and linked run ID. Fixed Python worker command; no arbitrary commands, executable paths or remote endpoints. Reuse actual experiment registry/pipeline/ledger, not an alternate simulator.
- One worker per queue and one training subprocess at a time. Cooperative cancel request, bounded TERM/KILL escalation of owned subprocess group, wall-clock/CPU/address-space limits, crash/stale job recovery without automatic retries. Cancellation/timeout cannot be advertised as a completed run; unfinished legacy registry runs become failed with reason, while job state provides the more specific result.
- Default per-job bounds wall300s/CPU180s/address-space4096MiB; absolute caps wall900s/CPU600s/address-space8192MiB. At most32 queued jobs, explicit finite worker max-jobs and total-time budget; no daemon auto-install.
- Declarative sweep expands at most4 unique variants after full preflight; no adaptive optimization/ranking on final test. Search objective is validation RMSE, with costs/trading results reported separately; no profit promise. Keep provenance and failed/cancelled runs visible. No partial submission if configuration validation fails. Batch validation and rollback cover ordinary exceptions, not a power-loss/SIGKILL transaction guarantee.
- AP5 v3 comparison signatures additionally include both SMA reference periods because these settings are classified as benchmark conditions. Legacy v1/v2 signature meaning stays unchanged.
- Demonstrate real baseline/variant jobs and replay in fresh private locations, status/error/cancel/timeout behavior, source-artifact immutability and AP4-compatible read-only projection. Bounded actual research budget: at most4 new fits, each max20epochs, plus one replay if needed within4 total; max1800s worker wall budget. Synthetic lifecycle tests do not masquerade as research data.
- Verify full suites with/without Torch, installed wheel outside checkout, actual no-Torch reload, independent reviews and documented results/limits. No changes to accepted reference files or source registries.

## Ownership and shared interface

Core owner: `experiment_config.py`, `neural.py`, `experiment_pipeline.py`, own new `tests/test_ap5_model.py`. Config v3 model uses the existing full canonical model dictionary with `layers`, not a second hidden-layer naming convention. TOML v3 may specify allowed adjustable fields; canonical snapshot includes all fixed/adjustable settings. Pipeline passes v3 model settings explicitly, while legacy calls/output remain frozen. `mlp_report(path,seed=0,settings=None)` keeps old default behavior; `settings` is the complete v3 canonical model dictionary. v2 JSON dispatch remains usable via existing `load_mlp_model` / `mlp_validation_evaluation` / strategy/report APIs.

Job owner: new `experiment_jobs.py` and `experiment_job_worker.py`, own `tests/test_experiment_jobs.py`. Do not edit CLI/config/neural/pipeline/live data. Public API:
- `submit_job(queue:Path, registry:Path, payload:dict, *, parent_id:str|None=None, budget:dict|None=None, metadata:dict|None=None)->dict`
- `list_jobs(queue:Path)->list[dict]`, `read_job(queue:Path,job_id:str)->dict`, `cancel_job(queue:Path,job_id:str)->dict`
- `work_jobs(queue:Path, *, max_jobs:int=1, total_seconds:int=900)->dict` drains bounded jobs then returns a summary.
- `budget` uses `wall_seconds`, `cpu_seconds`, `memory_mib`; validate integer bounds and reject bool/unknown keys. Full config validation before writing. Worker `python -m market_trader.experiment_job_worker --queue ... --job-id ...` only. The optional backward-compatible `register_run(..., on_allocated=...)` hook durably links the exact claimed run directory before any config/pending/running record publication; linking only inside the executor is too late for crash recovery.

Parent owns new `experiment_variants.py`, CLI/orchestration, integration tests, actual fits/replay, reviews and docs. Other changes by coordination only. All owners use RED→GREEN tests and Graphify-first/source checks. No agent commits, pushes, changes services or evaluates2025.

## Local operator workflow

Use the separate CPU training environment (or an installed AP5 training wheel), not the running web environment. `RUNS`, `QUEUE`, and outputs below are explicit private paths; `PARENT` must already exist in `RUNS`. For a fresh registry, first queue a baseline TOML and drain that job. To reuse an old reference in a separate research cohort, copy and verify the complete immutable parent record outside the original registry; never alter the accepted source cohort.

```sh
market-trader experiment derive --registry "$RUNS" --run-id "$PARENT" \
  --name "Smaller MLP" --hypothesis "Controlled width comparison" \
  --set 'model.layers=[4,8,4,1]' --set 'model.max_epochs=20' \
  --output /private/new-variant.json
market-trader experiment job submit --queue "$QUEUE" --registry "$RUNS" \
  --plan /private/new-variant.json --wall-seconds 300 --cpu-seconds 180 --memory-mib 4096
market-trader experiment job work --queue "$QUEUE" --max-jobs 1 --total-seconds 600
market-trader experiment job list --queue "$QUEUE"
market-trader experiment job show --queue "$QUEUE" --job-id "$JOB"
market-trader experiment job cancel --queue "$QUEUE" --job-id "$JOB"
market-trader experiment job submit --queue "$QUEUE" --registry "$RUNS" --replay-run "$RUN"
```

`derive` writes a new JSON plan, not a TOML file and not a completed result. It refuses overwrites/no-op changes; use `--replay-run` for an unchanged configuration. `submit` queues only; `work` explicitly performs the bounded execution. Queued cancellation is immediate; running cancellation requests TERM/KILL of the controller-owned subprocess. Job `run_id` links to the existing experiment registry. The worker must use the intended interpreter; a read-only/no-Torch environment can inspect/reload results but cannot fit an MLP.

A grid file contains e.g. `{"model.learning_rate":["0.0005","0.002"]}`. Run `experiment sweep` with the same parent/name/hypothesis options, `--grid /private/grid.json`, optional fixed `--set 'model.max_epochs=20'`, and a new `--output /private/sweep.json`. This creates a prevalidated plan of at most four unique configurations. Submit it via `job submit --plan`; batch validation/capacity checks occur before allocation. Drain with explicit `--max-jobs` and total budget. Then `job results --queue "$QUEUE" --job-id "$FIRST" --job-id "$SECOND"` verifies completed compatible results and orders them by validation RMSE, displaying trading results separately. Incompatible conditions have no selected best run. This is a small declared development comparison, not an optimal-model claim.

Read-only AP4 projections can consume v3 runs when running the AP5-capable package. The separately authorized live update now uses the accepted AP5-capable wheel and includes the AP5 registry while preserving old aliases and report URLs. Browser training controls remain deferred pending authentication, CSRF protection and an explicitly designed job-control service.
