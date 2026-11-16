# AP4 – read-only experiment workbench (approved scope)

Andreas explicitly approved AP4 after AP3. Base commit3093599. Build on the existing private report service at10.10.1.136:8086; preserve actual live client/Host ACLs and all existing report URLs. No training/start/stop/tuning/upload/write endpoints, public exposure or reserved2025 evaluation.

Implementation, acceptance and live publication: [AP4_RESULTS.md](AP4_RESULTS.md).
Andreas separately authorized commit, push and internal publication on
16 November 2026. The accepted r2 wheel is now live on the existing service,
with original report URLs and LAN/VPN allowlists preserved. No AP5 approval.

## Acceptance

- Structured real-run overview with source/model/status/search/filter, parameters/identities and links to detail reports.
- Detail views with actual configuration, primary and baseline forecast metrics, trading costs/trades/net/drawdown, training history and equity/drawdown curves kept conceptually separate.
- Select2–4 completed runs; show condition and parameter differences. Explicit incompatible warning for different data/cohort/cost/code/runtime; do not manufacture a best-model recommendation.
- Use actual final registries, not fake demo data: ap2=var/market-trader/experiments/ap2-acceptance-v2 and ap3=.../ap3-acceptance-v2. These contain completed Ridge/MLP/replays, changed costs and a failed import. Do not mutate or retrain them.
- Read-only integrity-checked JSON projections; no arbitrary filesystem paths/files. Root paths, raw inputs, model weights and private logs stay off HTTP.
- Responsive self-contained German UI with no third-party CDN/fonts/scripts. Dark precise research dashboard; system fonts, restrained violet/emerald accents, visible negatives and explicit units. Not a trading terminal.
- Real API/visual/browser/network/security/performance checks, independent review, existing/new tests, installed-wheel deployment and rollback plan before declaring complete.

## Shared module/API contract

### Data owner: workbench_data.py

`WorkbenchData(registries: dict[str,Path], *, curve_limit:int=800)` with methods:
- overview()->dict
- detail(key:str)->dict
- compare(keys:list[str])->dict
- report(key:str)->bytes

Keys are exact `<source>:<run-id>`; source aliases strict lowercase identifiers, IDs existing run-32hex. `UnknownRunError(ValueError)` maps404; `InvalidRunError(ValueError)` maps409; other ValueError maps400. No imports of trainers or Torch; use existing verified registry data/snapshots/manifests, not model fitting. Cache only after verified reads; cache fingerprint must detect changes to record/config/manifest/every declared artifact and source-directory identity. Reverify on mutation; no stale metrics after detected corruption. Serialise expensive cache fills and bound cache size. Invalid records are visible as invalid, not silently completed or zero-valued.

Overview schema:
`{runs:[row...], sources:[{id:alias}...], errors:[safe messages...], read_only:true}`.
Each row:
`key,source,run_id,name,hypothesis,model_kind,status,created_at,parent_id,selectable,training_rows,validation_rows,dataset_id,source_sha256,period,metrics,report_url`.
`period` contains nullable `training_start,training_end,validation_start,validation_end,reserved_test_start,reserved_test_end` from saved metadata, not run-creation time. Custom curve limits are integers6–800.
Metrics keys: `mae_bp,rmse_bp,net_points,cost_points,drawdown_points,trade_count`. Noncompleted has null metric values, never fake zero. Decimal point values may be strings; frontend formats German. Model primary=explicit config.model.kind, default Ridge for oldv1.

Detail schema:
`{run:row, config:sanitized_snapshot, signature:sanitized_conditions, runtime:safe_metadata, metrics_by_model:{model:{mae_bp,rmse_bp}...}, trading_by_model:{model:summary...}, training:null_or_{history:[{epoch,training_mse,validation_mse}],best_epoch,stopping_reason}, curves:{equity:[[at,net_points]...],drawdown:[[at,drawdown_points]...],source_count:int,sampled:bool}, warnings:[strings], error:null_or_safe_message}`.
Curves derive from actual primary backtest accounts, relative to saved initial capital, drawdown in points from prior peak. Downsample with endpoints/extrema preservation when needed; label sampled graphs and retain full-source metric totals. Training losses are standardized target MSE, not bp/return. Sources/backtests remain2024.

Comparison schema:
`{compatible:bool,runs:[row...],condition_differences:[{left:key,right:key,differences:[{field,left,right,...}]}],parameter_differences:[same pair schema],details:[detail...],warnings:[strings]}`.
Compute real differences before redacting private paths. Compare2–4 unique completed valid keys, across configured sources; same old registry signatures govern compatibility. Distinguish model choice/metadata from fair conditions. Sensitive path values in configs/differences/error messages redacted; no absolute source paths.
Capture each verified entry once for displayed details and comparison metadata;
recheck all selected fingerprints after computation and reject changed snapshots.
Name/hypothesis are displayed as run metadata, not parameter differences.

Report URLs row.report_url: `/experiment-report?run=<URL-encoded-key>`. Return only the checked fixed artifacts/report.html (max32MiB), no generic file endpoint.

### HTTP owner

Extend create_report_server with optional `registries:dict[str,Path]|None=None`; defaultNone keeps old gallery behavior/tests. CLI serve-reports accepts repeatable `--registry alias=/absolute/path` and passes exact mapping after strict validation. Use WorkbenchData only when enabled. Routes:
- `/` and `/experiments`: workbench HTML when enabled
- `/reports`: old HTML gallery when enabled; `/reports/<filename>.html` unchanged
- `/api/experiments`: overview JSON
- `/api/experiment?run=key`: detail JSON
- `/api/compare?run=key&run=key[...max4]`: comparison JSON
- `/experiment-report?run=key`: verified report HTML

GET/HEAD only; preserve client pre-worker rejection, Host checks, pinned/no-follow report directory, limits/security headers. Allow `connect-src 'self'` for same-origin API fetches, no CORS or external sources; no-store. Reject malformed/duplicate/oversized queries and unknown source/run; don't expose Python exceptions or paths. Preserve no-registry root gallery behavior. No live deployment by child.

### UI owner

New workbench_template.html.in with static self-contained HTML/CSS/JS, placeholderfree (data fetched from above routes). No libraries/CDNs. Element text from API via textContent or equivalent escaping, never interpolate user strings into innerHTML. Accessible controls, focus states, selection max4, min2, failed/invalid disabled, filtered-out selections preserved or explicitly indicated, clear selected list/reset. Fetch errors/empty/loading/invalid/incompatible states. Deep links via query run or compare accepted; direct reload works. Existing reports via /reports.

Show full model/parameter/condition detail without arbitrary JSON dumps as primary UI. Forecast metrics in bp, trading in points per fixed unit, negative sign preserved. Distinct training/validation loss charts and net-equity/drawdown charts with legends/units and conservative validation/selection caveats. No fake sample metrics when empty.

Parent owns integration/real QA/deployment/docs. Children use bounded owned files, strict TDD and no new external runtime dependencies. Source graph is navigation only; verify actual code/tests. Preserve existing work.
