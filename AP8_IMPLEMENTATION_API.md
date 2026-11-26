# AP8A shared implementation APIs

Read AGENTS/GRAPHIFY and AP8_ROBUSTNESS.md. Parallel owners must not edit each other's files. All development tests synthetic; only parent runs the four real fits after frozen-source review. No Git/live actions. Preserve AP1–AP7 semantics.

## A — seeded CNN (owner image_neural.py, NEW image_neural_seeded.py, owned tests)

Provide:

- `seeded_cnn_report(dataset_path, training_path, *, seed:int, cache_dir=None)->dict`.
- `load_seeded_cnn_model(path)->SavedCNNModel` (reuse immutable saved CNN class/kernel).
- `seeded_cnn_validation_evaluation(model_path,dataset_path,training_path, *,cache_dir=None)->(evaluation,points)`; points exact `{at,predicted,actual}` floats/UTC strings, same all-validation convention.
- New format **market-trader.cnn-validation.v2** for exactly seeds1/2, otherwise same frozen settings/shape/schema/bindings as AP7v1. Evaluation format `market-trader.cnn-seeded-validation-evaluation.v1`.
- Extract parameterized PRIVATE fitting/decoding helpers only if needed. AP7 public `cnn_report`, `_fit` and `load_cnn_model` retain seed0/v1 defaults and strict rejection of nonzero seeds/v2 artifacts. No monkeypatching globals, metadata relabelling or training seed0 then renaming it. New decoder validates actual stored seed1/2, finite/shape/history/parity/source/recipe bindings as strictly as v1.
- Reuse bounded uint16 cache/memmap, target scaler, measured all-validation Torch/stdlib parity, state restoration and actual-min checkpoint policy. Optional Torch/NumPy laziness preserved. Original chart_images.py/image_pixels_v1.py recipe files DO NOT change.
- Strict TDD, synthetic seed1/2 optimizer/reload/parity and deterministic repeated fixture tests, malformed re-sealed v2 and old-v1 compatibility. No full suites while others edit.

## B — diagnostics and true cost replay (owner NEW robustness_analysis.py and tests only)

Provide:

- `forecast_slices(points)->dict` exact point list convention, nonempty maximum50000, chronological unique UTC2024, finite numeric no bools. Output `{format:'market-trader.ap8-slices.v1', timezone:'Europe/Berlin', overall:{row_count,mae,rmse,zero_mae,zero_rmse}, months:[{key:'2024-01',...metrics}x12], quarters:[{key:'2024-Q1',...}x4], sessions:[{key:'09:00-12:00',...},...12:00-15:00,15:00-17:30]}`. Empty metrics are None with row_count0. Reject timestamps outside research time bins rather than dropping counts. All counts reconcile; no independent-test claims.
- `cost_backtests(dataset_path, points, *, expected_decisions, model_report_id, base_profile)->iterator[(scenario:str, report:dict, summary:dict)]`.
  - expected_decisions tuple UTC datetimes from VERIFIED source training validation rows; exact one-to-one at equality, no missing/extra/duplicate/nonfinite/bool rows. Scoring targets NEVER passed into strategy closure; test changing actual leaves trades untouched.
  - base_profile ExecutionProfile from saved source backtest `execution`, not a guessed default; enforce fixedlong-only/quantity1/noTPorSL/lookback8/horizon2 and no2025. Dataset m15/deuidxeur only2024, SessionCalendar. Shared threshold0 and session eligibility exactly `_learned_long`; no extra strategy/exit behavior.
  - Yield scenarios `'half','baseline','double'` in that order; factors0.5/1/2 affect only spread/slippage/commission, use dataclasses.replace and existing run_backtest. No training or second simulator, no percentage arithmetic on old P&L.
  - Backtest report JSON-compatible with existing BACKTEST_REPORT_FORMAT, strategy name`ap8_frozen_forecasts_long`, metadata model_report_id/lookback8/horizon2/entry_threshold0/quantity1 and scenario/factor/input-only forecast digest. Normalized execution field (not profile), stored manifest/provenance and declared runtime, asdict existing results→encode/json. Call existing ledger validation/trading_summary. Generator releases each report before building next; no whole-study all-accounts tensors.
  - Only current timestamp lookup, hold on noneligible timestamps, timed exits prioritized. Reconcile factor1 against real/synthetic direct learned strategy in tests. Invalidcost/row/date inputs fail before executing.
- Only this module/tests; do not edit strategies.py/backtesting.py/core model files. StrictTDD + real synthetic ledger tests.

## C — self-contained AP8A study HTML (owner NEW robustness_reporting.py/tests only)

Provide `render_robustness_report(study:dict)->bytes`. No files/network/framework/fits; bounded escaped self-contained HTML <=8MiB, no external resources. Full required study schema:

```
format='market-trader.ap8-study.v1'
status='complete'
protocol='ap8a-development-robustness-v1'
test_metrics_computed=False
reserved_test_accessed=False
period={'train':'2021-2023','validation':'2024','reserved':'2025'}
new_fit_count=4
reference_reuse_count=2
runs=[{model_kind:'cnn'|'mlp',seed:0|1|2,reused:bool,run_id:str,
       model_report_id:str,training_rows:int,validation_rows:int,
       training:{epoch_history:list,best_epoch:int,stopping_reason:str,...},
       forecast:{row_count:int,mae:float,rmse:float,zero_mae:float,zero_rmse:float},
       slices:forecast_slices output,
       costs:{half:trading_summary,baseline:trading_summary,double:trading_summary}} x6]
limitations:list[str]
```

- Validate exact six unique(model_kind,seed) pairs, seed0reusedTrue othersFalse, counts4new2reused; refuse missing/failed runs, bogus/bool/nonfinite numbers/negative errors. MLP/CNN policies differ and representations not equivalent, no automatic best-seed selection/claims.
- Render per-family all seeds and descriptive range/mean, individual forecast and cost/trade totals, per-run calendar/time slices, actual histories with best checkpoint. Existing `training` uses actual `training_mse` or `train_mse` field; accept and validate coherent history. Empty slices have None errors notzero.
- Warnings: all2024checkpoint-selected development, NOT independent2025/rolling test; no regime classifier/inferred profitability/significance. No fake attributions/omittedbadseeds. First2referencesnotnewfits.
- Accessible sections/details, tables contained horizontally, wrapping IDs, img-free or inlineSVG only. Desktop/mobile CSS with no global minwidth; escape all labels/adversarialmetadata; bounded input rows before expansion. Unknown extra top fields can be ignored but required shape/count/type mustvalidated. Add syntheticfixtures only; do not fit to create testfixtures.

## Parent

Own protocol/docs/CLI integration, NEW robustness_pipeline.py/tests and acceptance drivers. Use existing opaque-snapshot register_run (no experiment_config v5 required), AP8-specific summaries/comparison signatures and existing manifest verification. Local separate registry only; not projected through old Workbench as if it were v4. Validate source references/settings/everytrain/val feature+target row, seal input/source lineage; run seeds1/2 only and reuse0. Four kernel-limited sequential fit workers with no automatic retries. Reuse B for factor1 parity +cost/temporal diagnostics, C for complete study. Freeze source/full suites/reviews first; no live/Git approval inferred.
