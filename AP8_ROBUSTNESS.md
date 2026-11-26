# AP8A — bounded development robustness protocol

## Authorization and scope

On 22 November 2026 Andreas accepted the proposed first AP8 section: prepare and implement a bounded robustness protocol, then at most **four new real neural fits (CNN seeds1/2, MLP seeds1/2)**. Per fit: wall1800s, successful measured CPU<1200s, address-space2048MiB. Reuse the two selected seed0 references; do not retrain them. No additional retries, replays, searches or rolling-fit batches without another budget decision.

This is **AP8A**, not completion of the whole roadmap's independent final/rolling robustness stage. **2025 remains closed to model inputs and evaluation.** Do not reopen raw2021–2025 import: use only verified sealed 2021–2024 development/training and2024 validation artifacts. Model and trading rules remain fixed: eight m15 candles, horizon2, long-only quantity1, threshold0, SMA4/8; no new TP/SL, architecture, instruments or website job controls.

Base repository commit: `89b89c9dd7ec4e4ee291fbe5ca39ecee07b7ca85`. Existing AP1–AP7 configs, reports, registries, live site and signatures remain accepted historical records. AP8 is additive and local; no automatic Git push/service update authorization is inferred.

## Predeclared source references

- CNN seed0: `experiments/ap7-acceptance-20261118-v1/run-49393a443991482eb9b58eb9681336f0/artifacts`.
- MLP seed0: `experiments/ap5-acceptance-v1/run-ff5344830dd5430a8bb42472d88f3df1/artifacts` (fixed4→16→8→1, learning_rate0.001, max20, patience10). Do not substitute the100-epoch copied AP3 reference or choose the best baseline after looking at new results.
- New CNN retains AP7 fixed5409parameter policy,max20,patience5; only seed differs. Add a distinct `market-trader.cnn-validation.v2` artifact for seeds1/2; AP7v1 must continue rejecting nonzero seeds.
- New MLP retains the exact AP5 reference model settings and training source; its existing v2 model format already records seed.
- Verify both source records/manifests, exact settings, and equality of every eligible train/validation decision, numeric feature and target row before treating them as the common research cohort. Different artifact/provenance IDs are disclosed, not silently replaced.

## Diagnostics fixed before new fits

All are **2024 development diagnostics**: each model uses2024 checkpoint selection, so slicing2024 is NOT independent out-of-sample testing.

- All eligible validation forecasts: MAE/RMSE and zero-return comparator.
- Monthly12, quarterly4 and Europe/Berlin decision-time bins09:00–12:00,12:00–15:00,15:00–17:30. These are calendar/time-of-day slices, not a post-hoc bull/bear regime classifier. Empty slices explicitly have count0 and no fabricated metric.
- Per-family results for seed0(reused),1,2: every individual result and range/mean; no best-seed deployment/ranking or significance claim from three seeds. Explicitly separate RNG variation from cross-platform replay.
- Cost factors **0.5,1,2** multiply spread, slippage and commission only; price_tick, quantity, decisions, source data and model forecasts remain fixed. Re-run the EXISTING execution ledger using verified frozen causal forecasts, never estimate cost changes by subtracting a guessed fee from old totals. Tick rounding may prevent proportional cost changes.
- Cached forecasts are input-only(at,predicted) inside strategy code; actual targets are only for scoring. Strategy lookup may use only the current eligible decision timestamp. Verify factor1 reproduces the reference execution/P&L/trade sequence before accepting cost comparisons. Do not reset a quarter's trade engine and pretend it is a continuous backtest.
- Hold on missing/ineligible forecasts; reject mismatched/extra/duplicated/nonfinite/out-of-period rows before simulation. Preserve current session/horizon exit eligibility and timed exits.

## Execution and integrity

Strict RED→GREEN synthetic tests, optional-framework isolation, independent review and source freeze before real fits. New source cohort/cache remains bounded/private; no multi-GB PNG corpus. Preserve training-only scalers, actual-minimum checkpoints and all-validation framework/stdlib parity.

Use the existing opaque-snapshot experiment registry and existing backtester/ledger. New registry is private and separate; source lineage records alias/run/artifact/model IDs explicitly rather than inventing same-registry parents. New summaries/signatures are AP8-specific; do not reinterpret old snapshots to make them compare equal. No browser mutation or new live registry alias in this section.

Each fit runs sequentially in its own bounded owned subprocess, with durable invocation/ownership claims, exact source snapshot, CPU/address-space limits, wall termination and parent-death protection. Maximum four claims; a failed/timeout invocation is not silently retried. All source data/site files are hashed before and after; no other queued jobs are drained.

## Acceptance

1. Both source references and common cohort verified; fixed protocol/config snapshots saved before fits.
2. Unit/synthetic integrations and independent security/logic review pass; complete suites sequentially on frozen source.
3. Four actual fits at most, true backend/weights/history/seed/checkpoint evidence, metrics over ALL validation rows, canonical reload predictions and actual cost-ledger reconciliation.
4. Original references reused without modification; baseline factor1 parity and input-only forecast strategy verified.
5. Separately installed framework-free inference/report check, escaped bounded standalone report with all seeds/slices/costs and explicit development-only limitations; actual desktop/mobile browser verification.
6. Final AP8A evidence/results and limitations, including failures, not merely a plan/stub. Separate approval remains necessary for reserved2025 evaluation, further rolling fits and publication/Git.
