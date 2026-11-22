# AP7 implementation handoff — frozen shared APIs

Read AP7_VISUAL_REFERENCE.md and AGENTS.md. This handoff fixes interfaces for parallel TDD; do not edit another owner's files. No real fits/live/Git changes by implementation delegates. Parent performs stable-source acceptance. Architecture is the measured **5409-parameter single-convolution+flatten CNN**, NOT the alternative417-parameter proposal.

## Owner A — image pixels / full cohort / compact cache

Own NEW `image_pixels_v1.py`, `image_training_data.py`, tests for these. No AP6 module edits or expansion of audit-bundle limits.

`image_pixels_v1.py` is a small frozen preprocessing module whose file SHA is part of its recipe (separate from mutable cohort/cache code):

- `image_input_recipe() -> dict`: includes format/version, AP6 recipe identity, own implementation SHA, RGB8 PNG256x192, fixed8x8 integer block sums, CHW[3,24,32], uint16-little-endian storage, max sum16320, float64 conversion `sum/64/255`, recipe_id computed from canonical body.
- `block_sums(png:bytes, *, accelerated:bool=False) -> tuple[int,...]`: exactly2304 Python ints CHW; strict bounded canonical AP6 PNG decoder checks signature/chunks/CRCs/IHDR and bounded decompression/filter0, no extra metadata. No labels/window/targets arguments. Optional explicit NumPy accelerator must match exact sums, imports only if requested. Default stdlib.
- `normalized_input(sums) -> tuple[float,...]`: validates exact2304 bounded integer values (no bool), fixed sum/64/255. No fitted statistics.

`prepare_image_cohort(dataset_path:Path, training_path:Path) -> ImageCohort`:

- Frozen data object fields `.data` (verified NormalizedDataset), `.training` (existing TrainingDataset with train/validation, empty test), `.training_data_id` (original sealed numerical training ID), `.dataset_artifact_id`.
- Reconcile ALL source rows/features/targets with original build_open_training_data/checked decoder and fixed8/2, lowercase deuidxeur, session2021-24 only. Resource-bound source rows (maximum50000 eligible samples; no full PNG corpus).
- `.window(row) -> tuple[Candle,...]`: exact historical8 contiguous/available same-session candles ending at row.decision_at.
- `.index_document() -> dict`: format `market-trader.image-index.v1`, input_recipe (complete), ordered samples train then validation with sample_id/split/decision_at/image_key; `input_index_id` hashes canonical body excluding that key. Index contains input metadata only; source global/training/label IDs belong to model bindings, NOT index/cache identity.
- `.labels_document() -> dict`: separate ordered sample_id/entry_at/exit_at/target_available_at/target_return rows, source_training_data_id and its own identity. No labels passed to block_sums/rendering.

`build_input_cache(cohort, cache_dir:Path, *, accelerated=False) -> InputCache`:

- `.path:Path` points to headerless packed `<u2` row-major[train+validation,3,24,32]; `.row_count`, `.train_count`, `.input_index_id`, `.sha256`; `.row(index)->tuple[int,...]` for stdlib inference.
- Explicit private current-owner0700 cache dir/parent, bounded no-follow regular-file reads, no corrupt-cache repair/overwrites, atomic new-file publication; hardlink/symlink checks. Cache JSON stores dimensions/byte length/data hash/index and input-recipe identity, no labels. Existing cache verification must be bounded; trust owner, not a coherent malicious-owner rewrite sandbox. Torch fitting may use np.memmap on this verified headerless file; no all-cohort float64 tensor or convolution allocation.
- Parent passes a private temporary cache beneath its experiment staging dir and removes it before final manifest publication. CNN fit/evaluation share it. Replay may regenerate, no persistent cache path is part of config semantics.

## Owner B — fixed settings / CNN model

Own NEW `cnn_settings.py`, `image_neural.py`, own settings/model tests. Use API A via lazy imports; coordinate mismatches before changing.

- `cnn_settings.canonical_cnn_settings(value:dict|None=None, *, snapshot=False)->dict`: default/fixed policy. TOML accepts kind-only CNN (no tuning fields); snapshot requires complete exact schema/types/values. Kindcnn, architecturetiny_cnn_v1, input_shape[3,24,32], Conv3valid3→4/ReLU, AvgPool2, Flatten660, Linear8/ReLU/Linear1,5409params. CPUfloat64one-thread deterministic; AdamWlr.001/wd.0001/betas[.9,.999]/eps1e-8/amsgradFalse/foreachFalse/fusedFalse/maximizeFalse; batch128,max20,patience5,min_delta1e-6, targettrain-only-standard, fixedRGB normalization. Store numeric policy values as canonical strings where appropriate, reject bool/int aliases deeply. Parent config imports this module; avoid importing experiment_config back.
- `cnn_report(dataset_path, training_path, *, seed=0, settings=None, cache_dir=None)->dict`: real fit only when called, Torch/NumPy imports only inside fitting. Default private temporary cache if not supplied. Minibatch uint16→float64 conversion; target scaler fitted on train only. Preserve actual minimum checkpoint independently of patience/min_delta; restore global RNG/threads/determinism/default dtype. Immutable safe JSON report format `market-trader.cnn-validation.v1` with report_id, instrument/interval/config/target_name, training_data_id=input_index_id, source_training_data_id, dataset_id/artifact_id, input_recipe/index bindings, fixed settings, model weights/shapes/target scaler, training metadata/history/runtime.
- Training metadata follows existing neural fields row_count/first_decision_at/last_target_available_at/fit_settings/runtime_versions/epoch_history/best_epoch/stopping_reason; epoch fields match existing history (epoch,train_mse,validation_mse). Explicitly validate selection and train cutoff.
- `load_cnn_model(path)->SavedCNNModel`: bounded strict JSON/schema/hash/shape/finiteness/history/recipe checks, immutable tuple weights; no Pickle/torch.load/arbitrary operators. Stdlib inference with measured Torch parity.
- Saved object fields `.report_id`, `.instrument`, `.interval`, `.config`, `.last_target_available_at`, `.target_name`, `.model`; `.predict_window(candles, *,decision_at)->float` and `.model.predict_sums(sums)->float`, already inverse target-transformed. `.model.predict_window` alias optional, parent/report use saved.predict_window.
- `cnn_validation_evaluation(model_path,dataset_path,training_path, *,cache_dir=None)->(evaluation,points)`: canonical stdlib loaded predictor scores ALL validation rows, optionally using the verified compact cache. Actual predictions `{at,predicted,actual}` use existing convention and original unscaled return units. Include model kindcnn and zero_return metrics with row_count/first_decision_at/last_decision_at. Strict cohort/model source bindings and no2025.
- Every real training/inference test uses synthetic source artifacts until parent authorizes real fits. No extra MLP fits. No production code copied before observing failing tests.

## Owner C — report / workbench / trade audit

Own `reporting.py`, `experiment_reporting.py`, `report_template.html.in`, `workbench_data.py`, NEW `image_trade_audit.py`, own scoped report/workbench tests. No strategies/config/pipeline edits.

- Add explicit CNN branches, preserve Ridge/MLP versions and numeric explanations; CNN never gets fake four-feature additive contributions. Show actual input PNG and forecast with representation/2024-selection warnings.
- Extend MLP-only training history projections to CNN explicitly; existing dynamic filters/forecast tables should work. V4 summary fields below; no HTTP server endpoints/uploads/training controls.
- `build_trade_image_audit(backtest_path,model_path,dataset_path,predictions_path, *,limit=8)->dict`: deterministic first up-to8 executed CNN trades (not profitable selection), reconstruct original decision windows, regenerate actual PNG/image_key, re-predict frozen CNN, bind to saved prediction and actual ledger entry/exit/IDs. Safe self-contained image representation via dataURI; no arbitrary paths. If zero trades, explicit zero-trade audit/decision view, not a fabricated trade.
- `render_trade_image_audit(audit)->str`: escaped HTML, thumbnails/fill/forecast identities, outcome metadata outside PNG; audited count vs total. Parent appends to enriched report before body close and writes audit JSON separately. Bound size to fit32MiB total report.
- Summary v4 shape: existing AP2summary fields; config modelkindcnn; training_data_id=imageindex; source_training_data_id separate; training actualCNNmetadata; validation_forecast keys cnn/ridge/zero/zero_return plus cohort row/time fields; trading cnn/ridge/sma_timed. Parent produces new v4 comparison_signature declaring commoncohort/execution/representations; old v1-v3 signatures unchanged and not silently equivalent. Nonlinear model report provides no additive numerical contributions.

## Parent — integration and real acceptance

Own experiment_config.py, experiment_pipeline.py (earlyv4dispatch), NEW image_experiment_pipeline.py, strategies.py CNNexplicitloader/closure, experiment_variants.py explicitrejectCNNderive/search, own integration tests/config/docs. Reuse existing ledger/profile/helpers/registry/jobs, not another simulator. V4 accepts CNN only and fixed8/2/SMA4/8; canonical_model_settings legacy unchanged.

Build same frozen raw import/development/validation and numerical training artifacts, Ridge+SMAcontrols, then full image cohort/CNN/cache/eval/backtest/audit. Recursive manifest inventory for image artifacts; cache removed before manifest. Source-code fingerprint guard before/after runs remains.

Before real fits freeze production source, run full suites/reviews and create original-registry/live-service hash baseline. At most firstCNN+replay (2fits),20epochs each, wall1800s/CPU1200s/memory2048MiB including preprocessing/scoring/report. Recomputed Ridge controls allowed, no new MLP fit. Installed-wheel inference parity and browser comparison/audit required; user did not authorize AP7 Git/live publication automatically.
