# AP7 — first visual neural reference (frozen contract)

Technical acceptance and internal publication completed on 22 November 2026; actual results and remaining non-blocking limits: [AP7_RESULTS.md](AP7_RESULTS.md).

## Authorization and boundaries

Andreas requested AP7 after accepting AP6. Implement and locally exercise a first small image network in the existing experiment/registry/backtest/comparison workflow. The original request was not automatic commit/push/live-publication approval. On 22 November 2026 Andreas separately requested final technical acceptance followed by viewing the result on the website, authorizing internal publication of accepted outputs. Git closure was separately authorized after Andreas confirmed the website; further research remains a separate decision. Existing live AP6/AP5 content and accepted registries were preserved; AP7 outputs use a separate registry.

No new simulator/accounting. Preserve all frozen AP1/AP2 v1/AP3 v2/AP5 v3 semantics. Add a versioned visual model/config rather than silently changing historical snapshots. Fit2021–2023;2024 development/validation includes checkpoint selection; physically exclude2025 before any model input/feature work. No test-year metrics, parameter search, profit selection, live trading, external model APIs or new paid services.

## Representation and resource envelope

Use the AP6 original recipe/renderer: exactly eight available completed m15 candles,256x192RGB, input-window-only scale, no axes/text/labels/volume/future pixels. Preserve the literal source token `deuidxeur` and the existing next-open-to-horizon-open target.

The accepted64 AP6 audit images are NOT the training dataset. AP7 must use all eligible causal training/validation decisions, reconstruct and reconcile the sealed rows, and bind image inputs to the original recipe/window identities. Do not raise AP6's1–64-per-split audit-bundle limits or reuse only that sampled cohort for learning.

Host discovery:2logical CPUs, approximately4GiB RAM,9.5GiB free filesystem space at kickoff. Do not materialize/replay tens of thousands of uncompressed PNGs on this filesystem. Prefer a distinct full-cohort compact index plus transient bounded preprocessing. Keep model inputs/labels/provenance outside Git and the served report directory, with the approved current-owner/private0700 parent policy.

Frozen input preprocessing: decode the original RGB PNG, exact8x8 block-average pixels to32x24RGB, scale by fixed255. No validation-fitted image transforms or global price statistics. Explicitly disclose that this additional pixel reduction discards wick/detail information; aligned windows/targets do not imply equivalent information to tabular features or original full-resolution PNGs. This design was frozen after the isolated synthetic CPU feasibility probe; actual real-data acceptance is recorded in AP7_RESULTS.md.

## Frozen model and measured feasibility

The private synthetic probe `/home/hermes/.hermes/cache/scratch/ap7-feasibility-onz57f6r/results.json` was read back. Actual architecture: RGB CHW[3,24,32] → valid Conv3x3[4,22,30] → ReLU → AvgPool2[4,11,15] → Flatten660 → Linear8 → ReLU → Linear1, **5,409 parameters**. Two30-minibatch repeats had identical weights; JSON reload was bit-equal in Torch; stdlib inference max error1.67e-16 on18 probes and an additional eight no-framework fixtures passed. This is synthetic evidence, not a real-fit result or universal tolerance guarantee.

Frozen policy: float64 CPU/one thread, deterministic seed0, AdamW learning_rate0.001, weight_decay0.0001, betas[0.9,0.999], eps1e-8, amsgrad=False, foreach=False, fused=False, maximize=False; batch128, max20epochs, patience5, min_delta1e-6. Select the actual minimum validation-MSE checkpoint independently of the patience/min_delta counter. Restore global RNG/thread/determinism/default-dtype state after a library call. Target scaler fits2021–2023 only.

Actual cached model inputs are little-endian uint16 channel block sums (maximum16320), in CHW order; convert each minibatch to float64 using sum/64/255. This preserves subpixel averages rather than rounding to uint8 and requires approximately108.6MiB for the accepted full cohort, not434.5MiB float64. Allow an optional exact NumPy preprocessing accelerator only when explicitly used by the training runtime; core decoding/inference remains stdlib-only and must agree.

Real budget: at most **two CNN fits** (first plus replay), each wall1800s/CPU1200s/address-space2048MiB, maximum20epochs. This envelope includes preprocessing, reload scoring, backtest and report, not just optimizer time. Ridge/SMA/zero controls are recomputed by the existing pipeline (two cheap ridge reference fits allowed); no additional MLP fit or search. Keep preprocessing/cache/index input-only and bounded; no multi-GB PNG corpus. If limits are insufficient, report the actual blocker before requesting a changed budget.

A v4 CNN-only experiment branch will preserve v1–v3 behavior. Canonical v4 settings are fixed, not a new tuning surface; CNN derive/sweep is explicitly unsupported initially. Summary/report compare CNN against freshly recomputed Ridge/zero forecasts and Ridge/SMA ledger controls on identical eligible rows/costs. A new explicit v4 comparison policy will not silently declare legacy signatures equivalent.

## Initial candidate notes (superseded by frozen policy above)

Small convolutional network:3input channels, one valid3x3 convolution to4channels, ReLU,2x2 average pooling, flattened features, hidden dense8/ReLU, scalar linear output. Exact shapes and parameter count must be computed by the real framework, not assumed. Target standardization fitted only on2021–2023 labels and inverted for forecasts/trading. Fixed input normalization, no learned image scaler.

Candidate training settings: existing isolated CPU Torch stack, float64, one thread, deterministic algorithms and explicit seed0; AdamW learning rate0.001, weight decay0.0001, batch128, maximum20epochs, patience5 and min_delta0.000001. Validation standardized MSE chooses the best checkpoint; no profitability-ranking or tuning. Final architecture/settings/budgets must be frozen after measured feasibility and before any real fit. No real fits have run at this contract revision.

Plan a strictly bounded first fit and one replay (two real CNN fits maximum unless a new budget decision is made). Account separately for any inexpensive ridge reference refit required by the existing pipeline. Numerical references may be reused only through verified comparable artifacts; never label an old run a new fit. Synthetic optimizer steps and tests do not establish real-data acceptance.

## Safe model and integration requirements

Use strict versioned JSON weights/metadata, explicit tensor shapes and finite-number/bool/type validation; no Pickle, torch.load or arbitrary code. Reload/inference must work without Torch/NumPy in the serving/core runtime, with measured parity against actual framework output. Seed alone is not a cross-platform bit-identity guarantee; verify a real replay under the fixed runtime.

Integrate a visual model kind into the existing runner, registry, forecasts, long-only strategy, identical execution costs/ledger, and read-only comparison UI. Preserve numeric references/zero forecast and distinguish forecast quality from trading P&L. Comparison must explicitly disclose representation differences and refuse genuinely incompatible periods/targets/cohorts/costs.

Provide a bounded human trade audit showing the actual historical input PNG, timestamp, model forecast and corresponding real ledger decision/trade. Keep later labels/outcomes clearly outside input pixels. If the model executes no trades, report that truthfully and show evaluated decisions instead of fabricating trades. Do not embed thousands of images into an oversized report or serve arbitrary private files.

## Acceptance gates

1. TDD for decoder/preprocessing, full-cohort causal alignment, CNN training/JSON reload/inference and protocol integration. Graphify-first discovery and source verification.
2. Stable source before provenance-sensitive fits/tests; bounded synthetic runtime smoke first.
3. Actual first fit, best-checkpoint history, all validation forecasts, reconciled existing-ledger backtest and explicit numeric/zero-reference comparison.
4. Real replay with recorded equality/tolerance and unchanged accepted data/registries/site.
5. Separately installed wheel outside checkout, training-free reload/inference, actual desktop/mobile comparison and image/forecast trade audit.
6. Complete sequential suites with and without Torch, independent security/logic reviews, documented actual results and remaining limitations. No performance/profitability claims from an implementation success.

## Primary framework references

PyTorch official Conv2d, AvgPool2d and deterministic-algorithms/reproducibility documentation were consulted. The installed runtime and measured execution, not the moving documentation version or an assumed API, determine this experiment's accepted environment.
