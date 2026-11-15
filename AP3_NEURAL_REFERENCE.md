# AP3 – first tabular neural reference (approved scope)

**Status: implemented, actually trained/replayed, independently reviewed and internally published. Authoritative outcomes and limitations: [AP3_RESULTS.md](AP3_RESULTS.md).**

Andreas approved proceeding with the preceding small-MLP proposal. Base: AP2 commit `1483e32`. No NN tuning/search, image model, trading endpoint or AP4 workbench redesign.

## Frozen first model

- Same four existing causal features, eight completed 15-minute bars; 30-minute Open-to-Open target.
- MLP `4 -> 16 -> 8 -> 1`, ReLU hidden activations, linear output.
- AdamW, learning rate0.001, weight decay0.0001, batch128, maximum100epochs.
- MSE on standardized targets; input and target means/scales fitted ONLY on2021–2023 training. Inverse target transform for forecasts, errors and trading.
- Early stopping on2024 validation, patience10; minimum improvement1e-6 in standardized validation MSE. Store best weights, epoch history, actual stopping reason and chosen epoch. This makes2024 development/selection data, not an independent final test.
- Seed0 first run, CPU single-thread float64 and deterministic training. No random temporal split; mini-batch permutation may use only the training rows. Additional seeds are not an automatic search in this scope.
- Source quality audit may include2025, but physically exclude2025 before feature/model preparation. No reserved2025 metrics.
- Same cost, eligibility, one-unit size and timed exits as Ridge. Recompute Ridge in the same current environment for comparison.

## Compatibility spike actually exercised

Separate `/home/hermes/workspace/.venvs/market-trader-ap3`, CPython3.14.9. Official CPU index provided torch2.15.0+cpu cp314 Linux wheel; NumPy2.5.4 installed to resolve optional-NumPy startup warning. Real float64 forward/backward AdamW step lowered loss;225 trainable parameters, CPU-only. pipcheck passed. Existing AP1/AP2 environments and web service unchanged. Record exact transitive versions and commands before final acceptance.

## Integration boundaries

- Preserve strict AP2 version1/Ridge config and canonical snapshots. Introduce version2 requiring explicit model table for neural configuration, rather than silently expanding old v1.
- Reuse existing importer, supervised dataset checks, execution ledger, registry, replay and checksummed manifest. No second accounting implementation.
- Store scalers and exact weights in a checksummed JSON neural artifact. Loading/inference must not use pickle/torch.load and should not require Torch; training imports Torch lazily.
- New learned strategy `mlp_long` uses exactly the existing Ridge entry/exit/session eligibility logic with its verified frozen predictor.
- Report neural forecast errors alongside Ridge and zero; neural trading alongside Ridge/control under identical conditions, plus training curves/history. No invented linear feature contributions for a nonlinear model.
- Keep core stdlib-only; neural requirements and environment remain explicit and separate. No PyTorch import in baseline CLI/help/site service.

## Acceptance

Real bounded training, save/reload equivalence, validation predictions, registered run/replay, existing/new tests, independent review, readable internal report, exact source/config/code/dependency provenance. Model performance may disappoint; do not tune or replace results to manufacture success. No AP3 result exists at the time this contract is written.
