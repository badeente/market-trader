# AP8A — bounded development robustness results

Status as of 26 November 2026: **AP8A technically accepted and internally published** after Andreas separately approved readable renderer integration, publication and Git completion in Discord. The original immutable study/report and its earlier acceptance proofs remain unchanged. A new report from the CSS-updated installed renderer is live; publication evidence is recorded below. This is not completion of the broader AP8 roadmap or the user's own visual acceptance. No new real fits, rolling study or reserved-test evaluation were performed during publication.

## Scope and provenance

See `AP8_ROBUSTNESS.md` for the predeclared contract. This section tests fixed CNN and MLP policies with seeds 0/1/2, calendar/time-of-day slices and three modeled cost levels. Seed 0 is reused from the accepted AP7 CNN and AP5 MLP references; only seeds 1/2 were fitted for each family. No best-seed deployment or architecture search is authorized.

- Training: 2021–2023, 18,528 eligible rows; validation: 2024, 6,192 rows per model.
- 2024 selects checkpoints and therefore supplies development diagnostics, **not independent final-test evidence**. 2025 remains closed to model inputs/evaluation.
- Same causal decision/target rows; CNN images and MLP numeric features are not information-equivalent.
- Eight historical m15 candles, horizon two, existing long-only fixed-unit rule and execution engine. No new short/TP/SL policy or architecture.
- Seeds change both weight initialization and minibatch shuffle order; this does not isolate initialization alone.
- Frozen fitting code: `sha256:b100ddd35e0a44e4e1b26116575ee3f794632e51c62229e0c4fd8d506ecc6568`.
- Verified reference pair: `sha256:506957ad482b3056aa6393df38181d8918155ac68fca0a809de40b807c1d3bcc`.
- Base Git revision: `89b89c9dd7ec4e4ee291fbe5ca39ecee07b7ca85`; AP8 remains uncommitted. Reused references retain their original code/runtime provenance.

## Forecast and trading outcomes

MAE/RMSE below are return errors in basis points, rounded to six decimals. Smaller errors are preferable; they do not measure profitability or directional accuracy. The common zero-return RMSE is approximately **13.860328 bp**.

Net outcomes are simulated index points at fixed quantity one, **not euros or a portfolio return**. Half/baseline/double multiply spread, slippage and commission by 0.5/1/2; all ledgers are replayed through the existing engine, not estimated by subtracting a guessed fee.

| Model | Seed | MAE bp | RMSE bp | Half-cost net points | Baseline net points | Double-cost net points | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN, reused | 0 | 10.098540 | 13.862330 | -389.2990 | -1198.899 | -2818.099 | 736 |
| CNN, new | 1 | 10.083056 | 13.856241 | 92.8330 | -1820.067 | -5645.867 | 1739 |
| CNN, new | 2 | 10.084721 | 13.858733 | -685.5000 | -2570.900 | -6341.700 | 1714 |
| MLP, reused | 0 | 10.095152 | 13.863204 | -1030.6920 | -2473.892 | -5360.292 | 1312 |
| MLP, new | 1 | 10.091430 | 13.863211 | -2404.5110 | -4362.511 | -8278.511 | 1780 |
| MLP, new | 2 | 10.088538 | 13.867266 | -2084.6500 | -4261.550 | -8615.350 | 1979 |

All six baseline-cost results are negative. The new CNN errors are marginally below the zero-return comparator, but the corresponding baseline trading results remain negative. The only positive half-cost result is CNN seed 1; its baseline and double-cost results are negative. These results provide **no demonstrated robust trading advantage**. Three seeds and checkpoint-selected development data cannot establish statistical significance, independent generalization or a superior model family.

Actual epochs / selected checkpoint: CNN seeds 0/1/2 = 6/1, 12/7, 9/4; MLP seeds 0/1/2 = 20/20, 20/10, 20/16. Checkpoints use the actual minimum validation loss, not an assumed final epoch. The standalone report retains every seed, complete histories, all three cost scenarios, monthly trade outcomes, twelve forecast months, four quarters and three Europe/Berlin decision-time bins. Calendar/time bins are not learned market regimes.

## Real execution and budgets

Exactly four durable invocation claims were consumed. No automatic retry, extra fit or seed-0 retraining occurred. Each invocation stayed within wall 1,800 seconds, successful measured CPU below 1,200 seconds and 2 GiB address-space limits.

| Slot | Run ID | Total worker wall seconds | Measured CPU seconds |
|---|---|---:|---:|
| CNN seed 1 | `run-35ab0d0ebbac4b0d9703aa0ad8938db9` | 463.80593618098646 | 461.936539 |
| CNN seed 2 | `run-12f503f4123946a281f18bfe1e10d302` | 442.2523781340569 | 440.547078 |
| MLP seed 1 | `run-7d39368f46a44a6a94019e1d778eb34f` | 34.312676963047124 | 32.890606999999996 |
| MLP seed 2 | `run-56d2cb215f334957b9b0d9aa9a12fc69` | 34.77575161505956 | 33.526735 |

Total worker wall time is 975.1467428931501 seconds. This includes worker preparation/evaluation, is **not optimizer-only timing**, and excludes the controller's reference diagnostics and final study reconstruction. Maximum recorded worker RSS was 595.16796875 MiB; RSS is distinct from the enforced virtual-address-space limit. Model settings specify CPU/float64 and one training thread.

## Verified acceptance evidence

Private evidence paths below are relative to `/home/hermes/workspace/var/market-trader/`; do not publish registry/model/source directories.

Final local acceptance receipt: `acceptance/ap8a-local-acceptance-20261126.json`, binding the seven acceptance evidence receipts and both report hashes. This receipt explicitly excludes live publication, Git operations, user visual acceptance and completion of the broader AP8 roadmap.

- Frozen full suites ran sequentially: **core 1,647 passed / 22 optional skips**, **Torch 1,669 passed / zero skips**. `acceptance/ap8-full-suites-20261122.json` confirms unchanged source and successful exit codes.
- Numerical follow-up and final independent source/envelope review passed. The five earlier semantic-binding findings were fixed and rechecked using complete positive fixtures and checksum-valid adversarial mutations. Evidence: `acceptance/ap8-numeric-review-20261122.json`, `acceptance/ap8-final-review-20261122.json`.
- The fit controller verified the source cohort, fully replayed the two old reference ledgers, ran four bounded workers, canonically reconstructed new forecasts and all cost ledgers, and preserved old research/site state. Evidence: `acceptance/ap8-real-20261122-v1/verification.json`.
- Separately installed wheel: SHA256 `4a152dee5e2771f9fa00fa2d68c949c7da0cb21c02b32e8de8b9471b64cc1ec0`; all **50 application files** equal checkout/wheel bytes. Torch/NumPy are absent and training-framework imports are actively blocked.
- Installed-core verification checked all four new models and every validation row, independently recomputed train-only scalers, replayed canonical forecasts and every cost ledger, and reproduced the report **byte-for-byte**, with **zero new fits** and unchanged research. Only disclosed analysis-runtime Git fields differ because the installed environment is outside the checkout. Evidence: `acceptance/ap8-real-20261122-v1/installed-core-verification.json`.
- Actual original-report browser QA passed **54 checks with 18 captures**, reconciling all six runs/tables against the study JSON, warnings, keyboard details, requested/document widths, no hidden worst-seed rows and no browser errors. Evidence: `qa/ap8-real-browser-20261126/verification.json`.
- On 26 November, a fresh read-only preservation check found **1,395 original files and five HTTP bodies unchanged**; active release and service unit unchanged, service active, exactly four fit claims. Evidence: `acceptance/ap8-preservation-20261126.json`.
- Independent read-only review of the actual completed artifacts **passed 274 checks with no blockers**, separately from the earlier source review. It verified claims/ownership/resources, all six saved model/scaler/history envelopes, every forecast error/calendar/time-bin projection, all 18 scenario totals and 216 monthly projections, fill/account/drawdown reconciliation, and all 1,395 baseline hashes. It performed no fits/full inference sweeps or engine replays; nine MLP sample predictions matched exactly. The parent's separate installed-core check supplies the full four-model inference/ledger reconstruction. Evidence: `qa/ap8-actual-independent-review-20261126.json`.

## Presentation limitation and separate readable preview

The canonical report's desktop view is readable, but its original 320px tables wrap headings and numbers into unreadable fragments. Geometry/DOM success alone is **not mobile visual acceptance**.

A separately derived presentation preview uses wider, locally horizontally scrollable tables. It changes CSS/classes plus an origin note only, not model settings or numerical values. It is stored **outside** the immutable study and was checked against the same original study JSON: **54 browser checks, 18 captures**, with parent visual inspection of the actual 320px family and 390px cost views. Long IDs/captions remain dense; wide tables require horizontal scrolling.

- Canonical report: `studies/ap8a-development-20261122-v1/report.html`, SHA256 `62d2c8025266850be38bcce4653c3caf912715b4de7124c3130043aa927e3d5f`.
- Readable derived preview: `qa/ap8-readable-real-20261126/report.html`, SHA256 `a7073cffaffa672a608c49db51dee9551a769b6d1e8d9a34703876568718adab`.
- Preview origin receipt: `qa/ap8-readable-real-20261126/report.origin.json`; QA: `qa/ap8-readable-real-browser-20261126/verification.json`.

The canonical byte-parity proof and the derived preview's hash are separate. Neither local artifact is an AP8 live website publication. Existing AP7 remains deployed unchanged.

## Internal publication — 26 November 2026

Andreas approved the closing scope in Discord message `1575610167943372942`: readable production-renderer styling, internal publication, commit and push; no further real training or reserved-period access. Earlier statements above about local-only artifacts and the pre-commit base describe the original acceptance snapshot, not this later publication.

- Live report: `http://10.10.1.136:8086/reports/ap8a-development-robustness-20261126.html`, also linked from `/reports`. AP8A is a standalone study report, not a new source in the AP2–AP7 run-selection API.
- Integrated readable table styling via strict RED/GREEN tests. The regular renderer adds no preview-origin note. New report SHA256: `3745fffc18ca6f803242435572515bf48678ecc8a761b8326b1b0730ec4db072`; all **10,545 non-style DOM events** equal the original canonical report after ignoring CSS classes. The original canonical report/hash and its original byte-parity proof remain unchanged.
- Separately installed publication wheel SHA256: `65fac3bc08cedd6f98b5a83723a61f3c61af5cd92b1532d150ec0d62f6c096de`; all **50 application files** match wheel/checkout/installed bytes; Torch/NumPy absent. Rendering reads the completed study JSON only: no new real fits, inference sweep, study rebuild or reserved-period access.
- Fresh sequential complete suites: **1,649 passed / 22 optional skips** in core; **1,671 passed / zero skips** with Torch; each reports 65 subtests. Source/test inventory remained fixed throughout. An initial foreground suite attempt was interrupted by the tool's 420-second deadline and is not counted as a pass; the complete background rerun passed.
- Independent review of the complete 3,329-line, 15-file source/docs/test diff: no security or logic blockers; static scan zero hits. Documentation suggestions are incorporated here.
- New installed-renderer browser QA and subsequent **live HTTP browser QA each passed 54 checks with 18 captures** at 1440/390/320px. Actual live 320px family and 390px cost screenshots were inspected. Wide tables intentionally require local horizontal scrolling; long identifiers remain dense. The first browser startup attempt timed out; the subsequent bounded retry and live run passed.
- Live report body equals the generated bytes; gallery href, health, overview/API, existing report bytes and denial of raw/traversal/foreign-Host/POST requests checked. A second LAN host (`10.10.1.20`) returned HTTP 200 and the identical 83,031-byte report hash. No fresh VPN-client test was performed.
- **638 pre-existing study/registry/report files** unchanged in the publication inventory; service unit, PID and `active-release.txt` unchanged. Static-only publication needed neither a serving-wheel replacement nor a service restart; the existing AP7 viewer runtime continues serving all existing collections. Research implementation is delivered in Git and the separately installed publication wheel, not silently installed into the viewer.
- Evidence: `acceptance/ap8a-publication-20261126/` (`before.json`, `suites.json`, `review.json`, `package.json`, `semantic-parity.json`, `publication.json`); browser evidence: `qa/ap8a-production-renderer-retry-20261126/verification.json` and `qa/ap8a-live-20261126/verification.json`. Closing Git identity/push verification is recorded separately in the private publication directory, avoiding self-referential commit hashes.

## Remaining decisions

The user's own visual review remains outstanding. Further fits/rolling windows and any 2025 final evaluation require separate scope/approval; broader AP8 remains open. No profitable seed has been selected for deployment. Historical absence of hidden executions cannot be proven beyond the approved code path, receipts and inspected artifact inventory.
