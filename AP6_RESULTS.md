# AP6 results — causal chart-image data

## Final status and authorization

**Completed and internally published on 16 November 2026.** Andreas approved the narrower trusted-private-output-parent design, completion, commit/push and website updates after the earlier blocked reviews. Final independent review passed with no open security or logic findings under that explicit contract. Git completion is recorded in repository history and the final handoff note, not a self-referential commit hash here.

AP6 prepares a verified visual data basis, not an image model or profitability result. No new real fits/backtests or 2025 features/images/model metrics were run. AP7 is not approved. Contract: [AP6_IMAGE_DATA.md](AP6_IMAGE_DATA.md).

## Delivered and trust contract

- Deterministic 256x192 RGB PNGs with eight causal m15 candles, fixed geometry/colors, exact rational window-local scaling and stored DEFLATE. No text/axes/volume/targets/future pixels. Source-bound recipe and input-only identities; Decimal resource bounds checked before expansion/arithmetic.
- Complete sealed training/validation-row reconciliation, deterministic endpoint-retaining sampling, separate input/future-label evidence, identity/inventory/PNG regeneration and chronology/session/split/cutoff checks. Repeated candle evidence is globally consistent.
- Target regeneration preserves frozen precision-28 ROUND_HALF_EVEN subtraction followed by division, including exponent limits; historical training code is unchanged.
- CLI `images build/verify/preview`; coherent verified captures, escaped metadata, embedded original PNGs, restrictive CSP and explicitly separate collapsed targets. Preview is outside the bundle, including resolved `..` containment. Mobile timestamp wrapping is corrected. No browser training endpoints or server/queue code changes.

The output parent must already exist, belong to the current UID, and grant no group/other permissions (use0700). Checks occur before cache/output side effects and again before building/publication. Nonprivate0755/0777 and wrong-owner parents are rejected. The owner and same-UID programs are trusted; no root/hostile-same-UID sandbox is claimed.

A private0700 holder retains the verified child by descriptor. `renameat2(RENAME_NOREPLACE)` uses separate source-holder/output-parent descriptors. Rollback clears through the retained bundle descriptor, never recursively through the mutable outer destination; outer removal is identity-checked and nonrecursive. Holder/child acquisition states are tracked separately. Both EMFILE tests preserve the exact initiating exception and leave no staging residue. Exceptional later cleanup failure can leave owned remnants without masking the initiating error.

## Real cohort and reproducibility

Inputs: existing AP5 `development-dataset.json` and `training-data.json` from private `experiments/ap5-acceptance-v1/run-ff5344830dd5430a8bb42472d88f3df1/artifacts`. The original token `deuidxeur` is preserved, not normalized.

- Source dataset: `sha256:05bc5742f98258bf43b2d91434bfb114de7358fef6f94c84fadb9fd25dc950bb`.
- Source artifact: `sha256:f34aa5a5ab3e4c8cc08d07b81b201c2a267c8a623fdaa4436d66819a2f22c8cc`.
- Training-data ID: `sha256:126c9f3bd3213a3a5102327d3fcdf7fef0ce83a54e4c01a86dd6e33997c54e87`.
- All **18,528 training / 6,192 validation rows** rebuilt and reconciled.
- **32 training images (2021–2023), 32 validation images (2024), zero test images**. Eight available completed input candles and three separately stored future evidence candles per sample.
- Cold cache0 hits/64 misses; warm64 hits/0 misses. **All68 bundle files byte-identical**. Every real PNG independently checked for exact chunks/CRCs, RGB8 dimensions, decompression and zero filters.
- Image dataset ID: `sha256:e00546aa867179e57cc850ec85b5ab750cbacf186ec1a50e2eeaeaedefcacd3d`.
- Recipe ID: `sha256:c7c47deea72a2d69d7dea659b42d80c36096a2de22dfb395b1b82f241d5dbc7d`.

Private runtime evidence: `acceptance/ap6-real-v3/verification.json`; bundles `image-data/ap6-acceptance-v3-a`/`-b`; cache `image-cache/ap6-acceptance-v3`. All runtime-relative paths here are beneath `var/market-trader`, outside Git and the served directory.

## Installed package and tests

Separate non-editable `.venvs/market-trader-ap6-release`, without Torch/NumPy. Final wheel `releases/ap6-images-20261116-r3/market_trader-0.0.0-py3-none-any.whl`: **140,593 bytes**, SHA256 `afe73888179872ee5170b2d8bfb65fbbeea6c33271864960aad4e2be2817a8bb`.

All40 packaged application files equal checkout/installed bytes; `pip check` passed. Standalone verification passed with baseline/neural/storage imports blocked and experiment-source file reads denied by audit hook. The codec imports shared core training-data datatypes, not a trainer/reconstruction call. Installed export: same64 samples,64 hits,68 byte-identical files and byte-identical preview. Evidence: `acceptance/ap6-installed-wheel-r3.json`.

- Focused suite: **152 passed**; independent rerun **152 passed in130.12s**.
- Full Torch suite: **1,295 passed in359.94s**.
- Full no-Torch suite: **1,287 passed, 8 skips in361.60s**.
- Final review `deleg_8422cd21`: passed, empty security/logic finding lists. Independent renderer review also checked30 randomized windows with a separate pixel oracle.
- Nonblocking suggestion: additional explicit tests changing owner/permissions between checkpoints; implementation already rechecks them.

## Earlier failures and final resolution

Independent reviews found real ancestor/source-name races, rollback-name recursive deletion and EMFILE cleanup defects. They also caught Fraction-vs-frozen-Decimal target rounding and inconsistent repeated candles. Passing suites were not substituted for those failed probes. After two failed fix cycles, work stopped for Andreas's private-parent design decision. The final policy and descriptor-owned rollback/acquisition fixes have observed RED→GREEN regressions; prior r1/r2 packages are historical, not accepted final releases.

A provider HTTP400 interrupted one delegated fix; the parent completed it. Real-data validation caught uppercase synthetic fixture tokens differing from the actual lowercase source, corrected without rewriting original artifacts. One earlier parallel full-suite run exposed an existing one-second AP5 startup/timeout test under load; isolated and subsequent sequential full suites passed without changing AP5 code/deadlines. Final counts above are stable-source sequential results.

## Website publication

Accepted static HTML: `http://10.10.1.136:8086/reports/ap6-causal-chart-audit.html`, linked from `/reports`. **12,764,718 bytes**, SHA256 `84c3ad68bfcbcddb9d31d04c45490bd6d33f9b6499538ab2488d7467b791f330`.

The serving AP5 wheel, active-release pointer, unit/PID, bind/port and LAN/VPN ACLs remain unchanged; **no restart**. GET/HEAD and body equality passed; POST405, raw JSON404, encoded traversal404 and foreign Host403 remain enforced. Root, experiment page/API and health responses are byte-identical. Original research/registry/report hashes match `acceptance/ap6-before.json`, with exactly the authorized new HTML added.

Chromium followed the gallery link and decoded all64 live images pixel-equal to the independent decoder. Desktop1440px/mobile390/320px, wrapped headings, collapsed/keyboard-opened targets and scroll-contained tables passed with no app errors/external requests. Live desktop/mobile/target screenshots were inspected. Because plain HTTP lacks secure-context WebCrypto, the QA driver hashes browser-decoded pixel bytes in Node, without weakening browser security. A first preflight used an incorrect API route; corrected to `/api/experiments` before publication.

Read-only verification from Docker VM **10.10.1.20** fetched six routes successfully and matched full hashes including the new report. No VM/Dividend Screener changes. VPN rules are preserved; no fresh VPN-client route test is claimed.

Evidence: `acceptance/ap6-live-http.json`, `acceptance/ap6-live-lan.json`, `qa/ap6-live/verification.json` and screenshots. Undoing this static addition requires removing only its new HTML; no service rollback was exercised or needed.

## Limits

Linux `renameat2` and `/proc/self/fd` required. Internal consistency is not source authenticity against an owner rewriting all evidence. No power-loss transaction or hostile-owner sandbox. The64 samples are a bounded audit cohort, not a full AP7 dataset. Local scaling/quantization discard absolute price/precision while revealing path detail absent from aggregate features: neither identical information nor a strict superset. AP7 requires separate design/budget approval.