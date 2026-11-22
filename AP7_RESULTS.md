# AP7 — visual neural reference: final acceptance

**Status: technically accepted, internally published and visually accepted by Andreas on 22 November 2026. AP7 closure, including commit/push, was subsequently authorized.**

AP7 was requested after AP6 acceptance. On 22 November Andreas requested final technical acceptance followed by viewing the result on the website, authorizing the internal publication. After confirming the website, Andreas requested AP7 closure, authorizing commit and push of the accepted scope. This does not authorize a parameter search, reserved-2025 evaluation, live trading or additional fits. The verified wheel remains deployed unchanged; the closing Git operation does not regenerate accepted research artifacts.

## Delivered scope

- Full causal image cohort: **18,528 training examples (2021–2023), 6,192 development/validation examples (2024)**. All original feature/target rows reconciled; input-only index and compact uint16 block sums are separate from labels/source bindings. AP6 audit schemas/limits remain unchanged.
- Fixed **5,409-parameter CNN**: RGB CHW3×24×32, Conv3×3/4 channels, ReLU, AvgPool2, Flatten660, Linear8/ReLU/Linear1. CPU/float64, seed0, one thread, AdamW, batch128, at most20epochs.
- Actual-minimum validation-MSE checkpoint independently of patience/min_delta; target mean/scale fitted on training only. Strict bounded JSON weights and immutable stdlib inference, no Pickle/torch.load.
- Additive v4 runner, actual CNN strategy, fresh Ridge/SMA/zero controls, existing ledger, report, read-only Workbench and bounded ledger-bound causal image audit. Legacy v1–v3 semantics remain separate. CNN sweeps/variants are rejected rather than silently starting a search.

## Reviews and complete suites

Both independent implementation reviews in `deleg_25dab515` passed with no blocking security or logic findings. They covered pixel/cohort/cache safety, actual CNN optimization/checkpoint/reload/parity, strategies/registry/history/signatures, reporting and image/ledger bindings. Independent malformed-JSON, hidden nonfinite-before-ReLU and randomized pixel-parity probes passed.

The first foreground full-suite attempt hit a 420-second transport cutoff and supplies **no suite verdict**. The subsequent managed controller completed both suites sequentially, exit0:

- Core without Torch: **1,445 passed, 15 optional skips**, 359.695s controller wall time.
- Torch runtime: **1,460 passed**, 404.862s controller wall time.
- Production Python source and report template remained unchanged. Their bytes were rechecked against the deployed wheel; no source relaxation or additional fit was needed during final acceptance.

Private evidence: `acceptance/ap7-full-suites-20261118.json` and its separate logs.

## Actual first fit and replay

Registry: `/home/hermes/workspace/var/market-trader/experiments/ap7-acceptance-20261118-v1`.

- First: `run-49393a443991482eb9b58eb9681336f0`.
- Replay/child: `run-e28906643f89480b959b1ab2271a3261`.
- Both completed and **all 27 artifact files are byte-identical**. Run IDs/timestamps differ as intended; this is a reproducibility pair, not independent performance evidence.
- Exactly **two CNN fits and two Ridge control fits**, no MLP/search. No extra training during final acceptance or deployment.
- Both stopped after **6 epochs** by early stopping; actual minimum/checkpoint **epoch1**. Five subsequent epochs did not improve the minimum validation MSE.
- First complete pipeline: **675.335s wall, 669.362s CPU, 665.441MiB peak RSS**.
- Replay complete pipeline: **666.815s wall, 661.978s CPU, 663.938MiB peak RSS**.
- These timings include preprocessing, fitting, stdlib scoring, backtests and reports; optimizer-only duration was not separately measured. The host has two logical CPUs, but fitting was deliberately one-thread CPU, not two-core/GPU training.
- Each pipeline remained below wall1800s/CPU1200s/address-space2048MiB. Kernel CPU soft1200s/hard1201s allows termination cleanup; successful acceptance additionally requires measured CPU below1200s. Parent wall deadline and parent-death worker termination were configured. Existing AP5 queue limits were not changed.

All-validation framework/stdlib maximum absolute error: **4.336808689942018e-19** over6,192 rows, within the recorded fixed tolerances. Separate JSON-to-Torch reload parity measured the same maximum. This is measured evidence for this pinned runtime, not a universal platform guarantee.

## Actual 2024 development results

Forecast RMSE, basis points (lower is better):

- CNN **13.8623299248bp**.
- Fresh Ridge **13.8698176543bp**.
- Zero-return forecast **13.8603276761bp**.

The CNN is slightly worse than zero on RMSE. Small differences are not established predictive edge or statistical significance.

Reconciled fixed-unit ledger, points (not automatically euros):

| Control/model | Trades | Reference movement | Modeled costs | Net points |
|---|---:|---:|---:|---:|
| CNN |736|420.301|1619.200|**-1198.899**|
| Ridge |1784|-395.140|3924.800|**-4319.940**|
| SMA timed |1297|1062.135|2853.400|**-1791.265**|

The CNN is less negative than these trading controls, **not profitable**. Different trade counts/cost exposure matter. 2024 selected the checkpoint and is development evidence, not an independent final test. **2025 has no model inputs or evaluation metrics**; structural raw import is separate.

## Installed-wheel and browser acceptance

- Wheel `releases/ap7-cnn-20261118-r1/market_trader-0.0.0-py3-none-any.whl` SHA256 `f858ccc442e20dc86dc0cec62b747521a6eb9e3dd1f15aba6e338a6a611f1e17`.
- All **46 packaged application files** matched wheel/checkout and installed serving bytes. `pip check` passed. Serving and separate release environments have neither Torch nor NumPy installed.
- Separate installed release, outside checkout, under a framework import blocker: regenerated every actual validation PNG/window and obtained **6,192 bit-equal predictions (maximum error0)**. Independently recomputed train-only scaler, exact regenerated report bytes and identical audit JSON; original research unchanged. No fit/import of training frameworks.
- Actual Chromium desktop1440 and phones390/320: both real CNN details/histories, compatible first/replay, explicitly incompatible legacy AP5/CNN pair, model settings, report and first8chronological executed trades. No fabricated additive numerical attributions.
- Each browser pass captured **43 screenshots** and **48 decoded PNG pixel checks** across both runs/viewports, with zero runtime/app errors or external app requests. Parent inspected representative actual local and live screenshots.
- Important UI limitation: tables deliberately scroll inside narrow containers, and some very narrow comparison/report views expand the mobile layout viewport (recorded362/413CSSpx at a requested320px). The generic automated `scrollWidth <= innerWidth` check is therefore **not proof of perfect320px fitting**. Core content was visually inspected; further responsive polish is non-blocking, not claimed complete. The image audit is technical/metadata-heavy, not a compact attribution/explanation UI.

## Internal publication and readback

Published service `http://10.10.1.136:8086/`, source alias `ap7`. Workbench now exposes **13 verified runs**, preserving all11old run projections and adding the two completed CNN runs. Report gallery includes:

`/reports/ap7-visual-2024-reference.html`

Report SHA256: `e0eb3f8fcfc726414b3e8b91ee3840e9c961aeca3f4296e21c9cbf493930e5c6`.

- Verified old installed release against its rollback wheel before installation; backed up exact unit/pointer/wheel and prepared `rollback.sh` at `releases/ap7-deployment-backup-20261122`. Rollback was prepared, **not exercised**, because deployment passed.
- Installed accepted wheel only; appended the AP7 registry to the actual unit. Bind/port, LAN`10.10.1.0/24`, VPN`10.6.0.0/24`, read-only and systemd resource restrictions remained unchanged. No public exposure, browser training controls or additional data/weights serving.
- Service active/enabled, release pointer read back, package byte identity and no-framework serving checked. Initial post-deploy memory peak57,438,208bytes was below the unchanged256MiB cap.
- All **1,334 original baseline files** preserved during publication. Live QA independently checked **285 registry files** unchanged across all four aliases.
- Repeated actual HTTP/API/report/compare/security and43-screenshot browser/pixel acceptance against the live service, not only loopback. Successful full-body equality for **11 routes** from the approved second LAN host`10.10.1.20`, including overview/API, both reports/details and comparisons.
- Mutation methods, foreign Host, traversal and private JSON requests denied. A preserved VPN ACL is not a new VPN-client routing test; Andreas's own client review remains separate.

## Provenance and remaining boundaries

Code identity: `sha256:5f22731161cf538c26481366c5171eb1669b9344eb74882bf367ec61f77ba2ff`.
Report-template SHA256: `e8417019ff6479fef122d8205a12f662402fe756934835524784bc0a5f415a82`.

Private evidence lives under `/home/hermes/workspace/var/market-trader`:

- `acceptance/ap7-real-20261118-v1/verification.json`, `installed-core-verification.json`.
- `qa/ap7-final-20261122/verification.json` (isolated installed service).
- `acceptance/ap7-live-deployment-20261122.json` and backup/deployment record.
- `qa/ap7-live-20261122/verification.json`, browser proof and screenshots.

No blocking model/data/integration/security issue remains under the approved current-UID/private-parent trust model; it is not a sandbox against hostile same-UID rewrites. Generic report model autodetection retains its legacy initial JSON read; optional hardening remains non-blocking. Pixel inputs are not information-equivalent to tabular features: local scaling, rasterization and8×8averaging lose price/detail information. No optimization, significance/robustness claim or profitability guarantee follows from this technical milestone. AP8 and further fits require their own scope decision; the separately authorized Git closure is limited to this accepted AP7 scope.
