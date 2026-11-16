# Internal report viewer

The viewer is a small, read-only LAN/VPN service for generated HTML reports and,
when explicitly configured, the AP4 experiment workbench.
No desktop, GPU, database, broker connection or training framework is needed in
the serving runtime. Open the site in a browser on your own computer or phone.

## Scope and limitations

- Publish only generated reports to a **dedicated flat directory**. Never use
  the repository root, a home directory, datasets directory, or secrets directory.
- Only non-hidden `.html` files are listed and served by the static report routes.
  Raw artifact JSON, CSV, subdirectories, symlinks and path traversal are not
  served as files. AP4 has separate bounded JSON projections, described below.
  Reports are limited to 32 MiB each.
  The serving directory is pinned by an open directory descriptor so replacing
  its pathname cannot redirect requests to another directory.
- GET and HEAD are supported; mutation methods are rejected. There is no upload,
  trading endpoint, training endpoint or remote shell.
- Only explicitly configured loopback/private IPv4 bind addresses are accepted.
  The default bind and client allowlist permit loopback only. Additional private
  client subnets must be explicitly allowed. Host headers are restricted to the
  bound IP, with or without the actual port. Forwarded headers are not trusted.
  Disallowed source addresses are dropped before worker creation or HTTP parsing.
- This is an **internal MVP, not an Internet-facing production server**. It uses
  Python's HTTP server machinery, whose upstream documentation warns against
  production use. Do not port-forward it, expose it publicly or put untrusted
  clients on its allowlist. It has no login and uses plain HTTP: anyone in an
  allowed subnet can read the published reports. TLS/authentication through a
  suitable reverse proxy is a separate deployment task if the trust boundary grows.
- It does not configure a VPN or router/firewall. A VPN must already route to the
  server; its actual client source subnet needs explicit approval and configuration.
- Treat report HTML as trusted generated application content. Do not publish
  downloaded third-party HTML. Reports can contain financial information.

## Local use

After installing the package:

```sh
market-trader serve-reports --directory /path/to/published-reports
```

The loopback default is port 8086. For the approved Hermes LAN deployment:

```sh
market-trader serve-reports \
  --directory /home/hermes/workspace/var/market-trader/reports \
  --bind 10.10.1.136 --port 8086 --allow-network 10.10.1.0/24
```

Browser address: `http://10.10.1.136:8086/`.
A new report appears on the overview when the page is refreshed.
`/healthz` returns `ok` for an allowed client and Host header.

## AP4: opt-in experiment workbench

Published on the existing Hermes service on 16 November 2026 after separate
Git/deployment approval; live HTTP, a second LAN host and actual desktop/mobile
browser flows passed. See [AP4_RESULTS.md](AP4_RESULTS.md) for the accepted wheel,
proofs and rollback backup. Root now opens the workbench; `/reports` opens the
original gallery. Existing individual report URLs are unchanged.

The approved scope and data/API contract are in [AP4_WORKBENCH.md](AP4_WORKBENCH.md).
Add one or more explicit private registry mappings to enable it:

```sh
market-trader serve-reports \
  --directory /home/hermes/workspace/var/market-trader/reports \
  --bind 10.10.1.136 --port 8086 \
  --allow-network 10.10.1.0/24 \
  --registry ap2=/home/hermes/workspace/var/market-trader/experiments/ap2-acceptance-v2 \
  --registry ap3=/home/hermes/workspace/var/market-trader/experiments/ap3-acceptance-v2
```

Aliases are exact lowercase identifiers; paths must be absolute and unpadded.
The browser submits source aliases and existing run IDs, never paths. Registries
stay outside the published report directory. The existing approved live VPN
allowlist is preserved during deployment; do not copy an example subnet into a
new installation without approval.

- `/` and `/experiments`: overview, filters and selection of 2–4 valid completed
  runs. Failure/integrity states are visible but not selectable.
- `/experiments?run=<source:run-id>`: saved configuration/identities, forecast
  baselines, trading costs/results, MLP history and account curves. All query
  values must be URL-encoded when constructing links.
- `/experiments?compare=<key1,key2>`: differences and comparison, with a prominent
  warning when stored data/cohort/execution/code/runtime signatures differ.
- `/reports`: original static HTML archive; `/reports/<filename>.html` stays
  unchanged. Without `--registry`, the original gallery remains at `/`.
- GET/HEAD `/api/experiments`, `/api/experiment?run=...` and
  `/api/compare?run=...&run=...`: bounded JSON projections, not raw artifact files.
- `/experiment-report?run=...`: only the verified fixed `report.html` for a valid
  completed registered run. There is no general artifact-download endpoint.

Registry manifests and checksums are verified before projections are cached.
Changes to record/config/manifest/artifact filesystem fingerprints invalidate
cached entries. This is consistency verification within a trusted local registry,
not a cryptographic signature against the registry's owner. Do not grant
untrusted parties write access. Initial reads hash the underlying artifacts and
may be slower; this is not a job scheduler or a large-scale database index.

AP4 reads saved results only: no model inference/fitting, backtests, queue,
start/stop, parameter writes, uploads, arbitrary commands, raw CSV, model weights,
private logs or credential files are exposed. The serving process does not need
Torch, NumPy or Node. Node is a **test-tool prerequisite** for executable frontend
contract tests, not a production dependency.

Forecast errors are in basis points; trading metrics and account changes are in
points per fixed unit, not euros. Training/validation MSE is in standardized
training-target units, not trading return. Account curves may be downsampled;
the display identifies this and totals still come from the complete saved data.
2024 is development/selection data and 2025 remains reserved. An incompatible
comparison is descriptive only, never a fair ranking or profit recommendation.

## AP5 publication

The accepted AP5-capable wheel was published internally on 16 November 2026
after separate authorization. The service keeps AP2/AP3 sources and adds
`ap5=/home/hermes/workspace/var/market-trader/experiments/ap5-acceptance-v1`.
Filter the overview by source `ap5` for the declared learning-rate variants and
replay. The copied reference parent is provenance, not an additional fit.

The HTTP surface remains read-only. AP5 training jobs are local CLI operations
in the isolated training environment, not website buttons or server commands.
The serving environment still requires neither Torch nor NumPy. Existing
LAN/VPN rules and report URLs are preserved; rollback and live verification
are documented in [AP5_RESULTS.md](AP5_RESULTS.md).

## Generate and publish

Generate backtest JSON outside the served directory, then render a new report
into the served directory. The report CLI validates its saved input and publishes
without overwriting an existing file.

```sh
market-trader backtest --config configs/backtest-example.toml > /private/path/run.json
market-trader report --backtest /private/path/run.json \
  --output /home/hermes/workspace/var/market-trader/reports/synthetic-single-long.html
```

Use a new filename for each run. Use `--compare` for a compatible second run.
Reports named `synthetic-*` demonstrate artificial fixtures and must not be
interpreted as real market performance. Generating HTML does not itself prove
profitability or the truth of externally supplied market data.

## Persistent Hermes deployment

The example user-systemd unit is `deploy/market-trader-reports.service`.
It is specific to Hermes' approved IP, subnet and workspace paths; adapt it before
using it elsewhere. The runtime uses a separately installed wheel in
`/home/hermes/workspace/.venvs/market-trader-web`, not the mutable checkout.
Report HTML lives under `var/market-trader/reports`; source JSON remains separate.
No privileged system service or sudo is required.

```sh
systemctl --user daemon-reload
systemctl --user enable --now market-trader-reports.service
systemctl --user status market-trader-reports.service
journalctl --user -u market-trader-reports.service -n 50 --no-pager
```

Install the reviewed wheel and copy the unit to `~/.config/systemd/user/` before
running these commands. A user service needs lingering for boot without login;
check `loginctl show-user "$USER" -p Linger`. Enabling lingering may require an
administrator. The unit applies restart and resource limits, but runs under the
current user, not a separate OS identity. It is not a filesystem sandbox.

Stop access with `systemctl --user disable --now market-trader-reports.service`.
A server-side HTTP success only proves service access from that network path;
verify the browser URL from the client as well, especially over a VPN.
