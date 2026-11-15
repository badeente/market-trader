# Internal report viewer

The viewer is a small, read-only LAN/VPN service for generated HTML reports.
No desktop, GPU, browser on the server, database, or broker connection is needed.
Open the site in a browser on your own computer or phone.

## Scope and limitations

- Publish only generated reports to a **dedicated flat directory**. Never use
  the repository root, a home directory, datasets directory, or secrets directory.
- Only non-hidden `.html` files are listed and served. JSON, raw CSV, subdirectories,
  symlinks and path traversal are not served. Reports are limited to 32 MiB each.
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
