"""Small read-only HTTP viewer for a dedicated directory of generated reports.

This is an internal LAN/VPN viewer, not a public hosting service. It never
executes backtests or exposes raw inputs, and serves only flat HTML files.
"""

from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Network, ip_address, ip_network
import json
import os
from pathlib import Path
import re
import stat
from threading import BoundedSemaphore
from typing import Any, cast
from urllib.parse import parse_qs, quote, unquote, urlsplit


def _run_query(query: str, *, minimum: int, maximum: int,
               field: str = "run", comma_separated: bool = False) -> list[str]:
    if len(query) > 4096 or re.search(r"%(?![0-9a-fA-F]{2})", query):
        raise ValueError("invalid query")
    if any(part.partition("=")[0] != field for part in query.split("&") if query):
        raise ValueError("invalid query field")
    values = parse_qs(query, keep_blank_values=True, strict_parsing=True,
                      errors="strict", max_num_fields=4)
    runs = values.get(field, [])
    if comma_separated:
        if len(runs) != 1:
            raise ValueError("invalid comparison link")
        runs = runs[0].split(",")
    if not minimum <= len(runs) <= maximum or len(set(runs)) != len(runs):
        raise ValueError("invalid run count")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*:run-[0-9a-f]{32}", run) for run in runs):
        raise ValueError("invalid run key")
    return runs


class ReportServer(ThreadingHTTPServer):
    """HTTP server configuration, with explicit client and Host allowlists."""

    report_directory: Path
    report_directory_fd: int | None = None
    allowed_hosts: set[str]
    allowed_networks: tuple[IPv4Network, ...]
    max_report_bytes: int = 32 * 1024 * 1024
    workbench: Any = None
    workbench_html: bytes = b""
    workbench_slots: BoundedSemaphore

    def verify_request(self, request, client_address) -> bool:
        """Drop disallowed peers before creating a worker or reading headers."""
        client = ip_address(client_address[0])
        return any(client in network for network in self.allowed_networks)

    def server_close(self) -> None:
        super().server_close()
        if self.report_directory_fd is not None:
            os.close(self.report_directory_fd)
            self.report_directory_fd = None


class _ReportHandler(BaseHTTPRequestHandler):
    timeout = 10
    server_version = "MarketTraderReports"
    sys_version = ""

    def _respond(self, status: int, payload: bytes, content_type: str = "text/plain; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        connect = "'self'" if cast(ReportServer, self.server).workbench is not None else "'none'"
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; "
                         f"style-src 'unsafe-inline'; img-src data:; connect-src {connect}; "
                         "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        if self._authorized():
            if cast(ReportServer, self.server).workbench is not None:
                self._json_error(405, "Read-only service")
            else:
                self._respond(405, b"Read-only service\n")

    do_PUT = do_DELETE = do_PATCH = do_POST

    def _json_error(self, status: int, message: str) -> None:
        self._respond(status, json.dumps({"error": message}).encode("utf-8"),
                      "application/json; charset=utf-8")

    def _workbench_api(self, path: str, query: str) -> None:
        server = cast(ReportServer, self.server)
        if not server.workbench_slots.acquire(blocking=False):
            self._json_error(503, "Experiment reader busy; retry later")
            return
        try:
            self._read_workbench_api(path, query)
        finally:
            server.workbench_slots.release()

    def _read_workbench_api(self, path: str, query: str) -> None:
        from market_trader.workbench_data import InvalidRunError, UnknownRunError

        server = cast(ReportServer, self.server)
        try:
            minimum, maximum = (0, 0) if path == "/api/experiments" else (
                (2, 4) if path == "/api/compare" else (1, 1)
            )
            runs = _run_query(query, minimum=minimum, maximum=maximum)
            if path == "/api/experiments":
                result = server.workbench.overview()
            elif path == "/api/compare":
                result = server.workbench.compare(runs)
            elif path == "/experiment-report":
                payload = server.workbench.report(runs[0])
                if len(payload) > server.max_report_bytes:
                    self._json_error(413, "Report too large")
                else:
                    self._respond(200, payload, "text/html; charset=utf-8")
                return
            else:
                result = server.workbench.detail(runs[0])
            payload = bytearray()
            for chunk in json.JSONEncoder(allow_nan=False).iterencode(result):
                payload.extend(chunk.encode("utf-8"))
                if len(payload) > server.max_report_bytes:
                    self._json_error(413, "Response too large")
                    return
            self._respond(200, bytes(payload), "application/json; charset=utf-8")
        except UnknownRunError:
            self._json_error(404, "Unknown experiment")
        except InvalidRunError:
            self._json_error(409, "Experiment is not valid")
        except ValueError:
            self._json_error(400, "Invalid request")
        except OSError:
            self._json_error(503, "Experiment data unavailable")
        except Exception:
            self._json_error(500, "Unable to read experiment")

    def _authorized(self) -> bool:
        server = cast(ReportServer, self.server)
        client = ip_address(self.client_address[0])
        if not any(client in network for network in server.allowed_networks):
            self._respond(403, b"Forbidden\n")
            return False
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0] not in server.allowed_hosts:
            self._respond(403, b"Forbidden\n")
            return False
        return True

    def do_GET(self) -> None:
        if not self._authorized():
            return
        server = cast(ReportServer, self.server)
        try:
            target = urlsplit(self.path)
            path = unquote(target.path, errors="strict")
            if server.workbench is not None:
                if (not self.path.startswith("/") or target.scheme or target.netloc or target.fragment
                        or len(target.query) > 4096
                        or (target.path != path and not target.path.startswith("/reports/"))):
                    raise ValueError("invalid request target")
                if path in ("/", "/experiments") and target.query:
                    compare = target.query.startswith("compare=")
                    _run_query(target.query, minimum=2 if compare else 1,
                               maximum=4 if compare else 1, field="compare" if compare else "run",
                               comma_separated=compare)
                elif target.query and (path in ("/reports", "/healthz") or path.startswith("/reports/")):
                    raise ValueError("unexpected query")
        except (ValueError, UnicodeError):
            if server.workbench is not None:
                self._json_error(400, "Invalid request")
            else:
                self._respond(404, b"Not found\n")
            return
        if path == "/healthz":
            self._respond(200, b"ok\n")
            return
        if server.workbench is not None and path in (
            "/api/experiments", "/api/experiment", "/api/compare", "/experiment-report",
        ):
            self._workbench_api(path, target.query)
            return
        if server.workbench is not None and path in ("/", "/experiments"):
            self._respond(200, server.workbench_html, "text/html; charset=utf-8")
            return
        if server.workbench is not None and path.startswith("/api/"):
            self._json_error(404, "Not found")
            return
        directory_fd = server.report_directory_fd
        if directory_fd is None:
            self._respond(503, b"Unavailable\n")
            return
        if path == "/" or (server.workbench is not None and path == "/reports"):
            items = []
            with os.scandir(directory_fd) as entries:
                for entry in sorted(entries, key=lambda entry: entry.name.casefold()):
                    if (not entry.is_file(follow_symlinks=False) or Path(entry.name).suffix != ".html"
                            or entry.name.startswith(".")):
                        continue
                    items.append(f'<li><a href="/reports/{quote(entry.name, safe="")}">{escape(entry.name)}</a></li>')
            payload = (
                '<!doctype html><html lang="de"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                '<title>Market Trader – Reports</title>'
                '<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:960px;'
                'margin:48px auto;padding:0 24px}a{color:#93c5fd}li{margin:18px 0}'
                '.hint{color:#9ca3af}h1{font-size:1.8rem}</style>'
                '<h1>Market Trader · Reports</h1>'
                '<p class="hint">Interne, nur lesende Übersicht. Keine Handelsfunktionen.</p>'
                '<p class="hint">Beispielberichte mit „synthetic“ im Namen verwenden künstliche Testdaten, '
                'keine echten Marktergebnisse.</p>'
                + ('<ul>' + ''.join(items) + '</ul>' if items else '<p>Noch keine Reports vorhanden.</p>')
                + '</html>'
            ).encode("utf-8")
            self._respond(200, payload, "text/html; charset=utf-8")
            return
        prefix = "/reports/"
        name = path.removeprefix(prefix)
        if (not path.startswith(prefix) or not name or name.startswith(".")
                or "/" in name or "\\" in name or "\x00" in name or Path(name).suffix != ".html"):
            self._respond(404, b"Not found\n")
            return
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise OSError("not a regular file")
                payload = stream.read(server.max_report_bytes + 1)
        except OSError:
            self._respond(404, b"Not found\n")
            return
        if len(payload) > server.max_report_bytes:
            self._respond(413, b"Report too large\n")
            return
        self._respond(200, payload, "text/html; charset=utf-8")


def create_report_server(
    directory: Path, *, bind: str = "127.0.0.1", port: int = 8086,
    allowed_networks: tuple[str, ...] = ("127.0.0.0/8",),
    registries: dict[str, Path] | None = None,
) -> ReportServer:
    """Create a viewer bound to an explicit address, loopback by default."""
    private = tuple(IPv4Network(value) for value in (
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
    ))
    address = ip_address(bind)
    if address.version != 4 or not any(address in network for network in private):
        raise ValueError("bind must be an explicit loopback or private IPv4 address")
    networks = tuple(ip_network(value) for value in allowed_networks)
    if (not networks or any(not isinstance(network, IPv4Network)
            or not any(network.subnet_of(parent) for parent in private) for network in networks)):
        raise ValueError("allowed networks must be loopback or private IPv4 subnets")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("report directory must be a directory")
    workbench = None
    workbench_html = b""
    if registries is not None:
        from market_trader.workbench_data import WorkbenchData

        workbench = WorkbenchData(registries)
        workbench_html = Path(__file__).with_name("workbench_template.html.in").read_bytes()
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        server = ReportServer((bind, port), _ReportHandler)
    except BaseException:
        os.close(directory_fd)
        raise
    server.report_directory = directory
    server.workbench = workbench
    server.workbench_html = workbench_html
    server.workbench_slots = BoundedSemaphore(4)
    server.report_directory_fd = directory_fd
    actual_port = server.server_address[1]
    server.allowed_hosts = {f"{bind}:{actual_port}", bind}
    server.allowed_networks = tuple(cast(IPv4Network, network) for network in networks)
    return server
