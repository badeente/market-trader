"""Small read-only HTTP viewer for a dedicated directory of generated reports.

This is an internal LAN/VPN viewer, not a public hosting service. It never
executes backtests or exposes raw inputs, and serves only flat HTML files.
"""

from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Network, ip_address, ip_network
import os
from pathlib import Path
import stat
from typing import cast
from urllib.parse import quote, unquote, urlsplit


class ReportServer(ThreadingHTTPServer):
    """HTTP server configuration, with explicit client and Host allowlists."""

    report_directory: Path
    report_directory_fd: int | None = None
    allowed_hosts: set[str]
    allowed_networks: tuple[IPv4Network, ...]
    max_report_bytes: int = 32 * 1024 * 1024

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
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; "
                         "style-src 'unsafe-inline'; img-src data:; connect-src 'none'; "
                         "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        self._respond(405, b"Read-only service\n")

    do_PUT = do_DELETE = do_PATCH = do_POST

    def do_GET(self) -> None:
        server = cast(ReportServer, self.server)
        client = ip_address(self.client_address[0])
        if not any(client in network for network in server.allowed_networks):
            self._respond(403, b"Forbidden\n")
            return
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0] not in server.allowed_hosts:
            self._respond(403, b"Forbidden\n")
            return
        try:
            path = unquote(urlsplit(self.path).path, errors="strict")
        except (ValueError, UnicodeError):
            self._respond(404, b"Not found\n")
            return
        if path == "/healthz":
            self._respond(200, b"ok\n")
            return
        directory_fd = server.report_directory_fd
        if directory_fd is None:
            self._respond(503, b"Unavailable\n")
            return
        if path == "/":
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
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        server = ReportServer((bind, port), _ReportHandler)
    except BaseException:
        os.close(directory_fd)
        raise
    server.report_directory = directory
    server.report_directory_fd = directory_fd
    actual_port = server.server_address[1]
    server.allowed_hosts = {f"{bind}:{actual_port}", bind}
    server.allowed_networks = tuple(cast(IPv4Network, network) for network in networks)
    return server
