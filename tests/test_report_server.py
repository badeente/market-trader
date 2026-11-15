"""HTTP smoke/security checks for the read-only internal report viewer."""

from contextlib import contextmanager
from http.client import HTTPConnection
from importlib.util import find_spec
from threading import Thread


def request(server, path="/", *, method="GET", host=None):
    connection = HTTPConnection(*server.server_address, timeout=5)
    headers = {} if host is None else {"Host": host}
    try:
        connection.request(method, path, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


@contextmanager
def running(directory, **kwargs):
    assert find_spec("market_trader.report_server") is not None, "read-only report server is not implemented"
    from market_trader.report_server import create_report_server

    server = create_report_server(directory, bind="127.0.0.1", port=0, **kwargs)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_report_viewer_pins_root_directory_against_replacement(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "original.html").write_text("original report")
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "secret.html").write_text("secret content")
    with running(reports) as server:
        reports.rename(tmp_path / "original-directory")
        reports.symlink_to(outside, target_is_directory=True)
        assert request(server, "/reports/secret.html")[0] == 404
        assert b"secret.html" not in request(server)[2]
        assert request(server, "/reports/original.html")[2] == b"original report"


def test_report_viewer_cli_has_safe_defaults_and_reports_invalid_configuration(tmp_path):
    import subprocess
    import sys

    command = [sys.executable, "-m", "market_trader.cli", "serve-reports"]
    result = subprocess.run(command + ["--help"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert "--directory" in result.stdout and "--allow-network" in result.stdout
    result = subprocess.run(command + ["--directory", str(tmp_path), "--bind", "0.0.0.0"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert "private IPv4" in result.stderr
    assert "Traceback" not in result.stderr


def test_report_viewer_rejects_oversized_reports(tmp_path):
    (tmp_path / "large.html").write_bytes(b"x" * 100)
    with running(tmp_path) as server:
        server.max_report_bytes = 50
        assert request(server, "/reports/large.html")[0] == 413


def test_report_viewer_never_follows_symlinks_even_if_precheck_misses_them(tmp_path, monkeypatch):
    from pathlib import Path

    outside = tmp_path.parent / "outside-secret.html"
    outside.write_text("private secret")
    (tmp_path / "linked.html").symlink_to(outside)
    # Simulate a link appearing after a pathname precheck.
    monkeypatch.setattr(Path, "is_symlink", lambda _: False)
    with running(tmp_path) as server:
        assert request(server, "/reports/linked.html")[0] == 404
        for path in ("/reports/../outside-secret.html", "/reports/%2e%2e%2foutside-secret.html",
                     "/reports/.env.html", "/reports/subdir/report.html", "/reports/a%00.html"):
            assert request(server, path)[0] == 404


def test_report_viewer_has_read_only_methods_and_security_headers(tmp_path):
    (tmp_path / "example.html").write_text("<title>Example</title>")
    with running(tmp_path) as server:
        status, headers, body = request(server, "/reports/example.html", method="HEAD")
        assert status == 200 and body == b""
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Cache-Control"] == "no-store"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert "connect-src 'none'" in headers["Content-Security-Policy"]
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            assert request(server, "/reports/example.html", method=method)[0] == 405
        assert request(server, "/healthz")[0] == 200


def test_report_viewer_supports_explicit_private_networks_only(tmp_path):
    from inspect import signature
    import pytest
    from market_trader.report_server import create_report_server

    assert "allowed_networks" in signature(create_report_server).parameters
    with running(tmp_path, allowed_networks=("127.0.0.0/8", "10.10.1.0/24")) as server:
        assert request(server)[0] == 200
    for invalid in ("0.0.0.0/0", "203.0.113.0/24", "::/0"):
        with pytest.raises(ValueError, match="private IPv4"):
            create_report_server(tmp_path, port=0, allowed_networks=(invalid,))
    for bind in ("0.0.0.0", "203.0.113.1", "::"):
        with pytest.raises(ValueError, match="private IPv4"):
            create_report_server(tmp_path, bind=bind, port=0)


def test_report_viewer_denies_clients_before_reading_http_headers(tmp_path):
    from ipaddress import IPv4Network
    from socket import create_connection
    import pytest

    with running(tmp_path) as server:
        server.allowed_networks = (IPv4Network("192.168.50.0/24"),)
        with create_connection(server.server_address, timeout=1) as connection:
            connection.settimeout(0.3)
            try:
                assert connection.recv(1) == b""
            except TimeoutError:
                pytest.fail("denied client occupied an HTTP worker before sending headers")


def test_report_viewer_rejects_untrusted_host(tmp_path):
    with running(tmp_path) as server:
        assert request(server, host="untrusted.example:8086")[0] == 403


def test_report_viewer_lists_and_serves_only_html(tmp_path):
    (tmp_path / "example.html").write_text("<!doctype html><title>Example</title>", encoding="utf-8")
    (tmp_path / "secret.json").write_text('{"private": true}', encoding="utf-8")
    with running(tmp_path) as server:
        status, headers, body = request(server)
        assert status == 200
        assert b"example.html" in body
        assert b"secret.json" not in body
        status, headers, body = request(server, "/reports/example.html")
        assert status == 200
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert body == b"<!doctype html><title>Example</title>"
        assert request(server, "/reports/secret.json")[0] == 404
