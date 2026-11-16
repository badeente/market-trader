"""HTTP/CLI contracts; dispatch-only tests explicitly stub WorkbenchData."""

from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from types import ModuleType
import json
import sys

import pytest


def request(server, path="/", *, method="GET", host=None):
    connection = HTTPConnection(*server.server_address, timeout=5)
    try:
        connection.request(method, path, headers={} if host is None else {"Host": host})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


@contextmanager
def running(directory, **kwargs):
    from market_trader.report_server import create_report_server

    with create_report_server(directory, port=0, **kwargs) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.fixture
def dispatch_data(monkeypatch):
    """A deliberately synthetic data boundary, NOT artifact verification."""
    module = ModuleType("market_trader.workbench_data")
    class UnknownRunError(ValueError):
        pass
    class InvalidRunError(ValueError):
        pass
    class WorkbenchData:
        def __init__(self, registries, *, curve_limit=800):
            self.registries = registries
            self.calls = []
        def overview(self):
            self.calls.append(("overview",))
            return {"runs": [], "sources": [], "errors": [], "read_only": True}
    module.WorkbenchData = WorkbenchData
    module.UnknownRunError = UnknownRunError
    module.InvalidRunError = InvalidRunError
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_enabled_overview_empty_json_and_head(tmp_path, dispatch_data):
    with running(tmp_path, registries={}) as server:
        status, headers, body = request(server, "/api/experiments")
        assert status == 200
        assert json.loads(body) == {"runs": [], "sources": [], "errors": [], "read_only": True}
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        assert headers["Cache-Control"] == "no-store"
        assert "connect-src 'self'" in headers["Content-Security-Policy"]
        assert not any(key.lower().startswith("access-control") for key in headers)
        status, head_headers, body_head = request(server, "/api/experiments", method="HEAD")
        assert status == 200 and body_head == b""
        assert head_headers["Content-Length"] == str(len(body))


KEY = "ap2:run-" + "a" * 32
OTHER = "ap3:run-" + "b" * 32


def test_detail_compare_report_dispatch_and_head(tmp_path, dispatch_data, monkeypatch):
    monkeypatch.setattr(dispatch_data.WorkbenchData, "detail", lambda self, key: {"key": key}, raising=False)
    monkeypatch.setattr(dispatch_data.WorkbenchData, "compare", lambda self, keys: {"keys": keys}, raising=False)
    monkeypatch.setattr(dispatch_data.WorkbenchData, "report", lambda self, key: b"<title>Verified report</title>", raising=False)
    with running(tmp_path, registries={}) as server:
        for path, expected in (
            (f"/api/experiment?run={KEY}", {"key": KEY}),
            (f"/api/compare?run={KEY}&run={OTHER}", {"keys": [KEY, OTHER]}),
            (f"/experiment-report?run={KEY}", b"<title>Verified report</title>"),
        ):
            status, headers, body = request(server, path)
            assert status == 200
            assert (body if isinstance(expected, bytes) else json.loads(body)) == expected
            status, heads, head_body = request(server, path, method="HEAD")
            assert status == 200 and head_body == b""
            assert heads["Content-Length"] == str(len(body))


@pytest.mark.parametrize("error_name,status", [("UnknownRunError", 404), ("InvalidRunError", 409), ("ValueError", 400), ("OSError", 503), ("RuntimeError", 500)])
def test_data_failures_are_safe_json(tmp_path, dispatch_data, monkeypatch, error_name, status):
    errors = {"ValueError": ValueError, "OSError": OSError, "RuntimeError": RuntimeError}
    error_type = errors.get(error_name) or getattr(dispatch_data, error_name)
    def fail(self, key):
        raise error_type("private /absolute/secret/source.csv Traceback")
    monkeypatch.setattr(dispatch_data.WorkbenchData, "detail", fail, raising=False)
    with running(tmp_path, registries={}) as server:
        actual, headers, body = request(server, f"/api/experiment?run={KEY}")
        assert actual == status
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        assert isinstance(json.loads(body)["error"], str)
        assert b"/absolute" not in body and b"Traceback" not in body


def test_strict_bounded_queries_reject_before_dispatch(tmp_path, dispatch_data, monkeypatch):
    def forbidden(*args):
        pytest.fail("malformed query reached data boundary")
    for method in ("detail", "compare", "report", "overview"):
        monkeypatch.setattr(dispatch_data.WorkbenchData, method, forbidden, raising=False)
    paths = [
        "/api/experiment", "/api/experiment?run=", f"/api/experiment?run={KEY}&run={KEY}",
        f"/api/experiment?wrong={KEY}", f"/api/experiment?run={KEY}&file=/etc/passwd",
        "/api/experiment?run=%ZZ", "/api/experiment?run=%FF", "/api/experiment?run",
        f"/api/experiment?%72un={KEY}", f"/api/experiment?run={KEY}&",
        f"/api/experiment?run={KEY};file=secret", f"/api/experiment?run={KEY}%00",
        "/api/experiment?run=ap2:../secret", "/api/experiment?run=ap2%3Arun-%2e%2e%2f",
        f"/api/experiment?run=+{KEY}", f"/api/experiment?run={KEY.upper()}",
        f"/api/compare?run={KEY}", f"/api/compare?run={KEY}&run={KEY}",
        "/api/compare?" + "&".join(f"run=ap2:run-{n:032x}" for n in range(5)),
        "/api/experiments?run=anything", "/api/experiment?run=" + "x" * 4096,
        f"/experiment-report?run={KEY}&path=/etc/passwd",
    ]
    with running(tmp_path, registries={}) as server:
        for path in paths:
            status, headers, body = request(server, path)
            assert status == 400, path
            assert "error" in json.loads(body), path


def test_workbench_acl_and_mutation_checks_precede_dispatch(tmp_path, dispatch_data):
    from ipaddress import IPv4Network
    from socket import create_connection

    with running(tmp_path, registries={}) as server:
        for path in ("/", "/healthz", "/api/experiments", f"/api/experiment?run={KEY}"):
            for method in ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"):
                assert request(server, path, method=method, host="evil.example")[0] == 403
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            assert request(server, "/api/experiments", method=method)[0] == 405
        server.allowed_networks = (IPv4Network("192.168.50.0/24"),)
        with create_connection(server.server_address, timeout=1) as connection:
            connection.settimeout(0.3)
            assert connection.recv(1) == b""


def test_ui_routes_gallery_compatibility_and_deep_links(tmp_path, dispatch_data):
    from market_trader import report_server
    template = Path(report_server.__file__).with_name("workbench_template.html.in").read_bytes()
    (tmp_path / "legacy.html").write_bytes(b"old report")
    (tmp_path / "secret.json").write_bytes(b"private")
    (tmp_path / "linked.html").symlink_to(tmp_path / "secret.json")
    with running(tmp_path, registries={}) as server:
        for path in ("/", "/experiments", f"/experiments?run={KEY}", f"/?compare={KEY},{OTHER}"):
            status, headers, body = request(server, path)
            assert status == 200 and body == template
            status, heads, body = request(server, path, method="HEAD")
            assert status == 200 and body == b""
            assert heads["Content-Length"] == str(len(template))
        status, headers, body = request(server, "/reports")
        assert status == 200 and b"legacy.html" in body and b"secret.json" not in body
        assert request(server, "/reports/legacy.html")[2] == b"old report"
        assert request(server, "/reports/linked.html")[0] == 404
        for path in ("/reports/%2e%2e%2fsecret.html", "/files/secret.json", "/api/file?path=/etc/passwd"):
            assert request(server, path)[0] == 404
    with running(tmp_path) as server:
        assert b"legacy.html" in request(server)[2]
        assert "connect-src 'none'" in request(server)[1]["Content-Security-Policy"]
        for path in ("/experiments", "/reports", "/api/experiments"):
            assert request(server, path)[0] == 404


def test_workbench_payload_caps_apply_to_json_and_verified_report(tmp_path, dispatch_data, monkeypatch):
    monkeypatch.setattr(dispatch_data.WorkbenchData, "report", lambda self, key: b"x" * 101, raising=False)
    monkeypatch.setattr(dispatch_data.WorkbenchData, "overview", lambda self: {"data": "x" * 101})
    with running(tmp_path, registries={}) as server:
        server.max_report_bytes = 100
        for path in ("/api/experiments", f"/experiment-report?run={KEY}"):
            status, headers, body = request(server, path)
            assert status == 413
            assert "error" in json.loads(body)
            assert len(body) < 100


def test_workbench_bounds_concurrent_heavy_requests(tmp_path, dispatch_data, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event
    entered, release = Barrier(5), Event()
    def blocked(self):
        entered.wait(timeout=4)
        assert release.wait(timeout=4)
        return {"runs": []}
    monkeypatch.setattr(dispatch_data.WorkbenchData, "overview", blocked)
    with running(tmp_path, registries={}) as server, ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(request, server, "/api/experiments") for _ in range(4)]
        try:
            entered.wait(timeout=4)
            assert request(server, "/api/experiments")[0] == 503
            assert request(server, "/healthz")[0] == 200
        finally:
            release.set()
        assert all(future.result()[0] == 200 for future in futures)


def test_cli_registry_mapping_and_legacy_none(tmp_path, monkeypatch):
    from market_trader import cli
    calls = []
    class Server:
        server_address = ("127.0.0.1", 8086)
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def serve_forever(self): pass
    def create(directory, **kwargs):
        calls.append((directory, kwargs))
        return Server()
    monkeypatch.setattr(cli, "create_report_server", create)
    base = ["serve-reports", "--directory", str(tmp_path)]
    assert cli.main(base) == 0
    assert calls[-1][1].get("registries") is None
    assert cli.main(base + ["--registry", "ap2=/private/ap2", "--registry", "ap3=/private/ap3"]) == 0
    assert calls[-1] == (tmp_path, {"bind": "127.0.0.1", "port": 8086,
                                  "allowed_networks": ("127.0.0.0/8",),
                                  "registries": {"ap2": Path("/private/ap2"), "ap3": Path("/private/ap3")}})


def test_cli_registry_tokens_are_not_repaired(tmp_path, monkeypatch, capsys):
    from market_trader import cli
    def forbidden(*args, **kwargs):
        pytest.fail("invalid registry token reached server creation")
    monkeypatch.setattr(cli, "create_report_server", forbidden)
    base = ["serve-reports", "--directory", str(tmp_path)]
    for tokens in (["ap2=/private", "ap2=/another"], ["AP2=/private"], [" ap2=/private"],
                   ["ap-2=/private"], ["ap2=relative"], ["ap2="], ["=/private"],
                   ["ap2=/private "], ["ap2=/private\nsecret"], ["ap2"],
                   ["ap2=/private/../secret"], ["ap2=/private/./secret"], ["ap2=/private//secret"]):
        assert cli.main(base + [part for token in tokens for part in ("--registry", token)]) == 1
        assert "Traceback" not in capsys.readouterr().err


def test_workbench_request_targets_and_ui_queries_are_strict(tmp_path, dispatch_data):
    with running(tmp_path, registries={}) as server:
        for path in ("/?file=/etc/passwd", "/experiments?run=%FF", "/experiments?run=%ZZ",
                     f"/?run={KEY}&run={KEY}", f"/?compare={KEY},{KEY}", "/?" + "x" * 4097,
                     "/api/experiments#fragment", "http://evil.example/api/experiments",
                     "/api/%65xperiments", "/experiments?run=ap-2:run-" + "a" * 32):
            status, headers, body = request(server, path, host=f"127.0.0.1:{server.server_address[1]}")
            assert status == 400, path
            assert "error" in json.loads(body)


def synthetic_registered_run(root):
    """Real registry integrity envelope around explicitly synthetic bytes, no fit."""
    from market_trader import experiment_registry as registry
    from market_trader.experiment_config import ExperimentConfig, experiment_config_payload
    config = experiment_config_payload(ExperimentConfig(
        version=1, name="synthetic HTTP fixture", hypothesis="dispatch integrity only", seed=7,
        source_csv=Path("/private/input.csv"), source_sha256="sha256:" + "a" * 64))
    def executor(output):
        output.mkdir()
        summary = {
            "runtime": {"code_id": "fixture", "python": "3.14", "dependencies": {}},
            "training_rows": 3, "validation_rows": 2, "test_metrics_computed": False,
            "model_report_id": "synthetic-model", "training_data_id": "synthetic-training",
            "validation_dataset_id": "synthetic-validation", "comparison_signature": {"source_sha256": "fixture"},
            "validation_forecast": {"ridge": {"mae": 0.0002, "rmse": 0.0003}},
            "trading": {"ridge": {"net_points": "-3", "modeled_cost_points": "2.2",
                                  "max_drawdown_points": "3", "trade_count": 1}},
        }
        files = {"summary.json": registry._encode(summary),
                 "report.html": b"<title>Synthetic HTTP fixture; not a financial run</title>",
                 "ridge-backtest.json": registry._encode({"execution": {"initial_capital": "100000"},
                    "accounts": [{"at": "2024-01-02T08:00:00+00:00", "equity": "99997"}]}),
                 "weights.bin": b"PRIVATE WEIGHTS"}
        for name, content in files.items():
            (output / name).write_bytes(content)
        (output / "manifest.json").write_bytes(registry._encode({"status": "complete", "artifacts": {
            name: registry._digest(content) for name, content in files.items()}}))
        return summary
    record = registry.register_run(root, config, executor=executor)
    assert record["status"] == "completed"
    return record["run_id"]


def test_real_data_layer_synthetic_integrity_envelopes_over_http(tmp_path):
    reports, root = tmp_path / "reports", tmp_path / "registry"
    reports.mkdir()
    root.mkdir()
    first, second = synthetic_registered_run(root), synthetic_registered_run(root)
    key, other = "ap2:" + first, "ap2:" + second
    with running(reports, registries={"ap2": root}) as server:
        status, headers, body = request(server, "/api/experiments")
        assert status == 200
        rows = json.loads(body)["runs"]
        assert {row["key"] for row in rows} == {key, other}
        assert all(row["metrics"]["net_points"] == "-3" for row in rows)
        status, headers, body = request(server, f"/api/experiment?run={key}")
        detail = json.loads(body)
        assert status == 200 and detail["run"]["status"] == "completed"
        assert detail["curves"]["equity"][0][1] == "-3"
        assert detail["config"]["source_csv"] == "[redacted]"
        assert str(tmp_path).encode() not in body and b"PRIVATE WEIGHTS" not in body
        assert request(server, f"/api/compare?run={key}&run={other}")[0] == 200
        report_path = f"/experiment-report?run={key}"
        status, headers, report = request(server, report_path)
        assert status == 200 and report == (root / first / "artifacts/report.html").read_bytes()
        assert request(server, report_path, method="HEAD")[1]["Content-Length"] == str(len(report))
        for path in (f"/api/experiment?run=missing:{first}", f"/api/experiment?run={KEY}"):
            assert request(server, path)[0] == 404
        (root / first / "artifacts/weights.bin").write_bytes(b"changed private weights")
        assert request(server, report_path)[0] == 409
        status, headers, body = request(server, f"/api/experiment?run={key}")
        assert status == 200 and json.loads(body)["run"]["status"] == "invalid"
        assert request(server, f"/api/compare?run={key}&run={other}")[0] == 409


def test_real_data_layer_empty_and_symlinks_do_not_expose_private_bytes(tmp_path):
    reports, root = tmp_path / "reports", tmp_path / "registry"
    reports.mkdir()
    root.mkdir()
    with running(reports, registries={"ap2": root}) as server:
        assert json.loads(request(server, "/api/experiments")[2])["runs"] == []
    run_id = synthetic_registered_run(root)
    report = root / run_id / "artifacts/report.html"
    secret = tmp_path / "private-secret.html"
    secret.write_bytes(b"PRIVATE SECRET CONTENT")
    report.unlink()
    report.symlink_to(secret)
    source_link = tmp_path / "source-link"
    source_link.symlink_to(root, target_is_directory=True)
    with running(reports, registries={"ap2": root, "linked": source_link}) as server:
        for path in ("/api/experiments", f"/api/experiment?run=ap2:{run_id}",
                     f"/experiment-report?run=ap2:{run_id}", f"/experiment-report?run=linked:{run_id}"):
            status, headers, body = request(server, path)
            assert status in (200, 409)
            assert b"PRIVATE SECRET CONTENT" not in body and str(tmp_path).encode() not in body
        assert request(server, f"/experiment-report?run=ap2:{run_id}")[0] == 409


def test_enabled_api_route_and_method_errors_are_json(tmp_path, dispatch_data):
    with running(tmp_path, registries={}) as server:
        for path, method, expected in (("/api/unknown", "GET", 404),
                                       ("/api/experiments", "POST", 405)):
            status, headers, body = request(server, path, method=method)
            assert status == expected
            assert headers["Content-Type"] == "application/json; charset=utf-8"
            assert "error" in json.loads(body)
