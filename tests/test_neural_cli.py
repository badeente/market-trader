"""AP3 CLI reaches actual training, registry replay and Torch-free inference."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_neural_cli_trains_replays_and_loads_without_torch(tmp_path):
    pytest.importorskip('torch')
    from tests.test_reference import reference_fixture

    assert importlib.util.find_spec('market_trader.neural') is not None, 'AP3 neural implementation is missing'
    source, reference = reference_fixture(tmp_path)
    raw = source.read_bytes()
    config = tmp_path/'neural.toml'
    config.write_text(reference.read_text().replace('version = 1', 'version = 2').replace(
        'experiment = "ap1-deuidxeur-open30-v1"',
        'name = "synthetic neural integration"\nhypothesis = "save reload and replay"\nseed = 0')
        + '\n[model]\nkind = "mlp"\n')
    root = tmp_path/'registry'
    def cli(*args):
        result = subprocess.run([sys.executable, '-m', 'market_trader.cli', 'experiment',
                                  *args, '--registry', str(root)], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    first = cli('run', '--config', str(config))
    second = cli('replay', '--run-id', first['run_id'])
    assert first['status'] == second['status'] == 'completed'
    assert second['parent_id'] == first['run_id']
    assert cli('compare', '--left', first['run_id'], '--right', second['run_id'])['compatible'] is True
    left, right = (root/record['run_id']/'artifacts' for record in (first, second))
    assert sorted(p.name for p in left.iterdir()) == sorted(p.name for p in right.iterdir())
    assert all(p.read_bytes() == (right/p.name).read_bytes() for p in left.iterdir())
    summary = json.loads((left/'summary.json').read_text())
    assert summary['test_metrics_computed'] is False
    assert {'mlp','ridge','zero_return'} <= summary['validation_forecast'].keys()
    assert {'mlp','ridge','sma_timed'} <= summary['trading'].keys()
    assert (left/'training-history.json').exists()
    training = json.loads((left/'training-data.json').read_text())
    assert training['splits']['test'] == []
    assert source.read_bytes() == raw
    # Same interpreter, clean process: inference itself must not load Torch.
    code = ('import sys,json; from decimal import Decimal; '
            'from market_trader.neural import load_mlp_model; '
            'saved=load_mlp_model(__import__("pathlib").Path(sys.argv[1])); '
            'print(json.dumps(saved.model.predict(tuple(map(Decimal,["0.01","0.02","0.03","0.04"]))))); '
            'assert "torch" not in sys.modules')
    check = subprocess.run([sys.executable, '-c', code, str(left/'mlp-model.json')],
                            capture_output=True, text=True, timeout=20)
    assert check.returncode == 0, check.stderr
    assert isinstance(json.loads(check.stdout), float)
