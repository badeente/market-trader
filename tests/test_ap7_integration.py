"""AP7 additive protocol integration; synthetic sources, no real market fits."""
from dataclasses import replace
import sys
from types import SimpleNamespace


def test_v4_rejects_nonreference_seed_before_training(tmp_path):
    import pytest
    from tests.test_ap7_config import cnn_config
    from market_trader.experiment_config import read_experiment_config
    path=cnn_config(tmp_path);path.write_text(path.read_text().replace('seed=0','seed=1'))
    with pytest.raises(ValueError,match='seed 0'):
        read_experiment_config(path)


def test_checked_in_cnn_reference_uses_fixed_policy_and_original_source():
    from pathlib import Path
    from market_trader.experiment_config import read_experiment_config
    path=Path(__file__).resolve().parents[1]/'configs/ap7-cnn-reference.toml'
    config=read_experiment_config(path)
    assert config.version==4 and config.model_kind=='cnn' and config.seed==0
    assert config.model_settings['max_epochs']==20
    assert config.source_sha256=='sha256:d7c694f295a863076b4b00ceb0a2adfbeed6f1b0458a2098a52cd171e24cef2f'


def test_v4_dispatches_to_image_pipeline_before_legacy_source_or_fit(tmp_path,monkeypatch):
    from tests.test_ap7_config import cnn_config
    from market_trader.experiment_config import read_experiment_config
    from market_trader.experiment_pipeline import execute_experiment
    config=read_experiment_config(cnn_config(tmp_path))
    seen=[]
    def execute(config,output):
        seen.append((config,output))
        return {'synthetic_dispatch_only':True}
    monkeypatch.setitem(sys.modules,'market_trader.image_experiment_pipeline',SimpleNamespace(execute_image_experiment=execute))
    output=tmp_path/'output'
    assert execute_experiment(config,output)=={'synthetic_dispatch_only':True}
    assert seen==[(config,output)] and not output.exists()


def test_cnn_explicit_dispatch_uses_bounded_loader_without_generic_read(tmp_path,monkeypatch):
    from pathlib import Path
    from tests.test_image_neural import saved_body
    from market_trader._dataset_codec import json_bytes
    from market_trader.strategies import load_frozen_model
    body=saved_body();path=tmp_path/'cnn.json';path.write_bytes(json_bytes(body))
    def forbidden(*args,**kwargs):
        raise AssertionError('unbounded generic model read')
    monkeypatch.setattr(Path,'read_text',forbidden)
    assert load_frozen_model(path,expected_kind='cnn').report_id==body['report_id']


def test_cnn_json_loader_and_long_strategy_use_causal_image_predictor(tmp_path):
    from tests.test_image_neural import saved_body
    from tests.test_image_dataset import sources
    from market_trader._dataset_codec import json_bytes
    from market_trader.storage import load_dataset
    from market_trader.dukascopy_import import slice_session_dataset
    from market_trader.reference import _VALIDATION,_TEST
    from market_trader.strategies import load_frozen_model,configure_strategy
    from market_trader.backtesting import run_backtest
    path=tmp_path/'cnn-model.json';body=saved_body();path.write_bytes(json_bytes(body))
    saved=load_frozen_model(path,expected_kind='cnn')
    assert saved.report_id==body['report_id']
    dataset,_=sources(tmp_path)
    original=load_dataset(dataset).dataset
    validation=slice_session_dataset(original,_VALIDATION,original.config.calendar.coverage_end)
    strategy=configure_strategy({'name':'cnn_long','model_report':path.name,'entry_threshold':0,'quantity':1},config_dir=tmp_path)
    result=run_backtest(validation,strategy.create(validation),lookback=8)
    assert result.trades
    assert all(t.exit.candle_open_time-t.entry.candle_open_time==validation.interval*2 for t in result.trades)


def test_cnn_derive_and_search_are_explicitly_unsupported(tmp_path,monkeypatch):
    from market_trader import experiment_variants as variants
    from market_trader.experiment_config import read_experiment_config,experiment_config_payload
    from tests.test_ap7_config import cnn_config
    body=experiment_config_payload(read_experiment_config(cnn_config(tmp_path)))
    monkeypatch.setattr(variants,'_verified_record',lambda *a:({'config_sha256':'synthetic'},body))
    import pytest
    with pytest.raises(ValueError,match='CNN.*unsupported'):
        variants.derive_variant(tmp_path,'run-synthetic',{'seed':1},name='no search',hypothesis='must reject')


def test_framework_reload_parity_uses_actual_saved_weights_and_bounded_batches(tmp_path):
    import importlib.util,json,pytest
    if importlib.util.find_spec('torch') is None:pytest.skip('optional framework parity test')
    from tests.test_image_neural import saved_body
    from market_trader._dataset_codec import json_bytes
    from market_trader.image_experiment_pipeline import _framework_reload_parity
    path=tmp_path/'model.json';path.write_bytes(json_bytes(saved_body()))
    cache_path=tmp_path/'inputs.bin';cache_path.write_bytes(b'\x00'*(5*2304*2))
    cache=SimpleNamespace(path=cache_path,row_count=5,train_count=2)
    evidence=_framework_reload_parity(path,cache,[{'predicted':0.01} for _ in range(3)])
    assert evidence['validation_rows']==3 and evidence['max_absolute_error']==0
    assert evidence['tolerance']==1e-12


def test_actual_v4_pipeline_has_cnn_and_controls_with_no_published_cache(tmp_path):
    import importlib.util,json,pytest
    if importlib.util.find_spec('torch') is None:pytest.skip('optional CNN training runtime')
    from market_trader.cnn_settings import canonical_cnn_settings
    from market_trader.experiment_pipeline import execute_experiment
    from tests.test_experiment_pipeline import settings
    config,_=settings(tmp_path)
    config=replace(config,version=4,model_kind='cnn',model_settings=canonical_cnn_settings())
    output=tmp_path/'result'
    summary=execute_experiment(config,output)
    assert summary['config']['model']['kind']=='cnn'
    assert summary['format']=='market-trader.ap7-summary.v1'
    assert summary['test_metrics_computed'] is False
    assert set(summary['trading'])=={'cnn','ridge','sma_timed'}
    assert {'cnn','ridge','zero','zero_return'}<=summary['validation_forecast'].keys()
    assert summary['training_rows']==summary['validation_rows']==48
    assert summary['training_data_id']!=summary['source_training_data_id']
    assert not any('cache' in str(path.relative_to(output)) or path.suffix=='.bin' for path in output.rglob('*'))
    manifest=json.loads((output/'manifest.json').read_text())
    assert set(manifest['artifacts'])=={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}-{'manifest.json'}
    assert json.loads((output/'image-index.json').read_text())['samples']
    html=(output/'report.html').read_text()
    assert 'CNN' in html and 'data:image/png;base64,' in html
