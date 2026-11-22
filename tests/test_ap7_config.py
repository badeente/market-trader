"""AP7 is additive v4; historical snapshot semantics are not reinterpreted."""
from copy import deepcopy
import pytest
from market_trader.experiment_config import read_experiment_config,experiment_config_payload,experiment_config_from_payload


def cnn_config(tmp_path,extra=''):
    path=tmp_path/'cnn.toml'
    path.write_text('version=4\nname="visual"\nhypothesis="fixed causal CNN reference"\nseed=0\nsource_csv="source.csv"\nsource_sha256="sha256:'+'a'*64+'"\n'+extra+'\n[model]\nkind="cnn"\n')
    return path


def test_v4_cnn_roundtrip_is_complete_and_has_no_training_imports(tmp_path):
    config=read_experiment_config(cnn_config(tmp_path))
    body=experiment_config_payload(config)
    assert config.version==4 and config.model_kind=='cnn'
    assert body['model']['kind']=='cnn' and body['model']['input_shape']==[3,24,32]
    assert body['model']['max_epochs']==20
    assert body['training']=={'lookback_bars':8,'horizon_bars':2}
    assert experiment_config_payload(experiment_config_from_payload(body))==body


@pytest.mark.parametrize('extra',['[training]\nlookback_bars=6\nhorizon_bars=2\n[strategy]\nsma_slow_period=6',
                                  '[training]\nhorizon_bars=3','[strategy]\nsma_fast_period=3'])
def test_v4_fixed_window_and_benchmark_cannot_drift(tmp_path,extra):
    with pytest.raises(ValueError):read_experiment_config(cnn_config(tmp_path,extra))


@pytest.mark.parametrize('kind',['ridge','mlp','forest'])
def test_v4_accepts_only_explicit_cnn(tmp_path,kind):
    path=cnn_config(tmp_path)
    path.write_text(path.read_text().replace('kind="cnn"','kind="'+kind+'"'))
    with pytest.raises(ValueError):read_experiment_config(path)


@pytest.mark.parametrize('version',[1,2,3])
def test_legacy_versions_keep_rejecting_cnn(tmp_path,version):
    path=cnn_config(tmp_path)
    path.write_text(path.read_text().replace('version=4','version='+str(version)))
    with pytest.raises(ValueError):read_experiment_config(path)


def test_v4_snapshot_rejects_missing_settings_aliases_and_oversized_seed(tmp_path):
    body=experiment_config_payload(read_experiment_config(cnn_config(tmp_path)))
    cases=[]
    missing=deepcopy(body);missing['model'].pop('dtype');cases.append(missing)
    alias=deepcopy(body);alias['model']['threads']=True;cases.append(alias)
    extra=deepcopy(body);extra['model']['max_epochs']=21;cases.append(extra)
    seed=deepcopy(body);seed['seed']=2**63;cases.append(seed)
    for invalid in cases:
        with pytest.raises(ValueError):experiment_config_from_payload(invalid)
