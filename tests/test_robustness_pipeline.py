"""AP8 orchestration tests; synthetic-only optimizer calls, no market fits."""
import pytest


def test_ap8_request_is_a_fixed_seed_diagnostic_not_a_search():
    from market_trader.robustness_pipeline import run_request
    request=run_request('cnn',1)
    assert request['protocol']=='ap8a-development-robustness-v1'
    assert request['model_kind']=='cnn' and request['seed']==1
    assert request['test_metrics_computed'] is False
    assert request['reserved_test_accessed'] is False


@pytest.mark.parametrize('kind,seed',[('ridge',1),('cnn',0),('cnn',3),('cnn',True),('mlp',-1),('mlp',1.0)])
def test_ap8_refuses_unbudgeted_or_boolean_seed(kind,seed):
    from market_trader.robustness_pipeline import run_request
    with pytest.raises(ValueError):run_request(kind,seed)


def synthetic_references(tmp_path,*,bad_cnn_source=False):
    """Explicit untrained synthetic weights, real sealed sources and real ledger."""
    import json,shutil
    from dataclasses import asdict
    from decimal import Decimal
    from hashlib import sha256
    from market_trader._dataset_codec import json_bytes,encode
    from market_trader.artifacts import DatasetProvenance
    from market_trader.backtesting import ExecutionProfile
    from market_trader.dukascopy_import import slice_session_dataset
    from market_trader.experiment_registry import register_run
    from market_trader.image_training_data import prepare_image_cohort
    from market_trader.image_neural import _bindings,_training_rows
    from market_trader.experiment_config import canonical_model_settings
    from market_trader.neural import _TRAINING_METADATA,_decode_model
    from market_trader.reference_pipeline import _backtest,_runtime
    from market_trader.reference import _VALIDATION,_TEST,trading_summary
    from market_trader.storage import save_dataset
    from market_trader.baseline import _error_metrics
    from tests.test_image_dataset import sources
    from tests.test_image_neural import saved_body as cnn_body,seal as cnn_seal
    from tests.test_neural import saved_body as mlp_body,seal as mlp_seal
    source=tmp_path/'source';source.mkdir(mode=0o700)
    dataset,training=sources(source,size=34)
    cohort=prepare_image_cohort(dataset,training)
    training_doc=json.loads(training.read_text())
    validation=slice_session_dataset(cohort.data,_VALIDATION,cohort.data.config.calendar.coverage_end)
    mlp_settings=canonical_model_settings({'kind':'mlp','max_epochs':20})
    roots={}
    for kind in ('cnn','mlp'):
        body=cnn_body() if kind=='cnn' else mlp_body()
        body.update(_bindings(cohort) if kind=='cnn' else {k:training_doc[k] for k in _TRAINING_METADATA})
        if kind=='mlp':
            body['format']='market-trader.mlp-validation.v2';body['training_data_id']=cohort.training_data_id
            body['training']['fit_settings']={**mlp_settings,'seed':0}
            body['training']['epoch_history']=body['training']['epoch_history'][:20]
            body['training']['best_epoch']=20
        body['training'].update(_training_rows(cohort))
        predictor=None if kind=='cnn' else _decode_model(body['model'],mlp_settings['layers'])
        points=[{'at':r.decision_at.isoformat(),'predicted':0.01 if kind=='cnn' else predictor.predict(r.features),
                 'actual':float(r.target_return)} for r in cohort.training.validation]
        body['validation']={'row_count':len(points),kind:_error_metrics(tuple(p['predicted']-p['actual'] for p in points)),
                            'zero_return':_error_metrics(tuple(p['actual'] for p in points))}
        if kind=='cnn':
            body['training']['inference_parity']['row_count']=len(points)
            if bad_cnn_source:body['source_training_data_id']='sha256:'+'e'*64
        body=(cnn_seal if kind=='cnn' else mlp_seal)(body)
        cfg={'name':'synthetic untrained fixture','hypothesis':'test orchestration only','seed':0,
             'version':4 if kind=='cnn' else 3,'model':body['training']['fit_settings'].copy()}
        cfg['model'].pop('seed')
        def execute(output,kind=kind,body=body,points=points,cfg=cfg):
            output.mkdir(mode=0o700)
            shutil.copy2(dataset,output/'development-dataset.json');shutil.copy2(training,output/'training-data.json')
            origin=DatasetProvenance('synthetic fixture',())
            manifest=save_dataset(output/'validation-dataset.json',validation,provenance=origin)
            (output/(kind+'-model.json')).write_bytes(json_bytes(body))
            (output/(kind+'-validation-predictions.json')).write_bytes(json_bytes(points))
            report=_backtest(output,kind,{'name':kind+'_long','model_report':kind+'-model.json','entry_threshold':Decimal(0),'quantity':Decimal(1)},
                validation,manifest,origin,_runtime(),ExecutionProfile(initial_capital=Decimal(100000),spread=Decimal(2),slippage=Decimal('.1'),price_tick=Decimal('.001')))
            summary={'config':cfg,'training':body['training'],'training_rows':len(cohort.training.train),
                'validation_rows':len(points),'test_metrics_computed':False,'model_report_id':body['report_id'],
                'training_data_id':body['training_data_id'],'validation_dataset_id':validation.dataset_id,
                'runtime':_runtime(),'comparison_signature':{'synthetic':True},'validation_forecast':body['validation'],
                'trading':{kind:trading_summary(report)}}
            (output/'summary.json').write_bytes(json_bytes(summary,default=encode))
            files={p.name:'sha256:'+sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
            (output/'manifest.json').write_bytes(json_bytes({'status':'complete','artifacts':files}))
            return summary
        root=tmp_path/(kind+'-registry');rec=register_run(root,cfg,executor=execute)
        roots[kind]=(root,rec['run_id'])
    return roots


def test_reference_pair_checks_the_entire_real_synthetic_cohort(tmp_path):
    from market_trader.robustness_pipeline import reference_pair
    refs=synthetic_references(tmp_path)
    pair=reference_pair(*refs['cnn'],*refs['mlp'])
    assert pair.snapshot['all_rows_exact_equal'] is True
    assert pair.snapshot['training_rows']==pair.snapshot['validation_rows']>0
    assert pair.snapshot['test_metrics_computed'] is False
    assert pair.cohort.training.test==()


def test_reference_pair_rejects_validly_sealed_but_wrong_cnn_source(tmp_path):
    from market_trader.robustness_pipeline import reference_pair
    refs=synthetic_references(tmp_path,bad_cnn_source=True)
    with pytest.raises(ValueError,match='binding'):
        reference_pair(*refs['cnn'],*refs['mlp'])


def test_reference_diagnostics_reproduces_both_complete_source_ledgers(tmp_path):
    from market_trader.robustness_pipeline import reference_pair,reference_diagnostics
    refs=synthetic_references(tmp_path)
    pair=reference_pair(*refs['cnn'],*refs['mlp'])
    rows=reference_diagnostics(pair)
    assert {(r['model_kind'],r['seed'],r['reused']) for r in rows}=={('cnn',0,True),('mlp',0,True)}
    assert all(set(r['costs'])=={'half','baseline','double'} for r in rows)


def test_reference_diagnostics_rejects_different_account_ledger_even_with_equal_trade_totals(tmp_path, monkeypatch):
    from copy import deepcopy
    from market_trader import robustness_pipeline as pipeline
    refs=synthetic_references(tmp_path)
    pair=pipeline.reference_pair(*refs['cnn'],*refs['mlp'])
    original=pipeline._diagnostics
    def changed(*args,**kwargs):
        slices,costs,baseline=original(*args,**kwargs)
        baseline=deepcopy(baseline)
        baseline['accounts'][0]['balance']='999999'
        return slices,costs,baseline
    monkeypatch.setattr(pipeline,'_diagnostics',changed)
    with pytest.raises(ValueError,match='ledger'):
        pipeline.reference_diagnostics(pair)


def test_seed_fit_rejects_unbudgeted_request_without_output(tmp_path):
    from market_trader.robustness_pipeline import execute_seed_fit
    with pytest.raises(ValueError,match='seeds'):
        execute_seed_fit(None,model_kind='cnn',seed=0,output=tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_robustness_cli_reference_check_never_trains(tmp_path,capsys,monkeypatch):
    import json
    from market_trader.cli import main
    from market_trader import neural
    refs=synthetic_references(tmp_path)
    monkeypatch.setattr(neural,'mlp_report',lambda *a,**k:pytest.fail('reference command trained'))
    args=['robustness','references','--cnn-registry',str(refs['cnn'][0]),'--cnn-run',refs['cnn'][1],
          '--mlp-registry',str(refs['mlp'][0]),'--mlp-run',refs['mlp'][1]]
    assert main(args)==0
    output=capsys.readouterr()
    result=json.loads(output.out)
    assert result['all_rows_exact_equal'] is True and result['test_metrics_computed'] is False


def test_robustness_study_cli_refuses_duplicate_fit_ids(tmp_path,capsys):
    from market_trader.cli import main
    refs=synthetic_references(tmp_path)
    args=['robustness','study','--cnn-registry',str(refs['cnn'][0]),'--cnn-run',refs['cnn'][1],
          '--mlp-registry',str(refs['mlp'][0]),'--mlp-run',refs['mlp'][1],
          '--registry',str(tmp_path/'ap8'),'--output',str(tmp_path/'report')]
    for _ in range(4):args+=['--run-id','run-'+'a'*32]
    assert main(args)==1
    captured=capsys.readouterr()
    assert 'four' in captured.err and not (tmp_path/'report').exists()


def test_actual_synthetic_mlp_fit_uses_existing_ledger_and_no_published_cache(tmp_path):
    pytest.importorskip('torch')
    import json
    from market_trader.robustness_pipeline import reference_pair,execute_seed_fit
    refs=synthetic_references(tmp_path)
    pair=reference_pair(*refs['cnn'],*refs['mlp'])
    output=tmp_path/'result'
    summary=execute_seed_fit(pair,model_kind='mlp',seed=1,output=output)
    assert summary['format']=='market-trader.ap8-run.v1' and summary['seed']==1
    assert summary['model_kind']=='mlp' and summary['test_metrics_computed'] is False
    assert summary['training']['fit_settings']['seed']==1
    assert set(summary['costs'])=={'half','baseline','double'}
    assert not any(p.suffix=='.bin' or 'cache' in p.name for p in output.rglob('*'))
    manifest=json.loads((output/'manifest.json').read_text())
    assert set(manifest['artifacts'])=={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}-{'manifest.json'}


def test_study_refuses_missing_fits_without_creating_output(tmp_path):
    from market_trader.robustness_pipeline import build_study
    with pytest.raises(ValueError,match='four'):
        build_study(None,tmp_path/'registry',[],tmp_path/'study')
    assert not (tmp_path/'study').exists()


def test_synthetic_four_fit_study_in_existing_registry_is_framework_free_at_report(tmp_path):
    pytest.importorskip('torch')
    import json
    from market_trader.robustness_pipeline import reference_pair,execute_seed_fit,run_request,build_study
    from market_trader.experiment_registry import register_run,read_run
    refs=synthetic_references(tmp_path);pair=reference_pair(*refs['cnn'],*refs['mlp'])
    registry=tmp_path/'ap8';ids=[]
    for kind in ('cnn','mlp'):
        for seed in (1,2):
            record=register_run(registry,run_request(kind,seed),
                executor=lambda output,kind=kind,seed=seed:execute_seed_fit(pair,model_kind=kind,seed=seed,output=output))
            assert read_run(registry,record['run_id'])['status']=='completed'
            ids.append(record['run_id'])
    study=build_study(pair,registry,ids,tmp_path/'study')
    assert study['new_fit_count']==4 and study['reference_reuse_count']==2
    assert len(study['runs'])==6 and study['reserved_test_accessed'] is False
    assert (tmp_path/'study/report.html').read_bytes().startswith(b'<!doctype html>')
    assert json.loads((tmp_path/'study/study.json').read_text())==study


def synthetic_completed_study(tmp_path, mutate=lambda out,summary,model: None):
    """Resealed, explicitly untrained fixtures; never call an optimizer."""
    import json
    from copy import deepcopy
    from hashlib import sha256
    from market_trader import robustness_pipeline as p
    from market_trader._dataset_codec import json_bytes,encode
    from market_trader.experiment_registry import register_run
    from market_trader.reference_pipeline import _runtime
    from market_trader.image_neural import _seal
    from market_trader.image_neural_seeded import seeded_cnn_validation_evaluation
    from market_trader.neural import mlp_validation_evaluation
    refs=synthetic_references(tmp_path);pair=p.reference_pair(*refs['cnn'],*refs['mlp'])
    registry=tmp_path/'ap8';ids=[]
    for kind in ('cnn','mlp'):
        for seed in (1,2):
            def executor(out,kind=kind,seed=seed):
                out.mkdir(mode=0o700)
                source=__import__('pathlib').Path(pair.references[kind]['artifacts'])
                model=json.loads((source/(kind+'-model.json')).read_text())
                model.pop('report_id');model['training']['fit_settings']['seed']=seed
                model['training']['runtime_versions']['python']=__import__('sys').version.split()[0]
                if kind=='cnn':model['format']='market-trader.cnn-validation.v2'
                model=_seal(model);p._write(out/(kind+'-model.json'),model)
                if kind=='cnn':
                    evaluation,points=seeded_cnn_validation_evaluation(out/'cnn-model.json',source/'development-dataset.json',source/'training-data.json')
                else:evaluation,points=mlp_validation_evaluation(out/'mlp-model.json',source/'training-data.json')
                slices,costs,baseline=p._diagnostics(pair,kind,points,out)
                entry=dict(model_kind=kind,seed=seed,reused=False,run_id=out.parent.name,
                    model_report_id=model['report_id'],training_rows=pair.snapshot['training_rows'],
                    validation_rows=pair.snapshot['validation_rows'],training=deepcopy(model['training']),
                    forecast=slices['overall'],slices=slices,costs=deepcopy(costs))
                runtime=_runtime()
                summary=dict(format='market-trader.ap8-run.v1',**p.run_request(kind,seed),
                    **{k:v for k,v in entry.items() if k not in ('model_kind','seed')},runtime=runtime,
                    training_data_id=model['training_data_id'],validation_dataset_id=baseline['dataset_id'],
                    comparison_signature=dict(policy='market-trader.ap8-seed-comparison.v1',model_kind=kind,
                        cohort_rows_id=pair.snapshot['cohort_rows_id'],settings=pair.references[kind]['settings'],
                        execution=pair.snapshot['execution'],runtime=runtime),
                    validation_forecast=evaluation['validation'],trading={kind:costs['baseline']},
                    source_reference=pair.references[kind],reference_pair_id=pair.identity,study_entry=deepcopy(entry))
                for name,value in [('evaluation',evaluation),('validation-predictions',points),('source-inputs',pair.snapshot),
                                   ('slices',slices),('training-history',model['training'])]:p._write(out/(name+'.json'),value)
                mutate(out,summary,model)
                p._write(out/'summary.json',summary)
                p._write(out/'manifest.json',dict(format='market-trader.ap2-manifest.v1',status='complete',
                    test_metrics_computed=False,runtime=runtime,
                    artifacts={f.name:'sha256:'+sha256(f.read_bytes()).hexdigest() for f in out.iterdir()}))
                return json.loads(json_bytes(summary,default=encode))
            ids.append(register_run(registry,p.run_request(kind,seed),executor=executor)['run_id'])
    return pair,registry,ids


def test_untrained_complete_fixture_is_accepted_without_optimizer(tmp_path,monkeypatch):
    import builtins
    from market_trader import image_training_data
    from market_trader.robustness_pipeline import build_study
    pair,registry,ids=synthetic_completed_study(tmp_path)
    original_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in ('torch','numpy'):pytest.fail('study imported optional framework')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    monkeypatch.setattr(image_training_data,'build_input_cache',lambda *a,**k:pytest.fail('study built train image cache'))
    study=build_study(pair,registry,ids,tmp_path/'study')
    assert study['status']=='complete'
    assert any('minibatch' in item and 'initialization' in item for item in study['limitations'])


@pytest.mark.parametrize('field',['training','costs','training_rows','validation_rows','extra','top_extra'])
def test_study_rejects_checksum_valid_forged_entry(tmp_path,field):
    from market_trader.robustness_pipeline import build_study
    def mutate(out,summary,model):
        entry=summary['study_entry']
        if field=='training':entry[field]['epoch_history'][0]['training_mse']=987654.0
        elif field=='costs':entry[field]['double']=entry[field]['half']
        elif field=='top_extra':summary['unverified_claim']='accepted'
        else:entry[field]=999999
    pair,registry,ids=synthetic_completed_study(tmp_path,mutate)
    with pytest.raises(ValueError):build_study(pair,registry,ids,tmp_path/'study')
    assert not (tmp_path/'study').exists()


@pytest.mark.parametrize('case',['missing','model_id','account','scenario','factor','execution','dataset','digest','cost_projection','runtime'])
def test_study_replays_every_cost_ledger_field(tmp_path,case):
    import json
    from market_trader import robustness_pipeline as p
    def mutate(out,summary,model):
        path=out/'cost-double-backtest.json';ledger=json.loads(path.read_text())
        if case=='missing':path.unlink();return
        if case=='model_id':ledger['strategy']['model_report_id']='sha256:'+'a'*64
        elif case=='account':ledger['accounts'][0]['balance']='999999999'
        elif case=='scenario':ledger['strategy']['scenario']='half'
        elif case=='factor':ledger['strategy']['factor']='0.5'
        elif case=='execution':ledger['execution']['spread']='0'
        elif case=='dataset':ledger['dataset_artifact_id']='sha256:'+'b'*64
        elif case=='digest':ledger['strategy']['forecast_digest']='sha256:'+'c'*64
        elif case=='runtime':ledger['runtime']={'nonsense':True}
        else:
            summary['costs']['double']=summary['costs']['half']
            summary['study_entry']['costs']['double']=summary['costs']['half']
        p._write(path,ledger)
    pair,registry,ids=synthetic_completed_study(tmp_path,mutate)
    with pytest.raises(ValueError):p.build_study(pair,registry,ids,tmp_path/'study')
    assert not (tmp_path/'study').exists()


@pytest.mark.parametrize('kind',['cnn','mlp'])
def test_study_rejects_123_forecasts_even_with_replayed_ledgers_and_slices(tmp_path,kind):
    import json
    from pathlib import Path
    from dataclasses import asdict
    from datetime import datetime
    from decimal import Decimal
    from market_trader import robustness_pipeline as p
    from market_trader.robustness_analysis import cost_backtests,forecast_slices
    from market_trader.backtesting import ExecutionProfile
    def mutate(out,summary,model):
        if summary['model_kind']!=kind:return
        points=json.loads((out/'validation-predictions.json').read_text())
        for point in points:point['predicted']=123.0
        p._write(out/'validation-predictions.json',points)
        slices=forecast_slices(points);p._write(out/'slices.json',slices)
        source=Path(summary['source_reference']['artifacts'])
        execution=json.loads((source/(kind+'-backtest.json')).read_text())['execution']
        profile=ExecutionProfile(**{k:execution[k] if type(v) is bool else Decimal(execution[k]) for k,v in asdict(ExecutionProfile()).items()})
        costs={}
        for scenario,ledger,cost in cost_backtests(source/'validation-dataset.json',points,
                expected_decisions=tuple(datetime.fromisoformat(point['at']) for point in points),
                model_report_id=model['report_id'],base_profile=profile):
            p._write(out/('cost-'+scenario+'-backtest.json'),ledger);costs[scenario]=cost
        for projection in (summary,summary['study_entry']):projection.update(slices=slices,forecast=slices['overall'],costs=costs)
        summary['trading']={kind:costs['baseline']}
    pair,registry,ids=synthetic_completed_study(tmp_path,mutate)
    with pytest.raises(ValueError):p.build_study(pair,registry,ids,tmp_path/'study')


@pytest.mark.parametrize('case',['evaluation','summary_metrics','model_metrics'])
def test_study_reconciles_canonical_validation_metrics(tmp_path,case):
    from market_trader import robustness_pipeline as p
    from market_trader.image_neural import _seal
    def mutate(out,summary,model):
        if case=='evaluation':p._write(out/'evaluation.json',{'nonsense':True})
        elif case=='summary_metrics':summary['validation_forecast']={'nonsense':True}
        else:
            model['validation'][summary['model_kind']]['mae']=123.0
            model.pop('report_id');model=_seal(model)
            p._write(out/(summary['model_kind']+'-model.json'),model)
            summary['model_report_id']=summary['study_entry']['model_report_id']=model['report_id']
            import json
            for path in out.glob('cost-*-backtest.json'):
                ledger=json.loads(path.read_text());ledger['strategy']['model_report_id']=model['report_id'];p._write(path,ledger)
    pair,registry,ids=synthetic_completed_study(tmp_path,mutate)
    with pytest.raises(ValueError):p.build_study(pair,registry,ids,tmp_path/'study')


@pytest.mark.parametrize('kind,field',[(kind,field) for kind in ('cnn','mlp') for field in
    ('training_data_id','dataset_id','dataset_artifact_id','first_decision_at','row_count')]+[('cnn','source_training_data_id')])
def test_study_rejects_self_consistent_model_bound_to_wrong_source(tmp_path,kind,field):
    import json
    from copy import deepcopy
    from datetime import datetime,timedelta
    from market_trader import robustness_pipeline as p
    from market_trader.image_neural import _seal
    def mutate(out,summary,model):
        if summary['model_kind']!=kind:return
        if field=='first_decision_at':
            model['training'][field]=(datetime.fromisoformat(model['training'][field])+timedelta(minutes=15)).isoformat()
        elif field=='row_count':model['training'][field]+=1
        else:
            model[field]='sha256:'+'d'*64
            if kind=='cnn' and field=='training_data_id':model['input_index_id']=model[field]
        model.pop('report_id');model=_seal(model);p._write(out/(kind+'-model.json'),model)
        summary['model_report_id']=summary['study_entry']['model_report_id']=model['report_id']
        summary['training_data_id']=model['training_data_id']
        summary['training']=deepcopy(model['training']);summary['study_entry']['training']=deepcopy(model['training'])
        p._write(out/'training-history.json',model['training'])
        evaluation=json.loads((out/'evaluation.json').read_text());evaluation.pop('report_id')
        evaluation['model_report_id']=model['report_id']
        for key in (evaluation.keys() & model.keys())-{'format','validation'}:evaluation[key]=model[key]
        evaluation['format']=('market-trader.cnn-seeded-validation-evaluation.v1' if kind=='cnn' else 'market-trader.mlp-validation-evaluation.v1')
        p._write(out/'evaluation.json',_seal(evaluation))
        for path in out.glob('cost-*-backtest.json'):
            ledger=json.loads(path.read_text());ledger['strategy']['model_report_id']=model['report_id'];p._write(path,ledger)
    pair,registry,ids=synthetic_completed_study(tmp_path,mutate)
    with pytest.raises(ValueError):p.build_study(pair,registry,ids,tmp_path/'study')


@pytest.mark.parametrize('name',['evaluation.json','training-history.json','source-inputs.json','slices.json',
    'cost-half-backtest.json','cost-baseline-backtest.json','cost-double-backtest.json',
    'model','validation-predictions.json'])
def test_study_requires_complete_semantic_artifact_inventory(tmp_path,name):
    from market_trader import robustness_pipeline as p
    def mutate(out,summary,model):
        (out/(summary['model_kind']+'-model.json' if name=='model' else name)).unlink()
    pair,registry,ids=synthetic_completed_study(tmp_path,mutate)
    with pytest.raises(ValueError):p.build_study(pair,registry,ids,tmp_path/'study')
    assert not (tmp_path/'study').exists()


@pytest.mark.parametrize('case',['history','source_inputs','slices','signature','runtime','extra_artifact'])
def test_study_rejects_checksum_valid_detached_artifact_metadata(tmp_path,case):
    from market_trader import robustness_pipeline as p
    def mutate(out,summary,model):
        if case=='history':p._write(out/'training-history.json',{'epoch_history':[]})
        elif case=='source_inputs':p._write(out/'source-inputs.json',{'reference_pair_id':'wrong'})
        elif case=='slices':p._write(out/'slices.json',{'nonsense':True})
        elif case=='signature':summary['comparison_signature']={'synthetic':True}
        elif case=='runtime':summary['runtime']={**summary['runtime'],'python':'0.0.0'}
        else:p._write(out/'unadvertised.json',{})
    pair,registry,ids=synthetic_completed_study(tmp_path,mutate)
    with pytest.raises(ValueError):p.build_study(pair,registry,ids,tmp_path/'study')
    assert not (tmp_path/'study').exists()


@pytest.mark.parametrize('case',['fifo','symlink','oversize','duplicate_keys'])
def test_fit_json_bounds_special_files_and_duplicate_keys(tmp_path,case):
    import os
    from market_trader.robustness_pipeline import _fit_json
    path=tmp_path/'artifact.json'
    if case=='fifo':os.mkfifo(path)
    elif case=='symlink':path.symlink_to(tmp_path/'target')
    elif case=='oversize':path.write_bytes(b' '*33)
    else:path.write_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError):_fit_json(path,32)


def test_fit_inventory_bounds_before_general_registry_reads(tmp_path,monkeypatch):
    from market_trader import robustness_pipeline as p,experiment_registry
    pair,registry,ids=synthetic_completed_study(tmp_path)
    path=registry/ids[0]/'artifacts/validation-predictions.json'
    with path.open('wb') as stream:stream.truncate(8*1024*1024+1)
    original=experiment_registry.read_run
    def guarded(root,run_id):
        if root==registry:pytest.fail('unbounded registry read before artifact size preflight')
        return original(root,run_id)
    monkeypatch.setattr(experiment_registry,'read_run',guarded)
    with pytest.raises(ValueError,match='bounded'):p.build_study(pair,registry,ids,tmp_path/'study')


def test_fit_config_cannot_claim_reserved_access_even_when_rechecksummed(tmp_path):
    import json
    from hashlib import sha256
    from market_trader import robustness_pipeline as p
    pair,registry,ids=synthetic_completed_study(tmp_path)
    directory=registry/ids[0];config=json.loads((directory/'config.json').read_text())
    config['reserved_test_accessed']=True;p._write(directory/'config.json',config)
    record=json.loads((directory/'run.json').read_text())
    record['config_sha256']='sha256:'+sha256((directory/'config.json').read_bytes()).hexdigest()
    p._write(directory/'run.json',record)
    with pytest.raises(ValueError,match='configuration'):p.build_study(pair,registry,ids,tmp_path/'study')
