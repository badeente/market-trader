"""AP8A additive private robustness orchestration; reserved 2025 is never scored."""
PROTOCOL='ap8a-development-robustness-v1'

from dataclasses import dataclass,asdict
from hashlib import sha256
import json,math
from pathlib import Path
from market_trader._dataset_codec import json_bytes,encode,decode_record


@dataclass(frozen=True)
class ReferencePair:
    snapshot: dict
    references: dict
    cohort: object
    points: dict
    models: dict
    profile: object
    identity: str


def _digest(value):
    return 'sha256:'+sha256(json_bytes(value,default=encode)).hexdigest()


def reference_pair(cnn_registry,cnn_run,mlp_registry,mlp_run):
    """Verify both seed0 sources and exact rows before any new optimization."""
    from market_trader.experiment_registry import read_run
    from market_trader.image_training_data import prepare_image_cohort
    from market_trader.image_neural import load_cnn_model
    from market_trader.neural import load_mlp_model,_prepare
    from market_trader.experiment_config import canonical_model_settings
    from market_trader.cnn_settings import canonical_cnn_settings
    from market_trader.storage import load_dataset
    from market_trader.backtesting import ExecutionProfile
    refs={};models={};points={};profiles={}
    wanted={'cnn':canonical_cnn_settings(),'mlp':canonical_model_settings({'kind':'mlp','max_epochs':20})}
    for kind,root,run in [('cnn',Path(cnn_registry),cnn_run),('mlp',Path(mlp_registry),mlp_run)]:
        record=read_run(root,run)
        if record['status']!='completed' or type(record['seed']) is not int or record['seed']!=0:
            raise ValueError('AP8 requires completed seed0 references')
        art=root/run/'artifacts';cfg=json.loads((root/run/'config.json').read_text())
        if (cfg.get('version')!=(4 if kind=='cnn' else 3)
                or json_bytes(cfg.get('model'))!=json_bytes(wanted[kind])):
            raise ValueError('reference model policy differs from the fixed AP8 baseline')
        model_path=art/(kind+'-model.json')
        model=(load_cnn_model if kind=='cnn' else load_mlp_model)(model_path)
        body=json.loads(model_path.read_text())
        if json_bytes(body['training']['fit_settings'])!=json_bytes({**wanted[kind],'seed':0}):
            raise ValueError('reference fit settings do not match configuration')
        summary=json.loads((art/'summary.json').read_text())
        ledger=json.loads((art/(kind+'-backtest.json')).read_text())
        data=load_dataset(art/'validation-dataset.json')
        if (summary['test_metrics_computed'] is not False or model.report_id!=summary['model_report_id']
                or model.report_id!=ledger['strategy'].get('model_report_id')
                or data.dataset.dataset_id!=summary['validation_dataset_id']
                or data.manifest.artifact_id!=ledger['dataset_artifact_id']):
            raise ValueError('source model, ledger, validation or held-out binding mismatch')
        from decimal import Decimal
        profile=ledger['execution']
        template=asdict(ExecutionProfile())
        if set(profile)!=set(template) or any(type(profile[k]) is not bool for k in ('allow_short','liquidate_at_end')):
            raise ValueError('invalid source execution fields/types')
        profiles[kind]=ExecutionProfile(**{k:profile[k] if type(v) is bool else Decimal(profile[k])
                                         for k,v in template.items()})
        points[kind]=json.loads((art/(kind+'-validation-predictions.json')).read_text())
        models[kind]=model
        refs[kind]={'registry':str(root.resolve()),'run_id':run,'artifacts':str(art.resolve()),
                    'model_report_id':model.report_id,'settings':wanted[kind],
                    'manifest_sha256':record['manifest_sha256']}
    if profiles['cnn']!=profiles['mlp']:
        raise ValueError('reference execution conditions differ')
    cohort=prepare_image_cohort(Path(refs['cnn']['artifacts'])/'development-dataset.json',
                                Path(refs['cnn']['artifacts'])/'training-data.json')
    artifact,train,validation=_prepare(Path(refs['mlp']['artifacts'])/'training-data.json',configurable=True)
    if (tuple(train)!=cohort.training.train or tuple(validation)!=cohort.training.validation or cohort.training.test
            or any(not 2021<=r.decision_at.year<=r.target_available_at.year<=2023 for r in train)
            or any(r.decision_at.year!=2024 or r.target_available_at.year!=2024 for r in validation)):
        raise ValueError('source train/validation rows differ or reserved period reached')
    from market_trader.image_neural import _bindings
    from market_trader.neural import _TRAINING_METADATA
    cnn_body=json.loads((Path(refs['cnn']['artifacts'])/'cnn-model.json').read_text())
    mlp_body=json.loads((Path(refs['mlp']['artifacts'])/'mlp-model.json').read_text())
    if (any(json_bytes(cnn_body[k])!=json_bytes(v) for k,v in _bindings(cohort).items())
            or mlp_body['training_data_id']!=artifact['training_data_id']
            or any(json_bytes(mlp_body[k])!=json_bytes(artifact[k]) for k in _TRAINING_METADATA)):
        raise ValueError('source model/data binding does not match verified cohort')
    for kind,body in (('cnn',cnn_body),('mlp',mlp_body)):
        if (body['training']['row_count']!=len(train) or body['validation']['row_count']!=len(validation)
                or body['training']['first_decision_at']!=train[0].decision_at.isoformat()
                or body['training']['last_target_available_at']!=max(r.target_available_at for r in train).isoformat()):
            raise ValueError('source model training row binding mismatch')
    if (models['cnn'].config!=models['mlp'].config or models['cnn'].instrument!=models['mlp'].instrument
            or models['cnn'].instrument!='deuidxeur' or models['cnn'].interval!=models['mlp'].interval):
        raise ValueError('model target, instrument or clock differs')
    for kind in ('cnn','mlp'):
        pp=points[kind]
        if (type(pp) is not list or len(pp)!=len(validation)
                or any(type(p) is not dict or set(p)!={'at','predicted','actual'}
                    or p['at']!=r.decision_at.isoformat() or type(p['actual']) not in (int,float)
                    or type(p['predicted']) not in (int,float) or not math.isfinite(p['predicted'])
                    or p['actual']!=float(r.target_return) for p,r in zip(pp,validation,strict=True))):
            raise ValueError('saved forecasts do not match every source validation row')
    rows={'train':[asdict(r) for r in train],'validation':[asdict(r) for r in validation]}
    snapshot={'protocol':PROTOCOL,'references':refs,'training_rows':len(train),'validation_rows':len(validation),
              'all_rows_exact_equal':True,'cohort_rows_id':_digest(rows),'test_metrics_computed':False,
              'reserved_test_accessed':False,'execution':asdict(profiles['cnn']),
              'prediction_ids':{kind:_digest(pp) for kind,pp in points.items()}}
    snapshot=json.loads(json_bytes(snapshot,default=encode))
    return ReferencePair(snapshot,refs,cohort,points,models,profiles['cnn'],_digest(snapshot))


def run_request(model_kind,seed):
    if model_kind not in ('cnn','mlp') or type(seed) is not int or seed not in (1,2):
        raise ValueError('AP8A permits only cnn/mlp seeds 1 and 2')
    return {'protocol':PROTOCOL,'model_kind':model_kind,'seed':seed,
            'test_metrics_computed':False,'reserved_test_accessed':False}


def _check_pair(pair):
    from market_trader.experiment_registry import read_run
    if not isinstance(pair,ReferencePair) or _digest(pair.snapshot)!=pair.identity:
        raise ValueError('reference snapshot changed or invalid')
    if json_bytes(pair.references)!=json_bytes(pair.snapshot['references']):
        raise ValueError('reference lineage changed')
    for kind,ref in pair.references.items():
        if (_digest(pair.points[kind])!=pair.snapshot['prediction_ids'][kind]
                or read_run(Path(ref['registry']),ref['run_id'])['manifest_sha256']!=ref['manifest_sha256']):
            raise ValueError('reference source changed')


def _write(path,value):
    path.write_bytes(json_bytes(value,default=encode))


def _diagnostics(pair,kind,points,stage=None):
    from market_trader.robustness_analysis import forecast_slices,cost_backtests
    slices=forecast_slices(points);costs={};baseline=None
    expected=tuple(row.decision_at for row in pair.cohort.training.validation)
    source=Path(pair.references[kind]['artifacts'])
    model_id=(json.loads((stage/(kind+'-model.json')).read_text())['report_id']
              if stage is not None else pair.references[kind]['model_report_id'])
    for scenario,report,summary in cost_backtests(source/'validation-dataset.json',points,
            expected_decisions=expected,model_report_id=model_id,base_profile=pair.profile):
        costs[scenario]=summary
        if scenario=='baseline':baseline=report
        if stage is not None:_write(stage/('cost-'+scenario+'-backtest.json'),report)
    if set(costs)!={'half','baseline','double'} or baseline is None:
        raise ValueError('incomplete AP8 cost scenarios')
    return slices,costs,baseline


def _ledger_identity(report):
    # Only execution provenance and the strategy descriptor legitimately differ.
    # Preserve every ledger field: orders/fills, accounts, rejections and terminal
    # position/order state, not just completed-trade totals.
    return json_bytes({k:v for k,v in report.items() if k not in ('strategy','runtime')})


def reference_diagnostics(pair):
    """Reuse both seed0 models/forecasts; compare factor1 with original ledger."""
    from market_trader.reference import trading_summary
    _check_pair(pair);runs=[]
    for kind in ('cnn','mlp'):
        art=Path(pair.references[kind]['artifacts'])
        model=json.loads((art/(kind+'-model.json')).read_text())
        original=json.loads((art/(kind+'-backtest.json')).read_text())
        slices,costs,baseline=_diagnostics(pair,kind,pair.points[kind])
        if costs['baseline']!=trading_summary(original) or _ledger_identity(baseline)!=_ledger_identity(original):
            raise ValueError('factor1 frozen-forecast replay differs from original reference ledger')
        runs.append({'model_kind':kind,'seed':0,'reused':True,'run_id':pair.references[kind]['run_id'],
            'model_report_id':model['report_id'],'training_rows':pair.snapshot['training_rows'],
            'validation_rows':pair.snapshot['validation_rows'],'training':model['training'],
            'forecast':slices['overall'],'slices':slices,'costs':costs})
    _check_pair(pair)
    return runs


def execute_seed_fit(pair,*,model_kind,seed,output):
    """Exactly one explicit fit; bounded workers are the caller's responsibility."""
    request=run_request(model_kind,seed)
    _check_pair(pair)
    from tempfile import TemporaryDirectory
    import os
    from market_trader.image_dataset import _directory,_private,_publish_directory
    from market_trader.reference_pipeline import _runtime,_source_id
    output=Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError(str(output))
    with _directory(output.parent) as parent:_private(parent,output_parent=True)
    runtime=_runtime();source=Path(pair.references[model_kind]['artifacts'])
    with TemporaryDirectory(prefix='.market-trader-ap8-',dir=output.parent) as directory:
        holder=Path(directory);stage=holder/'bundle';stage.mkdir(mode=0o700)
        training_path=source/'training-data.json';model_path=stage/(model_kind+'-model.json')
        if model_kind=='cnn':
            from market_trader.image_neural_seeded import seeded_cnn_report,seeded_cnn_validation_evaluation,load_seeded_cnn_model
            with TemporaryDirectory(prefix='.inputs-',dir=stage) as cache:
                model=seeded_cnn_report(source/'development-dataset.json',training_path,seed=seed,cache_dir=Path(cache))
                _write(model_path,model);saved=load_seeded_cnn_model(model_path)
                evaluation,points=seeded_cnn_validation_evaluation(model_path,source/'development-dataset.json',training_path,cache_dir=Path(cache))
        else:
            from market_trader.neural import mlp_report,mlp_validation_evaluation,load_mlp_model
            model=mlp_report(training_path,seed=seed,settings=pair.references['mlp']['settings'])
            _write(model_path,model);saved=load_mlp_model(model_path)
            evaluation,points=mlp_validation_evaluation(model_path,training_path)
        if (type(model['training']['fit_settings'].get('seed')) is not int
                or model['training']['fit_settings']['seed']!=seed or saved.report_id!=model['report_id']
                or json_bytes(model['training']['fit_settings'])!=json_bytes({**pair.references[model_kind]['settings'],'seed':seed})):
            raise ValueError('actual trained model/seed/policy does not match fixed request')
        if ([(p['at'],p['actual']) for p in points]!=[(p['at'],p['actual']) for p in pair.points[model_kind]]):
            raise ValueError('new model forecast cohort differs from reference')
        _write(stage/'validation-predictions.json',points);_write(stage/'evaluation.json',evaluation)
        _write(stage/'source-inputs.json',pair.snapshot)
        slices,costs,baseline=_diagnostics(pair,model_kind,points,stage)
        _write(stage/'slices.json',slices);_write(stage/'training-history.json',model['training'])
        entry={'model_kind':model_kind,'seed':seed,'reused':False,'run_id':output.parent.name,
               'model_report_id':model['report_id'],'training_rows':pair.snapshot['training_rows'],
               'validation_rows':pair.snapshot['validation_rows'],'training':model['training'],
               'forecast':slices['overall'],'slices':slices,'costs':costs}
        signature={'policy':'market-trader.ap8-seed-comparison.v1','model_kind':model_kind,
                   'cohort_rows_id':pair.snapshot['cohort_rows_id'],'settings':pair.references[model_kind]['settings'],
                   'execution':pair.snapshot['execution'],'runtime':runtime}
        summary={'format':'market-trader.ap8-run.v1',**request,**entry,'runtime':runtime,
            'training_data_id':model['training_data_id'],'validation_dataset_id':baseline['dataset_id'],
            'comparison_signature':signature,'validation_forecast':evaluation['validation'],
            'trading':{model_kind:costs['baseline']},'source_reference':pair.references[model_kind],
            'reference_pair_id':pair.identity,'study_entry':entry}
        summary=json.loads(json_bytes(summary,default=encode));_write(stage/'summary.json',summary)
        _check_pair(pair)
        if _source_id()!=runtime['code_id']:raise ValueError('source changed during AP8 fit')
        files={str(p.relative_to(stage)):'sha256:'+sha256(p.read_bytes()).hexdigest()
               for p in sorted(stage.rglob('*')) if p.is_file()}
        _write(stage/'manifest.json',{'format':'market-trader.ap2-manifest.v1','status':'complete',
                                    'test_metrics_computed':False,'runtime':runtime,'artifacts':files})
        with _directory(output.parent) as parent:
            _private(parent,output_parent=True)
            holder_fd=os.open(holder,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            stage_fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:_publish_directory(holder,output,parent_fd=parent,stage_fd=stage_fd,container_fd=holder_fd)
            finally:os.close(stage_fd);os.close(holder_fd)
    return summary


def _canonical_evaluation(pair,kind,saved,model):
    """Validation-only, stdlib replay: one rendered window at a time, no cache."""
    from market_trader.baseline import _error_metrics,_TRAINING_METADATA
    from market_trader.image_neural import _seal
    rows=pair.cohort.training.validation
    points=[{'at':row.decision_at.isoformat(),
             'predicted':(saved.predict_window(pair.cohort.window(row),decision_at=row.decision_at)
                          if kind=='cnn' else saved.model.predict(row.features)),
             'actual':float(row.target_return)} for row in rows]
    metrics={'row_count':len(rows),kind:_error_metrics(tuple(p['predicted']-p['actual'] for p in points)),
             'zero_return':_error_metrics(tuple(p['actual'] for p in points))}
    metadata=({k:model[k] for k in ('training_data_id','input_index_id','source_training_data_id',
              'dataset_id','dataset_artifact_id','instrument','interval_microseconds','config','target_name','input_recipe')}
              if kind=='cnn' else {k:model[k] for k in ('training_data_id',*_TRAINING_METADATA)})
    evaluation={'format':('market-trader.cnn-seeded-validation-evaluation.v1' if kind=='cnn'
                           else 'market-trader.mlp-validation-evaluation.v1'),
                'model_report_id':saved.report_id,**metadata,
                'validation':{**metrics,'first_decision_at':points[0]['at'],'last_decision_at':points[-1]['at']}}
    if kind=='cnn':evaluation['model_kind']='cnn'
    return _seal(evaluation),points,metrics


def _fit_json(path,limit=8*1024*1024):
    """Bounded no-follow current-owner JSON; never block on a special file."""
    import os,stat
    from market_trader.experiment_registry import _decode
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1 or info.st_size>limit:
                raise ValueError('fit artifact must be bounded current-owner regular JSON')
            with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(limit+1)
            if len(raw)>limit:raise ValueError('fit artifact exceeds resource bound')
        finally:os.close(fd)
    except OSError as error:raise ValueError('missing or unreadable fit artifact: '+str(path)) from error
    return _decode(raw)


def _fit_inventory(registry,run_id):
    """Preflight BEFORE the general registry's otherwise unbounded reads.

    The private current owner is trusted not to replace files concurrently.
    Exact flat inventory also forbids unexpected directories/caches/PNGs.
    """
    import os,re,stat
    from market_trader.image_dataset import _directory,_private
    if type(run_id) is not str or re.fullmatch(r'run-[0-9a-f]{32}',run_id) is None:
        raise ValueError('invalid fit run identifier')
    run=registry/run_id;art=run/'artifacts'
    try:
        for directory in (registry,run,art):
            with _directory(directory) as fd:_private(fd)
        cfg=_fit_json(run/'config.json',1024*1024)
        _fit_json(run/'run.json',1024*1024)
        kind=cfg.get('model_kind');seed=cfg.get('seed')
        if json_bytes(cfg)!=json_bytes(run_request(kind,seed)):
            raise ValueError('fit configuration snapshot differs from fixed request')
        limits={name:8*1024*1024 for name in ('summary.json','validation-predictions.json','evaluation.json',
                'training-history.json','source-inputs.json','slices.json')}
        limits[kind+'-model.json']=2*1024*1024
        limits.update({f'cost-{s}-backtest.json':128*1024*1024 for s in ('half','baseline','double')})
        limits['manifest.json']=1024*1024
        if {p.name for p in art.iterdir()}!=set(limits):
            raise ValueError('incomplete or unexpected AP8 fit artifact inventory')
        for name,limit in limits.items():
            info=(art/name).lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid()
                    or info.st_nlink!=1 or info.st_size>limit):
                raise ValueError('fit artifact exceeds bounded regular current-owner contract')
        manifest=_fit_json(art/'manifest.json',1024*1024)
        if set(manifest.get('artifacts',{}))!=set(limits)-{'manifest.json'}:
            raise ValueError('manifest inventory differs from required fit inventory')
        return cfg,art,manifest
    except OSError as error:raise ValueError('missing or unreadable AP8 fit inventory') from error


def build_study(pair,registry,run_ids,output):
    """Assemble all four completed fits plus reused references; never optimize."""
    if (not isinstance(run_ids,(list,tuple)) or len(run_ids)!=4 or len(set(run_ids))!=4):
        raise ValueError('AP8 study requires exactly four distinct completed fit IDs')
    _check_pair(pair)
    from market_trader.experiment_registry import read_run,load_config_snapshot
    from market_trader.robustness_reporting import render_robustness_report
    from market_trader.robustness_analysis import forecast_slices,cost_backtests
    from market_trader.reference import trading_summary
    from market_trader.reference_pipeline import _runtime,_source_id
    from market_trader.image_dataset import _directory,_private,_publish_directory
    from tempfile import TemporaryDirectory
    registry=Path(registry);output=Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError(str(output))
    with _directory(output.parent) as parent:_private(parent,output_parent=True)
    runtime=_runtime();entries=[];seen=set();fit_sources=set()
    from market_trader.image_neural import _bindings,_training_rows
    from market_trader.neural import _TRAINING_METADATA
    mlp_source=_fit_json(Path(pair.references['mlp']['artifacts'])/'training-data.json',128*1024*1024)
    bindings={'cnn':_bindings(pair.cohort),
              'mlp':{k:mlp_source[k] for k in ('training_data_id',*_TRAINING_METADATA)}}
    for run_id in run_ids:
        cfg,art,manifest=_fit_inventory(registry,run_id)
        record=read_run(registry,run_id)
        if json_bytes(cfg)!=json_bytes(load_config_snapshot(registry,run_id)):
            raise ValueError('fit snapshot changed during verification')
        summary=_fit_json(art/'summary.json')
        if record['status']!='completed' or summary.get('format')!='market-trader.ap8-run.v1':
            raise ValueError('study requires completed AP8 fit records')
        kind,seed=cfg.get('model_kind'),cfg.get('seed');run_request(kind,seed)
        if (kind,seed) in seen:raise ValueError('duplicate model/seed in AP8 study')
        seen.add((kind,seed))
        if (cfg.get('protocol')!=PROTOCOL or summary['protocol']!=PROTOCOL
                or summary['model_kind']!=kind or type(summary['seed']) is not int or summary['seed']!=seed
                or summary['reference_pair_id']!=pair.identity or summary['test_metrics_computed'] is not False
                or summary['reserved_test_accessed'] is not False
                or json_bytes(summary['source_reference'])!=json_bytes(pair.references[kind])):
            raise ValueError('fit record/snapshot/source/held-out binding mismatch')
        if kind=='cnn':
            from market_trader.image_neural_seeded import load_seeded_cnn_model
            saved=load_seeded_cnn_model(art/'cnn-model.json')
        else:
            from market_trader.neural import load_mlp_model
            saved=load_mlp_model(art/'mlp-model.json')
        model=_fit_json(art/(kind+'-model.json'),2*1024*1024)
        points=_fit_json(art/'validation-predictions.json')
        signature={'policy':'market-trader.ap8-seed-comparison.v1','model_kind':kind,
                   'cohort_rows_id':pair.snapshot['cohort_rows_id'],'settings':pair.references[kind]['settings'],
                   'execution':pair.snapshot['execution'],'runtime':summary['runtime']}
        import re
        fit_runtime=summary['runtime']
        if (type(fit_runtime) is not dict or set(fit_runtime)!=set(runtime)
                or any(type(fit_runtime[k]) is not str or not 0<len(fit_runtime[k])<=128 for k in ('python','market_trader','code_id'))
                or re.fullmatch(r'sha256:[0-9a-f]{64}',fit_runtime['code_id']) is None
                or (fit_runtime['git_head'] is not None and (type(fit_runtime['git_head']) is not str
                    or re.fullmatch(r'[0-9a-f]{40}',fit_runtime['git_head']) is None))
                or (fit_runtime['git_dirty'] is not None and type(fit_runtime['git_dirty']) is not bool)
                or fit_runtime['python']!=model['training']['runtime_versions']['python']
                or json_bytes(summary['comparison_signature'])!=json_bytes(signature)
                or json_bytes(manifest)!=json_bytes({'format':'market-trader.ap2-manifest.v1','status':'complete',
                    'test_metrics_computed':False,'runtime':fit_runtime,'artifacts':manifest['artifacts']})
                or json_bytes(_fit_json(art/'source-inputs.json'))!=json_bytes(pair.snapshot)
                or json_bytes(_fit_json(art/'training-history.json'))!=json_bytes(model['training'])):
            raise ValueError('fit config/runtime/manifest/history/source snapshot binding mismatch')
        source_model=pair.models[kind]
        if (any(json_bytes(model.get(k))!=json_bytes(v) for k,v in bindings[kind].items())
                or any(json_bytes(model['training'].get(k))!=json_bytes(v) for k,v in _training_rows(pair.cohort).items())
                or model['validation']['row_count']!=len(pair.cohort.training.validation)
                or saved.config!=source_model.config or saved.instrument!=source_model.instrument
                or saved.interval!=source_model.interval or saved.target_name!=source_model.target_name
                or summary['training_data_id']!=bindings[kind]['training_data_id']):
            raise ValueError('new model source/cohort/training row binding mismatch')
        if (saved.report_id!=summary['model_report_id'] or model['training']['fit_settings']['seed']!=seed
                or json_bytes(model['training'])!=json_bytes(summary['training'])
                or json_bytes(model['training']['fit_settings'])!=json_bytes({**pair.references[kind]['settings'],'seed':seed})
                or [(p['at'],p['actual']) for p in points]!=[(p['at'],p['actual']) for p in pair.points[kind]]):
            raise ValueError('model/history/forecast binding mismatch')
        evaluation,canonical_points,metrics=_canonical_evaluation(pair,kind,saved,model)
        if json_bytes(points)!=json_bytes(canonical_points):
            raise ValueError('forecasts differ from canonical model reload on validation rows')
        if (json_bytes(model['validation'])!=json_bytes(metrics)
                or json_bytes(_fit_json(art/'evaluation.json'))!=json_bytes(evaluation)
                or json_bytes(summary['validation_forecast'])!=json_bytes(evaluation['validation'])):
            raise ValueError('model/evaluation/summary metrics differ from canonical evaluator')
        slices=forecast_slices(points)
        if (json_bytes(slices)!=json_bytes(summary['slices'])
                or json_bytes(_fit_json(art/'slices.json'))!=json_bytes(slices)):
            raise ValueError('stored diagnostic metrics differ from forecasts')
        costs={};baseline_dataset_id=None
        for scenario,replay,cost in cost_backtests(Path(pair.references[kind]['artifacts'])/'validation-dataset.json',
                points,expected_decisions=tuple(r.decision_at for r in pair.cohort.training.validation),
                model_report_id=saved.report_id,base_profile=pair.profile):
            ledger=_fit_json(art/('cost-'+scenario+'-backtest.json'),128*1024*1024)
            ledger_runtime={k:fit_runtime[k] for k in ('python','market_trader')}
            ledger_runtime['execution']='existing run_backtest; frozen forecasts; no fitting'
            if json_bytes(ledger.get('runtime'))!=json_bytes(ledger_runtime):
                raise ValueError('cost ledger declared runtime differs from fit runtime')
            # Unlike seed0 parity, strategy metadata MUST also match exactly.
            if json_bytes({k:v for k,v in ledger.items() if k!='runtime'})!=json_bytes({k:v for k,v in replay.items() if k!='runtime'}):
                raise ValueError('cost ledger differs from independent frozen-forecast replay')
            costs[scenario]=trading_summary(ledger)
            if costs[scenario]!=cost:raise ValueError('cost summary differs from ledger')
            if scenario=='baseline':baseline_dataset_id=replay['dataset_id']
            del ledger,replay
        if (json_bytes(summary['costs'])!=json_bytes(costs)
                or json_bytes(summary['trading'])!=json_bytes({kind:costs['baseline']})
                or baseline_dataset_id is None or summary['validation_dataset_id']!=baseline_dataset_id):
            raise ValueError('cost projection differs from replayed ledger')
        entry={'model_kind':kind,'seed':seed,'reused':False,'run_id':run_id,
               'model_report_id':saved.report_id,'training_rows':pair.snapshot['training_rows'],
               'validation_rows':pair.snapshot['validation_rows'],'training':model['training'],
               'forecast':slices['overall'],'slices':slices,'costs':costs}
        if (json_bytes(summary['study_entry'])!=json_bytes(entry)
                or any(json_bytes(summary.get(k))!=json_bytes(v) for k,v in entry.items())):
            raise ValueError('study projection does not match canonical fit record')
        canonical_summary={'format':'market-trader.ap8-run.v1',**run_request(kind,seed),**entry,
            'runtime':fit_runtime,'training_data_id':bindings[kind]['training_data_id'],
            'validation_dataset_id':baseline_dataset_id,'comparison_signature':signature,
            'validation_forecast':evaluation['validation'],'trading':{kind:costs['baseline']},
            'source_reference':pair.references[kind],'reference_pair_id':pair.identity,'study_entry':entry}
        if json_bytes(summary)!=json_bytes(canonical_summary):
            raise ValueError('top-level summary differs from complete canonical fit projection')
        entries.append(entry);fit_sources.add(summary['runtime']['code_id'])
    if seen!={(kind,seed) for kind in ('cnn','mlp') for seed in (1,2)} or len(fit_sources)!=1:
        raise ValueError('all four seeds must use one frozen fitting implementation')
    runs=reference_diagnostics(pair)+entries
    study={'format':'market-trader.ap8-study.v1','status':'complete','protocol':PROTOCOL,
        'test_metrics_computed':False,'reserved_test_accessed':False,
        'period':{'train':'2021-2023','validation':'2024','reserved':'2025'},
        'new_fit_count':4,'reference_reuse_count':2,'runs':runs,'reference_pair_id':pair.identity,
        'fit_code_id':next(iter(fit_sources)),'analysis_runtime':runtime,
        'limitations':['All 2024 results are checkpoint-selected development diagnostics, not independent test evidence.',
            '2025 remains closed; no rolling fits or final-test metrics were computed.',
            'Three seeds per family do not establish significance; no best seed is selected.',
            'For both CNN and MLP, seed changes weight initialization and minibatch shuffle order; this does not isolate initialization effects.',
            'CNN pixels and MLP numeric features are not information-equivalent.',
            'Time-of-day/calendar slices are not a learned market-regime classifier.',
            'Seed0 references retain their original code/runtime provenance and were not retrained.']}
    study=json.loads(json_bytes(study,default=encode));html=render_robustness_report(study)
    _check_pair(pair)
    if _source_id()!=runtime['code_id']:raise ValueError('source changed during AP8 analysis')
    with TemporaryDirectory(prefix='.market-trader-ap8-study-',dir=output.parent) as directory:
        holder=Path(directory);stage=holder/'bundle';stage.mkdir(mode=0o700)
        _write(stage/'study.json',study);(stage/'report.html').write_bytes(html)
        _write(stage/'manifest.json',{'format':'market-trader.ap8-study-manifest.v1','status':'complete',
             'artifacts':{p.name:'sha256:'+sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()}})
        with _directory(output.parent) as parent,_directory(holder) as holder_fd,_directory(stage) as stage_fd:
            _private(parent,output_parent=True)
            _publish_directory(holder,output,parent_fd=parent,stage_fd=stage_fd,container_fd=holder_fd)
    return study
