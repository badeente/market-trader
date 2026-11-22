"""AP7 v4 visual reference; existing import, registry and execution ledger reused.

This separate protocol preserves frozen v1-v3 behavior. It never scores 2025,
creates a full PNG corpus, or implicitly starts a parameter search.
"""
from dataclasses import asdict
from datetime import datetime,timedelta
from decimal import Decimal
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from market_trader._dataset_codec import json_bytes
from market_trader.artifacts import DatasetProvenance
from market_trader.backtesting import ExecutionProfile
from market_trader.baseline import baseline_report,validation_evaluation
from market_trader.dukascopy_import import import_dukascopy_reference,slice_session_dataset
from market_trader.experiment_config import experiment_config_from_payload,experiment_config_payload
from market_trader.experiment_pipeline import _label_checks
from market_trader.experiment_reporting import enrich_experiment_report
from market_trader.image_dataset import _directory,_private,_publish_directory
from market_trader.reference import _START,_VALIDATION,_TEST,_END,trading_summary
from market_trader.reference_pipeline import _assert_trade_alignment,_backtest,_json_value,_runtime,_source_id,_write_json
from market_trader.reporting import render_report
from market_trader.storage import load_dataset,publish_new_file,save_dataset
from market_trader.training_data import TrainingConfig,OPEN_TARGET_NAME,OPEN_TRAINING_DATA_FORMAT,build_open_training_data


def _framework_reload_parity(model_path, inputs, points):
    """Reload JSON weights into fixed Torch operators; never fit or update them."""
    import math
    import numpy as np
    import torch
    from market_trader.image_neural import load_cnn_model
    saved=load_cnn_model(model_path)
    count=inputs.row_count-inputs.train_count
    if len(points)!=count or inputs.path.stat().st_size!=inputs.row_count*2304*2:
        raise ValueError('framework parity input cohort/length mismatch')
    mapped=np.memmap(inputs.path,dtype='<u2',mode='r',shape=(inputs.row_count,2304))
    threads=torch.get_num_threads()
    deterministic=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    maximum=0.0
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            network=torch.nn.Sequential(torch.nn.Conv2d(3,4,3,dtype=torch.float64),torch.nn.ReLU(),
                torch.nn.AvgPool2d(2),torch.nn.Flatten(),torch.nn.Linear(660,8,dtype=torch.float64),
                torch.nn.ReLU(),torch.nn.Linear(8,1,dtype=torch.float64))
            names=('0.weight','0.bias','4.weight','4.bias','6.weight','6.bias')
            shapes=((4,3,3,3),(4,),(8,660),(8,),(1,8),(1,))
            network.load_state_dict({name:torch.tensor(values,dtype=torch.float64).reshape(shape)
                for name,shape,values in zip(names,shapes,saved.model.tensors,strict=True)})
            network.eval()
            for start in range(0,count,128):
                raw=np.array(mapped[inputs.train_count+start:inputs.train_count+min(count,start+128)],dtype=np.float64,copy=True)
                if np.any(raw>16320):raise ValueError('invalid image sums in parity input')
                batch=torch.from_numpy(raw).reshape(-1,3,24,32)/64/255
                predictions=(network(batch).reshape(-1)*saved.model.target_scale+saved.model.target_mean).tolist()
                for predicted,point in zip(predictions,points[start:start+128],strict=True):
                    if (type(point['predicted']) not in (int,float) or not math.isfinite(point['predicted'])
                            or not math.isfinite(predicted)):
                        raise ValueError('nonfinite framework parity prediction')
                    maximum=max(maximum,abs(predicted-point['predicted']))
    finally:
        del mapped
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)
    if maximum>1e-12:raise ValueError('CNN framework/stdlib reload parity exceeded tolerance')
    return {'format':'market-trader.cnn-reload-parity.v1','passed':True,'model_report_id':saved.report_id,
            'validation_rows':count,'max_absolute_error':maximum,'tolerance':1e-12}


def execute_image_experiment(config,output):
    """Run the fixed visual model only; publish into an existing private parent."""
    from market_trader.image_neural import cnn_report,load_cnn_model,cnn_validation_evaluation
    from market_trader.image_training_data import prepare_image_cohort,build_input_cache
    from market_trader.image_trade_audit import build_trade_image_audit,render_trade_image_audit
    output=Path(output)
    config=experiment_config_from_payload(experiment_config_payload(config))
    if config.version!=4 or config.model_kind!='cnn':
        raise ValueError('visual pipeline requires the explicit v4 CNN protocol')
    if output.exists() or output.is_symlink():raise FileExistsError(str(output))
    with _directory(output.parent) as parent:
        _private(parent,output_parent=True)
    snapshot=experiment_config_payload(config)
    runtime=_runtime()
    runtime['distributions']={name:version(name) for name in ('torch','numpy')}
    template=Path(__file__).with_name('report_template.html.in')
    runtime['report_template_sha256']='sha256:'+sha256(template.read_bytes()).hexdigest()
    lock=Path(__file__).resolve().parents[2]/'requirements/ap3-cpu-lock.txt'
    if lock.is_file():runtime['dependency_lock_sha256']='sha256:'+sha256(lock.read_bytes()).hexdigest()
    imported=import_dukascopy_reference(config.source_csv,instrument='deuidxeur',start=_START,end=_END,
                                       expected_sha256=config.source_sha256)
    development=slice_session_dataset(imported.dataset,_START,_TEST)
    validation=slice_session_dataset(imported.dataset,_VALIDATION,_TEST)
    if any(bar.close_time>_TEST or bar.available_at>=_TEST for bar in development.candles):
        raise ValueError('reserved candles reached AP7 development')
    training=build_open_training_data(development,TrainingConfig(8,2,_VALIDATION,_TEST),include_test=False)
    if training.test:raise ValueError('AP7 test split must remain empty')
    profile=ExecutionProfile(initial_capital=config.initial_capital,spread=config.spread,slippage=config.slippage,
        price_tick=config.price_tick,quantity_step=Decimal(1),commission_rate=config.commission_rate,
        allow_short=False,liquidate_at_end=False)
    with _directory(output.parent) as parent:
        _private(parent,output_parent=True)
        with TemporaryDirectory(prefix='.market-trader-ap7-',dir=output.parent) as directory:
            holder=Path(directory);stage=holder/'bundle';stage.mkdir(mode=0o700)
            origin=DatasetProvenance('Dukascopy DEUIDXEUR bid; complete-session research filter',(imported.source,))
            full=save_dataset(stage/'research-dataset.json',imported.dataset,provenance=origin)
            dev_origin=DatasetProvenance('AP7 development excludes reserved 2025',(imported.source,),(full.dataset_id,))
            dev_manifest=save_dataset(stage/'development-dataset.json',development,provenance=dev_origin)
            val_origin=DatasetProvenance('AP7 validation 2024; development selection only',(imported.source,),(dev_manifest.dataset_id,))
            val_manifest=save_dataset(stage/'validation-dataset.json',validation,provenance=val_origin)
            for name in ('research','development','validation'):load_dataset(stage/f'{name}-dataset.json')
            _write_json(stage/'config.json',snapshot)
            _write_json(stage/'import-audit.json',imported.audit)
            _write_json(stage/'label-checks.json',_label_checks(training,development))
            body={'format':OPEN_TRAINING_DATA_FORMAT,'dataset_id':dev_manifest.dataset_id,
                  'dataset_artifact_id':dev_manifest.artifact_id,'instrument':development.instrument,
                  'interval_microseconds':development.interval//timedelta(microseconds=1),
                  'config':asdict(training.config),'feature_names':training.feature_names,'target_name':OPEN_TARGET_NAME,
                  'splits':{'train':[asdict(r) for r in training.train],
                            'validation':[asdict(r) for r in training.validation],'test':[]}}
            training_id='sha256:'+sha256(json_bytes(body,default=_json_value)).hexdigest()
            training_path=stage/'training-data.json'
            _write_json(training_path,{**body,'training_data_id':training_id})
            ridge_path=stage/'ridge-model.json';_write_json(ridge_path,baseline_report(training_path))
            ridge_eval,ridge_points=validation_evaluation(ridge_path,training_path)
            _write_json(stage/'ridge-validation-evaluation.json',ridge_eval)
            _write_json(stage/'ridge-validation-predictions.json',ridge_points)
            cohort=prepare_image_cohort(stage/'development-dataset.json',training_path)
            _write_json(stage/'image-index.json',cohort.index_document())
            _write_json(stage/'image-labels.json',cohort.labels_document())
            with TemporaryDirectory(prefix='.image-cache-',dir=stage) as cache:
                cache_path=Path(cache)
                inputs=build_input_cache(cohort,cache_path,accelerated=True)
                model=cnn_report(stage/'development-dataset.json',training_path,seed=config.seed,
                                 settings=snapshot['model'],cache_dir=cache_path)
                model_path=stage/'cnn-model.json';_write_json(model_path,model)
                saved=load_cnn_model(model_path)
                if saved.target_name!=OPEN_TARGET_NAME:raise ValueError('CNN target mismatch')
                evaluation,points=cnn_validation_evaluation(model_path,stage/'development-dataset.json',training_path,
                                                            cache_dir=cache_path)
                if ([(p['at'],p['actual']) for p in points] != [(p['at'],p['actual']) for p in ridge_points]
                        or any(not _VALIDATION<=datetime.fromisoformat(p['at'])<_TEST for p in points)):
                    raise ValueError('CNN/Ridge validation cohort mismatch or reserved forecast')
                parity_evidence=_framework_reload_parity(model_path,inputs,points)
                cache_evidence={'format':'market-trader.image-input-evidence.v1','input_index_id':inputs.input_index_id,
                                'row_count':inputs.row_count,'train_count':inputs.train_count,'sha256':inputs.sha256,
                                'temporary_cache_removed_before_publication':True}
            _write_json(stage/'cnn-reload-parity.json',parity_evidence)
            _write_json(stage/'image-input-evidence.json',cache_evidence)
            _write_json(stage/'cnn-validation-evaluation.json',evaluation)
            _write_json(stage/'cnn-validation-predictions.json',points)
            _write_json(stage/'validation-evaluation.json',evaluation)
            _write_json(stage/'validation-predictions.json',points)
            _write_json(stage/'training-history.json',{'format':'market-trader.cnn-training-history.v1',
                        'model_report_id':model['report_id'],'training_data_id':model['training_data_id'],**model['training']})
            reports={}
            for name,strategy in [('ridge',{'name':'ridge_long','model_report':'ridge-model.json'}),
                                  ('cnn',{'name':'cnn_long','model_report':'cnn-model.json'})]:
                reports[name]=_backtest(stage,name,{**strategy,'entry_threshold':config.entry_threshold,'quantity':Decimal(1)},
                                        validation,val_manifest,val_origin,runtime,profile)
            reports['sma_timed']=_backtest(stage,'sma',{'name':'sma_timed','fast_period':4,'slow_period':8,
                    'holding_bars':2,'quantity':Decimal(1)},validation,val_manifest,val_origin,runtime,profile)
            for report in reports.values():_assert_trade_alignment(report,training.validation)
            signature={'policy':'market-trader.ap7-comparison.v1','source_sha256':imported.source.sha256,
                       'development_dataset_id':development.dataset_id,'validation_dataset_id':validation.dataset_id,
                       'source_training_data_id':training_id,'input_index_id':model['training_data_id'],
                       'input_recipe':cohort.index_document()['input_recipe'],'target_name':OPEN_TARGET_NAME,
                       'lookback_bars':8,'horizon_bars':2,'interval_microseconds':900000000,
                       'splits':{'train_start':_START.isoformat(),'validation_start':_VALIDATION.isoformat(),
                                 'test_start':_TEST.isoformat(),'test_end':_END.isoformat()},
                       'execution':asdict(profile),'quantity':'1','contract_multiplier':'1',
                       'reference_strategy':{'sma_fast_period':4,'sma_slow_period':8},
                       'representation_policy':{'cnn':'AP6 RGB -> fixed 8x8 block averages -> CHW3x24x32',
                                                'ridge_features':list(training.feature_names),'information_equivalent':False},
                       'runtime':runtime}
            summary={'format':'market-trader.ap7-summary.v1','name':config.name,'hypothesis':config.hypothesis,
                     'seed':config.seed,'config':snapshot,'training_rows':len(training.train),
                     'validation_rows':len(training.validation),'test_metrics_computed':False,
                     'reserved_test_start':_TEST.isoformat(),'reserved_test_end':_END.isoformat(),
                     'target_name':OPEN_TARGET_NAME,'training_data_id':model['training_data_id'],
                     'source_training_data_id':training_id,'model_report_id':model['report_id'],
                     'development_dataset_id':development.dataset_id,'validation_dataset_id':validation.dataset_id,
                     'source_sha256':imported.source.sha256,'runtime':runtime,'comparison_signature':signature,
                     'training':model['training'],'validation_forecast':{
                         'cnn':evaluation['validation']['cnn'],'ridge':ridge_eval['validation']['ridge'],
                         'zero':evaluation['validation']['zero_return'],'zero_return':evaluation['validation']['zero_return'],
                         **{k:evaluation['validation'][k] for k in ('row_count','first_decision_at','last_decision_at')}},
                     'trading':{name:trading_summary(report) for name,report in reports.items()}}
            summary=json.loads(json_bytes(summary,default=_json_value));_write_json(stage/'summary.json',summary)
            audit=build_trade_image_audit(stage/'cnn-backtest.json',model_path,stage/'validation-dataset.json',
                                          stage/'cnn-validation-predictions.json',limit=8)
            _write_json(stage/'trade-image-audit.json',audit)
            report=render_report(stage/'cnn-backtest.json',compare=stage/'sma-backtest.json',
                                 model=model_path,dataset=stage/'validation-dataset.json')
            report=enrich_experiment_report(report,summary=summary)
            report=report.replace(b'</body>',render_trade_image_audit(audit).encode()+b'</body>')
            if len(report)>32*1024*1024:raise ValueError('AP7 report exceeds 32MiB')
            publish_new_file(stage/'report.html',report)
            if (_source_id()!=runtime['code_id'] or runtime['report_template_sha256']!='sha256:'+sha256(template.read_bytes()).hexdigest()):
                raise ValueError('source changed during experiment; retry from a stable checkout')
            files={str(path.relative_to(stage)):'sha256:'+sha256(path.read_bytes()).hexdigest()
                   for path in sorted(stage.rglob('*')) if path.is_file()}
            _write_json(stage/'manifest.json',{'format':'market-trader.ap2-manifest.v1','status':'complete',
                                             'test_metrics_computed':False,'runtime':runtime,'artifacts':files})
            holder_fd=os.open(holder,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            stage_fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:_publish_directory(holder,output,parent_fd=parent,stage_fd=stage_fd,container_fd=holder_fd)
            finally:os.close(stage_fd);os.close(holder_fd)
    return summary
