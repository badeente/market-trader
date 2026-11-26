"""AP8A HTML contract: synthetic dictionaries only; never fit or read artifacts."""
import copy
import importlib
import unittest


def study_fixture() -> dict:
    def metrics(n=2):
        return dict(row_count=n, **{k: (0.25 if n else None) for k in ('mae','rmse','zero_mae','zero_rmse')})
    months = [dict(key=f'2024-{i:02}', **metrics(2 if i == 1 else 0)) for i in range(1,13)]
    quarters = [dict(key=f'2024-Q{i}', **metrics(2 if i == 1 else 0)) for i in range(1,5)]
    sessions = [dict(key=k, **metrics(2 if i == 0 else 0)) for i,k in enumerate(('09:00-12:00','12:00-15:00','15:00-17:30'))]
    costs = dict(trade_count=1, net_points='-1', reference_points='1', modeled_cost_points='2',
                 max_drawdown_points='1', max_drawdown_fraction='-0.01', win_fraction='0',
                 months=[dict(month=f'2024-{i:02}', trade_count=int(i==1), net_points='-1' if i==1 else '0',
                              reference_points='1' if i==1 else '0', modeled_cost_points='2' if i==1 else '0') for i in range(1,13)])
    runs = []
    for family in ('cnn','mlp'):
        for seed in range(3):
            runs.append(dict(model_kind=family, seed=seed, reused=seed==0, run_id=f'{family}-run-{seed}',
                             model_report_id=f'{family}-model-{seed}', training_rows=10, validation_rows=2,
                             training=dict(epoch_history=[dict(epoch=i, training_mse=1/i, validation_mse=1/i) for i in range(1,21)],
                                           best_epoch=20, stopping_reason='max_epochs'),
                             forecast=metrics(), slices=dict(format='market-trader.ap8-slices.v1',timezone='Europe/Berlin',
                                 overall=metrics(),months=copy.deepcopy(months),quarters=copy.deepcopy(quarters),sessions=copy.deepcopy(sessions)),
                             costs={k:copy.deepcopy(costs) for k in ('half','baseline','double')}))
    return dict(format='market-trader.ap8-study.v1',status='complete', protocol='ap8a-development-robustness-v1',
                test_metrics_computed=False,reserved_test_accessed=False,
                period=dict(train='2021-2023',validation='2024',reserved='2025'),new_fit_count=4,reference_reuse_count=2,
                runs=runs,limitations=['Synthetic fixture; not a fitted result.'])


class RobustnessReportingTests(unittest.TestCase):
    def render(self, study=None):
        module = importlib.import_module('market_trader.robustness_reporting')
        return module.render_robustness_report(study_fixture() if study is None else study)

    def test_refuses_incomplete_or_contradictory_study_identity(self):
        mutations = [
            lambda s:s.update(format='wrong'), lambda s:s.update(status='failed'),
            lambda s:s.update(protocol='other'),lambda s:s.update(test_metrics_computed=True),
            lambda s:s.update(reserved_test_accessed=0),lambda s:s['period'].update(validation='2025'),
            lambda s:s.update(new_fit_count=True),lambda s:s.update(reference_reuse_count=3),
            lambda s:s['runs'].pop(),lambda s:s['runs'].append(copy.deepcopy(s['runs'][0])),
            lambda s:s['runs'][0].update(seed=True),lambda s:s['runs'][0].update(model_kind='rnn'),
            lambda s:s['runs'][0].update(reused=False),lambda s:s['runs'][1].update(reused=True),
            lambda s:s['runs'][1].update(seed=0),lambda s:s['runs'][0].update(training_rows=True),
            lambda s:s['runs'][0].update(validation_rows=0),lambda s:s['runs'][0].update(run_id=''),
            lambda s:s['runs'][0].update(model_report_id=None),lambda s:s.pop('limitations'),
            lambda s:s.update(limitations=[False]),lambda s:s['runs'][0].pop('costs'),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                s=study_fixture();mutate(s)
                with self.assertRaises(ValueError): self.render(s)

    def test_metrics_slices_and_cost_totals_must_be_coherent_and_finite(self):
        mutations = [
            lambda r:r['forecast'].update(mae=True),lambda r:r['forecast'].update(rmse=float('nan')),
            lambda r:r['forecast'].update(zero_mae=float('inf')),lambda r:r['forecast'].update(mae=-1),
            lambda r:r['forecast'].update(mae='0.25'),lambda r:r['forecast'].update(row_count=3),
            lambda r:r['slices'].update(format='wrong'),lambda r:r['slices'].update(timezone='UTC'),
            lambda r:r['slices']['overall'].update(mae=1),lambda r:r['slices']['months'].pop(),
            lambda r:r['slices']['months'][0].update(key='2025-01'),
            lambda r:r['slices']['months'][1].update(mae=0),
            lambda r:r['slices']['months'][0].update(row_count=True),
            lambda r:r['slices']['quarters'][1].update(row_count=2),
            lambda r:r['slices']['sessions'][0].update(key='other'),
            lambda r:r['costs'].pop('double'),lambda r:r['costs']['baseline'].update(trade_count=True),
            lambda r:r['costs']['baseline'].update(net_points='NaN'),
            lambda r:r['costs']['baseline'].update(modeled_cost_points=True),
            lambda r:r['costs']['baseline'].update(win_fraction='2'),
            lambda r:r['costs']['baseline'].update(net_points='3'),
            lambda r:r['costs']['baseline']['months'].pop(),
            lambda r:r['costs']['baseline']['months'][0].update(trade_count=2),
            lambda r:r['costs']['baseline']['months'][0].update(net_points='999'),
            lambda r:r['costs']['baseline']['months'][0].update(month='2025-01'),
            lambda r:r['costs']['baseline'].update(max_drawdown_points='-1'),
            lambda r:r['costs']['baseline'].update(modeled_cost_points='-1'),
            lambda r:r['costs']['baseline']['months'][0].update(modeled_cost_points='-1'),
        ]
        for i,mutate in enumerate(mutations):
            with self.subTest(case=i):
                s=study_fixture();mutate(s['runs'][0])
                with self.assertRaises(ValueError): self.render(s)

    def test_shared_ledger_signed_drawdown_renders_unchanged(self):
        from dataclasses import asdict
        from datetime import UTC, datetime, timedelta
        from decimal import Decimal as D
        import json
        from market_trader.backtesting import Action, Signal, run_backtest
        from market_trader.domain import Candle
        from market_trader.normalization import NormalizationConfig, normalize_candles
        from market_trader.reference import trading_summary
        from market_trader.reference_pipeline import _json_value

        start = datetime(2024, 1, 2, 8, tzinfo=UTC)
        interval = timedelta(minutes=15)
        candles = tuple(Candle('deuidxeur', interval, start+i*interval,
                               start+(i+1)*interval, start+(i+1)*interval,
                               D(price), D(price)+1, D(price)-1, D(price), D(1))
                        for i, price in enumerate((100, 100, 90)))
        data, = normalize_candles(candles, config=NormalizationConfig(calendar='continuous')).datasets
        signals = iter((Signal(Action.ENTER_LONG), Signal(Action.EXIT)))
        result = run_backtest(data, lambda decision: next(signals, Signal(Action.HOLD)))
        report = json.loads(json.dumps(asdict(result), default=_json_value))
        report['execution'] = report.pop('profile')
        summary = trading_summary(report)
        self.assertEqual(summary['max_drawdown_fraction'], str(result.max_drawdown))
        self.assertLess(D(summary['max_drawdown_fraction']), 0)
        self.assertGreater(D(summary['max_drawdown_points']), 0)
        self.assertEqual(summary['trade_count'], 1)
        self.assertEqual(sum(m['trade_count'] for m in summary['months']), 1)
        study = study_fixture()
        for run in study['runs']:
            run['costs'] = {key: copy.deepcopy(summary) for key in ('half', 'baseline', 'double')}
        before = copy.deepcopy(study)
        self.assertIn(summary['max_drawdown_fraction'], self.render(study).decode())
        self.assertEqual(study, before)

    def test_positive_drawdown_fraction_is_rejected(self):
        study = study_fixture()
        for run in study['runs']:
            for cost in run['costs'].values():
                cost['max_drawdown_fraction'] = '0'
        study['runs'][0]['costs']['half']['max_drawdown_fraction'] = '0.01'
        with self.assertRaises(ValueError):
            self.render(study)

    def test_actual_history_checkpoint_and_stopping_policy_are_validated(self):
        mutations = [
            lambda t:t.update(epoch_history=[]), lambda t:t.update(epoch_history=t['epoch_history']*1000),
            lambda t:t.update(best_epoch=True),lambda t:t.update(best_epoch=1),
            lambda t:t['epoch_history'][0].update(epoch=True),lambda t:t['epoch_history'][1].update(epoch=3),
            lambda t:t['epoch_history'][0].update(training_mse=False),
            lambda t:t['epoch_history'][0].update(validation_mse=float('inf')),
            lambda t:t['epoch_history'][0].update(training_mse=-1),
            lambda t:t['epoch_history'][0].update(train_mse=999),
            lambda t:t['epoch_history'][0].pop('training_mse'),
            lambda t:t.update(stopping_reason='claimed_success'),
            lambda t:t.update(stopping_reason='early_stopping'),
            lambda t:t['epoch_history'].pop(),
            lambda t:t.update(fit_settings=dict(max_epochs=20,patience=True,min_delta='0.000001')),
        ]
        for i,mutate in enumerate(mutations):
            with self.subTest(case=i):
                s=study_fixture();mutate(s['runs'][0]['training'])
                with self.assertRaises(ValueError): self.render(s)
        # A raw minimum below min_delta is still the selected checkpoint.
        s=study_fixture()
        t=s['runs'][0]['training']
        t['epoch_history']=[dict(epoch=i,train_mse=1,validation_mse=(1 if i==1 else 1-1e-8)) for i in range(1,7)]
        t.update(best_epoch=2,stopping_reason='early_stopping')
        self.assertIn('Best checkpoint: 2',self.render(s).decode())
        t['epoch_history'].append(dict(epoch=7,train_mse=1,validation_mse=1))
        with self.assertRaises(ValueError): self.render(s)

    def test_bounded_escaped_standalone_render_and_extreme_finite_mean(self):
        from html import escape
        from unittest.mock import patch
        import socket
        s=study_fixture()
        hostile='</summary><script>alert("x")</script><img src=x onerror=boom>&'
        s['runs'][0]['run_id']=hostile
        s['runs'][0]['model_report_id']=hostile
        s['limitations']=[hostile]
        # Unknown metadata is neither expanded nor interpreted as markup.
        s['unknown_extra']={'adversarial':object()}
        for r in s['runs']:
            for container in (r['forecast'],r['slices']['overall'],r['slices']['months'][0],
                              r['slices']['quarters'][0],r['slices']['sessions'][0]):
                container.update({k:1e308 for k in ('mae','rmse','zero_mae','zero_rmse')})
        before=copy.deepcopy(s['runs'])
        with patch('builtins.open',side_effect=AssertionError('file access')), patch.object(socket,'socket',side_effect=AssertionError('network')):
            text=self.render(s).decode()
        self.assertIn(escape(hostile,quote=True),text)
        self.assertNotIn('<script',text)
        self.assertNotIn('<img',text)
        self.assertEqual(s['runs'],before)
        self.assertEqual(text,self.render(s).decode())
        for field,value in [('run_id','x'*4097),('run_id','\ud800')]:
            s=study_fixture();s['runs'][0][field]=value
            with self.assertRaises(ValueError): self.render(s)
        for key,value in [('limitations',['x']*101),('runs',[{}]*100000)]:
            s=study_fixture();s[key]=value
            with self.assertRaises(ValueError): self.render(s)

    def test_zero_trade_none_and_cost_descriptive_ranges_are_explicit(self):
        s=study_fixture()
        for r in s['runs']:
            for cost in r['costs'].values():
                cost.update(trade_count=0,win_fraction=None,net_points='0',reference_points='0',modeled_cost_points='0',max_drawdown_points='0',max_drawdown_fraction='0')
                for m in cost['months']:
                    m.update(trade_count=0,net_points='0',reference_points='0',modeled_cost_points='0')
        text=self.render(s).decode()
        self.assertIn('None (no trades)',text)
        self.assertIn('Cost descriptive range / mean',text)
        self.assertIn('CNN: max 20 epochs, patience 5',text)
        self.assertIn('MLP: max 20 epochs, patience 10',text)

    def test_failed_run_cannot_be_presented_as_complete(self):
        s=study_fixture();s['runs'][0]['status']='failed'
        with self.assertRaises(ValueError): self.render(s)

    def test_empty_limitations_are_explicit(self):
        s=study_fixture();s['limitations']=[]
        self.assertIn('None declared',self.render(s).decode())

    def test_decimal_context_cannot_change_rendering_or_validation(self):
        from decimal import localcontext, Inexact
        s=study_fixture()
        for r in s['runs']:
            for container in (r['forecast'],r['slices']['overall'],r['slices']['months'][0],
                              r['slices']['quarters'][0],r['slices']['sessions'][0]):
                container.update({k:(1,2,4)[r['seed']] for k in ('mae','rmse','zero_mae','zero_rmse')})
        expected=self.render(s)
        with localcontext() as ctx:
            ctx.prec=2;ctx.Emax=2;ctx.traps[Inexact]=True
            self.assertEqual(self.render(s),expected)

    def test_all_supplied_rows_are_rendered_and_bad_seed_affects_range(self):
        from html import unescape
        import re
        from decimal import Decimal
        s=study_fixture()
        for r in s['runs']:
            error=(0.1,0.2,0.9)[r['seed']]
            for m in (r['forecast'],r['slices']['overall'],r['slices']['months'][0],r['slices']['quarters'][0],r['slices']['sessions'][0]):
                m.update(mae=error,rmse=error)
            net=(-1,-3,-11)[r['seed']]
            for cost in r['costs'].values():
                cost.update(net_points=str(net),modeled_cost_points=str(1-net))
                cost['months'][0].update(net_points=str(net),modeled_cost_points=str(1-net))
        text=self.render(s).decode()
        tables=[]
        for caption,body in re.findall(r'<table(?: class="ap8-individual")?><caption>(.*?)</caption>.*?<tbody>(.*?)</tbody></table>',text):
            rows=[[unescape(x) for x in re.findall(r'<td>(.*?)</td>',row)] for row in re.findall(r'<tr>(.*?)</tr>',body)]
            tables.append((unescape(caption),rows))
        expected_per_run=[20,3,12,12,12,12,4,3]
        expected=[3,4,21]+expected_per_run*3
        self.assertEqual([len(rows) for _,rows in tables],expected*2)
        for caption, rows in tables:
            if caption.startswith('Descriptive range'):
                self.assertEqual(rows[0][1:3],['0.1','0.9'])
                self.assertEqual(Decimal(rows[0][3]),Decimal('0.4'))
            if caption.startswith('Cost descriptive'):
                netrow=next(row for row in rows if row[:2]==['baseline','net_points'])
                self.assertEqual(netrow[2:],['3','-11','-1','-5'])
        self.assertEqual(text.count('<details>'),len(s['runs']))
        self.assertEqual(text.count('<table'),text.count('class="table-scroll"'))

    def test_maximum_escaped_text_stays_bounded_and_size_guard_is_real(self):
        from unittest.mock import patch
        from market_trader import robustness_reporting
        s=study_fixture();s['limitations']=['"'*4096]*100
        for r in s['runs']:
            r['run_id']='"'*4096;r['model_report_id']='"'*4096
        self.assertLessEqual(len(self.render(s)),8*1024*1024)
        with patch.object(robustness_reporting,'_MAX_BYTES',1):
            with self.assertRaisesRegex(ValueError,'8 MiB'): self.render()
        s=study_fixture();s['runs'][0]['training']['epoch_history']=[{}]*100000
        with patch.object(robustness_reporting,'_table',side_effect=AssertionError('expanded before validation')):
            with self.assertRaises(ValueError): self.render(s)

    def test_framework_free_in_clean_interpreter(self):
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        src = root / 'src'
        reporter = src / 'market_trader' / 'robustness_reporting.py'
        self.assertTrue(reporter.is_file())
        # -I ignores inherited PYTHONPATH; -S disables site and editable .pth
        # handling. Add only the verified repository roots, never site-packages.
        code = f"""
import importlib.util
from pathlib import Path
import sys
assert sys.flags.no_site and sys.flags.isolated
sys.path[:0] = [{str(src)!r}, {str(root)!r}]
frameworks = ('torch', 'numpy', 'pandas')
assert all(importlib.util.find_spec(name) is None for name in frameworks)
from tests.test_robustness_reporting import study_fixture
from market_trader import robustness_reporting
assert Path(robustness_reporting.__file__).resolve() == Path({str(reporter)!r})
b = robustness_reporting.render_robustness_report(study_fixture())
assert not any(name.split('.')[0] in frameworks for name in sys.modules)
assert all(importlib.util.find_spec(name) is None for name in frameworks)
print(len(b))
"""
        result=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertGreater(int(result.stdout),0)

    def test_mobile_tables_preserve_readable_columns_inside_scroll_regions(self):
        import re
        text = self.render().decode()
        style = re.search(r'<style>(.*?)</style>', text, re.S)
        assert style is not None
        css = style.group(1)
        rules = {selector.strip(): dict(re.findall(r'([\w-]+):([^;}]+)', declarations))
                 for selector, declarations in re.findall(r'([^{}]+)\{([^{}]*)\}', css)}
        for property_name, value in dict(display='block', **{'min-width': '0',
                'max-width': '100%', 'overflow-x': 'auto'}).items():
            self.assertEqual(rules['.table-scroll'].get(property_name), value)
        self.assertEqual(rules['.table-scroll table']['min-width'], '55rem')
        self.assertEqual(rules['.table-scroll table']['width'], '100%')
        self.assertEqual(rules['.table-scroll th']['min-width'], '7rem')
        for selector, whitespace in (('.table-scroll th', 'normal'), ('.table-scroll td', 'nowrap')):
            self.assertEqual(rules[selector]['white-space'], whitespace)
            self.assertEqual(rules[selector]['word-break'], 'normal')
            self.assertEqual(rules[selector]['overflow-wrap'], 'normal')
        regions = re.findall(r'<div class="table-scroll" role="region" tabindex="0" aria-label="([^"]+)">(<table.*?</table>)</div>', text)
        self.assertEqual(len(regions), text.count('<table'))
        for label, table in regions:
            self.assertIn('<caption>' + label + '</caption>', table)
            self.assertIn('<th scope="col">', table)
        for preview_only in ('ap8-preview-note', 'ap8-source-report-sha256', 'Derived presentation', 'Darstellungsvorschau'):
            self.assertNotIn(preview_only, text)

    def test_only_forecast_identifier_columns_wrap_without_splitting_numbers(self):
        import re
        from html import escape
        study = study_fixture()
        study['runs'][0]['run_id'] = 'run-' + 'a' * 128 + '<&'
        study['runs'][0]['model_report_id'] = 'sha256:' + 'b' * 128
        before = copy.deepcopy(study)
        text = self.render(study).decode()
        tables = re.findall(r'<table([^>]*)><caption>(.*?)</caption>', text)
        individual = [attributes for attributes, caption in tables if caption == 'Individual forecasts']
        self.assertEqual(len(individual), 2)
        self.assertTrue(all(attributes == ' class="ap8-individual"' for attributes in individual))
        self.assertTrue(all(attributes == '' for attributes, caption in tables if caption != 'Individual forecasts'))
        style = re.search(r'<style>(.*?)</style>', text, re.S)
        assert style is not None
        css = style.group(1)
        selector = '.ap8-individual td:nth-child(3),.ap8-individual td:nth-child(4)'
        identifier_style = re.search(re.escape(selector) + r'\{([^}]+)\}', css)
        assert identifier_style is not None
        rules = dict(re.findall(r'([\w-]+):([^;]+)', identifier_style.group(1)))
        self.assertEqual(rules, {'min-width': '15rem', 'max-width': '20rem',
                                'white-space': 'normal', 'overflow-wrap': 'anywhere'})
        for key in ('run_id', 'model_report_id'):
            self.assertIn('<td>' + escape(study['runs'][0][key], quote=True) + '</td>', text)
        self.assertIn('<td>0.25</td>', text)
        self.assertEqual(study, before)

    def test_complete_report_has_all_seeds_warnings_and_contained_tables(self):
        body = self.render()
        self.assertIsInstance(body, bytes)
        self.assertLessEqual(len(body), 8*1024*1024)
        text=body.decode('utf-8')
        for family in ('cnn','mlp'):
            for seed in range(3):
                self.assertIn(f'{family}-run-{seed}', text)
                self.assertIn(f'{family}-model-{seed}', text)
        for token in ('Descriptive range / mean','2025 remains closed','2024 checkpoint-selected development',
                      'not equivalent','No automatic best-seed selection','no additive feature attribution',
                      '2 reused references','4 new fits','overflow-x:auto','overflow-wrap:anywhere','@media',
                      '<details','<summary','<caption','None (empty slice)','Europe/Berlin','max_epochs','Best checkpoint: 20'):
            self.assertIn(token, text)
        for bad in ('<script','<img','<link','<iframe','http://','https://'):
            self.assertNotIn(bad,text)


if __name__ == '__main__':
    unittest.main()
