"""Self-contained AP4 UI contract tests; Node executes the real inline script.

These are structural/DOM contract tests, not visual browser QA.
"""
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess

TEMPLATE = Path(__file__).parents[1] / "src/market_trader/workbench_template.html.in"


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.elements = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def test_self_contained_accessible_read_only_shell():
    assert TEMPLATE.exists(), "AP4 workbench template must exist"
    text = TEMPLATE.read_text()
    doc = Document(text)
    assert '<html lang="de">' in text
    for identifier in ("search", "model-filter", "status-filter", "source-filter",
                       "compare-button", "clear-selection", "run-list", "detail-view", "compare-view"):
        assert any(attrs.get("id") == identifier for _, attrs in doc.elements), identifier
    assert any(attrs.get("role") == "status" and attrs.get("aria-live") == "polite"
               for _, attrs in doc.elements)
    for tag, attrs in doc.elements:
        assert not (tag == "script" and "src" in attrs)
        assert not (tag == "link" and attrs.get("rel") == "stylesheet")
        if tag in ("input", "select"):
            assert attrs.get("aria-label") or any(
                t == "label" and a.get("for") == attrs.get("id") for t, a in doc.elements)
    assert "2025 bleibt reserviert" in text
    assert "nicht final" in text
    assert 'href="/reports"' in text
    assert any(tag == 'link' and attrs.get('rel') == 'icon'
               and attrs.get('href', '').startswith('data:image/') for tag, attrs in doc.elements)
    assert "connect-src 'self'" in text
    assert "@media" in text and "min-width:0" in text
    assert "innerHTML" not in text
    assert "insertAdjacentHTML" not in text
    assert "document.write" not in text


def script():
    match = re.search(r"<script>(.*?)</script>", TEMPLATE.read_text(), re.S)
    assert match
    return match[1]


def node_check(body):
    node = shutil.which("node")
    assert node, "Node is required to execute workbench JS contract tests"
    result = subprocess.run([node, "-"], input=body, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


DOM = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag='div') { this.tagName=tag; this.children=[]; this.attributes={}; this.dataset={}; this.value=''; this.hidden=false; this.disabled=false; this.listeners={}; this._text=''; this.style={setProperty:(k,v)=>{this.style[k]=v;}}; }
  set textContent(v) { this._text=String(v); this.children=[]; }
  get textContent() { return this._text + this.children.map(c=>c.textContent ?? String(c)).join(''); }
  append(...nodes) { this.children.push(...nodes); }
  appendChild(n) { this.append(n); return n; }
  replaceChildren(...nodes) { this._text=''; this.children=[...nodes]; }
  setAttribute(k,v) { this.attributes[k]=String(v); }
  getAttribute(k) { return this.attributes[k]; }
  addEventListener(k,f) { this.listeners[k]=f; }
  focus() { this.focused=true; }
}
const elements=new Map();
const document={createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t),
 getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},
 addEventListener:()=>{}};
const location={origin:'http://localhost:8086',pathname:'/experiments',search:''};
const history={pushState:(s,t,u)=>{location.search=new URL(u,location.origin).search;}};
const window={addEventListener:()=>{}};
function walk(n) { return [n,...n.children.flatMap(c=>c instanceof Element?walk(c):[])]; }
"""


DETAIL = r"""
const detail={run:{key:'ap3:run-a',run_id:'run-a',source:'ap3',name:'Saved MLP',model_kind:'mlp',status:'completed',selectable:true,
 parent_id:'run-parent',hypothesis:'Saved hypothesis',training_rows:18,validation_rows:5,created_at:'2026-01-01',metrics:{mae_bp:2,net_points:'-1.25'},
 report_url:'/experiment-report?run=ap3%3Arun-a'},config:{model:{kind:'mlp',learning_rate:0.001},seed:7},
 signature:{validation_dataset_id:'sha256:dataset',code_revision:'abc'},runtime:{python:'3.14'},
 metrics_by_model:{mlp:{mae_bp:2,rmse_bp:3},ridge:{mae_bp:3,rmse_bp:4},zero_return:{mae_bp:5,rmse_bp:6}},
 trading_by_model:{mlp:{trade_count:3,net_points:'-1.25',reference_points:4,modeled_cost_points:5.25,max_drawdown_points:2}},
 training:{history:[{epoch:1,training_mse:1,validation_mse:2},{epoch:2,training_mse:0.8,validation_mse:1.8}],best_epoch:2,stopping_reason:'patience'},
 curves:{equity:[['2024-01-01',0],['2024-02-01',-1.25]],drawdown:[['2024-01-01',0],['2024-02-01',2]],sampled:true,source_count:1001},warnings:['Saved warning'],error:null};
"""


def test_detail_real_parameters_units_baselines_training_and_curves():
    node_check(DOM + script() + DETAIL + r"""
renderDetail(detail);
const root=$('detail-view'),all=walk(root),txt=root.textContent;
for(const value of ['Saved MLP','run-parent','learning_rate','0.001','sha256:dataset','3.14','zero_return',
 'ridge','MAE (bp)','RMSE (bp)','Netto (Punkte)','-1,25 Punkte','Beste Epoche','patience','standardisierte Ziel-MSE','heruntergesampelt','1001']) assert(txt.includes(value),value);
assert(all.some(n=>n.tagName==='a'&&n.href==='/experiment-report?run=ap3%3Arun-a'));
assert(all.some(n=>n.className==='negative'&&n.textContent==='-1,25 Punkte'));
assert.equal(all.filter(n=>n.tagName==='svg').length,3,'MSE, equity and drawdown are separate charts');
renderDetail({...detail,run:{...detail.run,status:'invalid'},error:'Integrität ungültig'});
assert($('detail-view').textContent.includes('Integrität ungültig'));
assert(!walk($('detail-view')).some(n=>n.tagName==='svg'));
renderDetail({...detail,training:null,curves:{equity:[],drawdown:[],source_count:0,sampled:false}});
assert($('detail-view').textContent.includes('Keine gespeicherte Trainingshistorie'));
""")


def test_compare_differences_precede_metrics_and_never_rank_models():
    node_check(DOM + script() + DETAIL + r"""
const pair={left:'ap3:run-a',right:'ap2:run-b',differences:[{field:'execution.spread',left:'1',right:'3'}]};
renderCompare({compatible:false,runs:[detail.run,{...detail.run,key:'ap2:run-b'}],details:[detail],
 condition_differences:[pair],parameter_differences:[{...pair,differences:[{field:'model.kind',left:'mlp',right:'ridge'}]}],warnings:['Not comparable']});
const txt=$('compare-view').textContent;
assert(txt.includes('Nicht kompatibel'));
assert(txt.indexOf('execution.spread')<txt.indexOf('MAE (bp)'));
assert(txt.indexOf('model.kind')<txt.indexOf('MAE (bp)'));
assert(txt.includes('Not comparable'));
assert(txt.includes('keine Rangliste'));
assert(!/bestes Modell|Gewinner|profitabelste/i.test(txt));
""")


def test_time_alignment_single_point_focus_and_invalid_detail_contract():
    node_check(DOM + script() + DETAIL + r"""
const g=chart('Aligned',[
 {name:'A',color:'#fff',points:[['2024-01-01',0],['2024-01-02',2],['2024-01-03',1]]},
 {name:'B',color:'#aaa',points:[['2024-01-02',3],['2024-01-03',2]]}], 'Punkte','Zeit');
const paths=walk(g).filter(n=>n.tagName==='path');
assert(paths[1].attributes.d.startsWith(' M348.00 '),'matching timestamps must have matching x coordinates');
const single=chart('Single',[{name:'A',color:'#fff',points:[[1,2]]}],'MSE','Epoche');
assert(walk(single).some(n=>n.tagName==='circle'),'single observation must be visible');
renderDetail(detail);assert(walk($('detail-view')).some(n=>n.tagName==='button'&&n.textContent==='Aktualisieren'));
state.runs=[detail.run];toggleSelection(detail.run.key,true);
assert(walk($('run-list')).find(n=>n.tagName==='input'&&n.dataset.runKey===detail.run.key).focused,'keyboard focus survives selection render');
(async()=>{
location.search='?run=ap3:run-'+ 'a'.repeat(32);
global.fetch=async()=>({ok:true,json:async()=>({...detail,signature:null,run:{...detail.run,status:'invalid'},error:'Invalid checksum'})});
await routeFromURL();assert($('detail-view').textContent.includes('Invalid checksum'));
assert(!$('error-box').textContent.includes('API-Schema'));
})().catch(e=>{console.error(e);process.exitCode=1;});
""")


def test_inline_script_syntax_read_only_routes_and_mobile_legend_wrapping():
    node = shutil.which('node')
    assert node
    checked = subprocess.run([node, '--check'], input=script(), text=True, capture_output=True)
    assert checked.returncode == 0, checked.stderr
    text = TEMPLATE.read_text()
    assert '.legend span{overflow-wrap:anywhere;min-width:0;max-width:100%}' in text
    assert "method:'GET'" in script()
    for forbidden in ('XMLHttpRequest', 'WebSocket', 'sendBeacon', 'eval(', 'new Function(',
                      '/api/train', '/api/start', '/api/upload', 'localStorage'):
        assert forbidden not in script()
    assert 'popstate' in script() and 'DOMContentLoaded' in script()
    assert '/api/experiments' in script()
    assert '/api/experiment?run=' in script()
    assert "'/api/compare?'" in script()


def test_back_navigation_clears_route_error_and_preserves_registry_warnings():
    node_check(DOM + script() + r"""
(async()=>{
global.fetch=async()=>({ok:true,json:async()=>({runs:[],sources:[],errors:['Quelle nicht lesbar'],read_only:true})});
await loadOverview();
location.search='?run=ap3:run-'+ 'a'.repeat(32);
global.fetch=async()=>({ok:false,status:404});await routeFromURL();
assert($('error-box').textContent.includes('nicht gefunden'));
await navigate('overview');
assert(!$('error-box').textContent.includes('nicht gefunden'));
assert($('error-box').textContent.includes('Quelle nicht lesbar'));
})().catch(e=>{console.error(e);process.exitCode=1;});
""")


def test_navigation_fetch_contract_errors_refresh_and_stale_response_guard():
    node_check(DOM + script() + DETAIL + r"""
(async()=>{
const a='ap3:run-'+ 'a'.repeat(32), b='ap2:run-'+ 'b'.repeat(32);
assert.deepEqual(parseRoute('?run='+encodeURIComponent(a)),{kind:'run',keys:[a]});
assert.deepEqual(parseRoute('?compare='+encodeURIComponent(a+','+b)),{kind:'compare',keys:[a,b]});
for(const q of ['?run=x','?compare='+a,'?compare='+a+','+a,'?run='+a+'&run='+a,'?run='+a+'&compare='+b,'?oops=1']) assert.throws(()=>parseRoute(q));
const calls=[];
global.fetch=async(url,options)=>{calls.push([url,options]);return {ok:true,json:async()=>url==='/api/experiments'?{runs:[{...detail.run,key:a}],sources:[{id:'ap3'}],errors:[],read_only:true}:url.startsWith('/api/compare')?{compatible:true,runs:[detail.run],condition_differences:[],parameter_differences:[],details:[detail],warnings:[]}:detail};};
await loadOverview();assert.equal(state.runs.length,1);
await navigate('run',[a]);assert(location.search.startsWith('?run='));
assert(calls.at(-1)[0]==='/api/experiment?run='+encodeURIComponent(a));
assert(!$('detail-view').hidden);assert($('overview-view').hidden);
await navigate('compare',[a,b]);
assert.equal(calls.at(-1)[0],'/api/compare?run='+encodeURIComponent(a)+'&run='+encodeURIComponent(b));
assert(!$('compare-view').hidden);
for(const [,o] of calls){assert.equal(o.method,'GET');assert.equal(o.credentials,'same-origin');assert.equal(o.cache,'no-store');assert.equal(o.redirect,'error');}
await navigate('overview');assert(!$('overview-view').hidden);
location.search='?run='+encodeURIComponent(a);await routeFromURL();assert(!$('detail-view').hidden);
global.fetch=async()=>({ok:false,status:404});await routeFromURL();
assert($('error-box').textContent.includes('nicht gefunden'));
assert(!$('detail-view').textContent.includes('Saved MLP'),'no stale metrics after error');
global.fetch=async()=>{throw Error('/private/path secret');};await loadOverview();
assert.equal(state.runs.length,0);assert(!$('error-box').textContent.includes('/private/path'));
let release;
global.fetch=()=>new Promise(resolve=>release=resolve);
const pending=routeFromURL();location.search='';await routeFromURL();
release({ok:true,json:async()=>detail});await pending;
assert($('detail-view').hidden,'late detail response must not replace overview');
})().catch(e=>{console.error(e);process.exitCode=1;});
""")


def test_overview_filters_selection_bounds_hidden_and_failed_rows():
    node_check(DOM + script() + r"""
const rows=Array.from({length:5},(_,i)=>({key:'ap3:run-'+i,run_id:'run-'+i,name:'Real '+i,
 source:'ap3',model_kind:i===0?'ridge':'mlp',status:'completed',selectable:true,
 hypothesis:'Saved hypothesis',parent_id:i?'run-0':null,created_at:'2026-01-01',training_rows:10,validation_rows:4,
 metrics:{mae_bp:1.23,rmse_bp:2,net_points:'-3.2'}}));
rows.push({...rows[0],key:'ap2:failed',source:'ap2',status:'failed',selectable:false,metrics:{mae_bp:null}});
state.runs=rows;renderRuns();
assert.equal(walk($('run-list')).filter(n=>n.tagName==='input').length,6);
assert(walk($('run-list')).find(n=>n.dataset.runKey==='ap2:failed'&&n.tagName==='input').disabled);
assert.equal(toggleSelection('ap2:failed',true),false);
assert.equal(toggleSelection(rows[0].key,true),true);
assert($('compare-button').disabled);
assert.equal(toggleSelection(rows[1].key,true),true);
assert(!$('compare-button').disabled);
toggleSelection(rows[2].key,true);toggleSelection(rows[3].key,true);
assert.equal(toggleSelection(rows[4].key,true),false);
assert.equal(state.selected.size,4);
$('model-filter').value='ridge'; renderRuns();
assert.equal(state.selected.size,4);
assert($('selection-summary').textContent.includes('3 ausgeblendet'));
$('search').value='impossible';renderRuns();assert($('run-list').textContent.includes('Keine Runs'));
assert($('selected-keys').textContent.includes('ap3:run-1'));
clearSelection();assert.equal(state.selected.size,0);assert($('compare-button').disabled);
""")


def test_mobile_charts_use_a_readable_viewbox_not_desktop_scaled_text():
    node_check(DOM + script() + r"""
window.innerWidth=390;
const g=chart('Loss',[{name:'Training',color:'#fff',points:[[1,1],[30,0.5]]}],'standardisierte Ziel-MSE','Epoche');
const svg=walk(g).find(n=>n.tagName==='svg');const width=Number(svg.attributes.viewBox.split(' ')[2]);
assert(width>=260&&width<=320,'phone chart should not shrink 640px axis text to half size');
const path=walk(svg).find(n=>n.tagName==='path');
assert(!path.attributes.d.includes('NaN'));
assert(walk(svg).filter(n=>n.tagName==='line').every(n=>Number(n.attributes.x2)<=width),'grid stays inside responsive viewbox');
""")


def test_forecast_basis_points_keep_four_digits_in_all_views():
    node_check(DOM + script() + DETAIL + r"""
detail.run.metrics.mae_bp='10.095152201669954';detail.metrics_by_model.mlp.mae_bp='10.095152201669954';
state.runs=[detail.run];renderRuns();renderDetail(detail);
renderCompare({compatible:true,runs:[detail.run],details:[detail],condition_differences:[],parameter_differences:[],warnings:[]});
for(const id of ['run-list','detail-view','compare-view']) assert($(id).textContent.includes('10,0952 bp'),id+' must not hide small saved forecast differences through rounding');
""")


def test_configuration_differences_are_collapsible_but_key_change_is_previewed():
    node_check(DOM + script() + DETAIL + r"""
const pair={left:detail.run.key,right:'ap3:run-b',differences:[{field:'model',left:{kind:'mlp',layers:[4,16,8,1]},right:null}]};
renderCompare({compatible:true,runs:[detail.run],details:[detail],condition_differences:[],parameter_differences:[pair],warnings:[]});
const all=walk($('compare-view'));
const fold=all.find(n=>n.tagName==='details'&&n.children.some(c=>c.tagName==='summary'&&c.textContent.includes('Konfigurationsunterschiede')));
assert(fold,'long parameter difference table must not bury results');
assert(fold.children.find(n=>n.tagName==='summary').textContent.includes('model.kind'),'model change must be visible without expanding');
assert(fold.open!==true,'parameter detail should initially be collapsed');
assert(all.some(n=>n.tagName==='section'&&n.textContent.includes('Vergleichsbedingungen')),'condition checks stay visible');
""")


def test_overview_shows_actual_dataset_period_separate_from_creation_time():
    node_check(DOM + script() + r"""
const key='ap3:run-'+ 'a'.repeat(32);
state.runs=[{key,run_id:key.split(':')[1],source:'ap3',name:'MLP',model_kind:'mlp',
 status:'completed',selectable:true,created_at:'2026-11-15T19:00:00Z',
 dataset_id:'sha256:actual-validation-dataset',source_sha256:'sha256:actual-source',
 period:{validation_start:'2024-01-02T10:00:00+00:00',validation_end:'2024-12-31T15:45:00+00:00'},metrics:{}}];
renderRuns();const txt=$('run-list').textContent;
assert(txt.includes('2024-01-02') && txt.includes('2024-12-31'),'actual validation period must be visible');
assert(txt.includes('sha256:actual-validation-dataset'),'saved dataset identity must be visible');
assert(txt.includes('Erstellt: 2026-11-15'),'creation time must be labeled separately');
assert(txt.includes('Validierung:'),'dates must say what period they describe');
""")


def test_detail_results_precede_collapsed_identity_and_parameters():
    node_check(DOM + script() + DETAIL + r"""
renderDetail(detail);const root=$('detail-view'),txt=root.textContent;
assert(txt.indexOf('Prognosefehler') < txt.indexOf('Konfiguration & Parameter'),'results must not be buried under full metadata');
const folds=walk(root).filter(n=>n.tagName==='details');
for(const label of ['Run-Identität','Konfiguration & Parameter','Datenidentitäten & Vergleichsbedingungen','Code & Laufzeit'])
 assert(folds.some(n=>n.children.some(c=>c.tagName==='summary'&&c.textContent===label)),'collapsible '+label);
assert(txt.includes('0,0001 Rendite'));
assert(txt.includes('mittlerer absoluter'),'forecast quality criterion explained');
""")


def test_formatting_safe_links_flattening_and_graph_missing_values():
    node_check(DOM + script() + r"""
assert.equal(number(null,'bp'),'—');
assert.equal(number('', 'bp'),'—');
assert.equal(number('-12.345','Punkte'),'-12,345 Punkte');
assert.equal(number(0,'bp'),'0 bp');
assert.equal(number('NaN','bp'),'—');
assert.equal(safeReportURL('javascript:alert(1)','ap3:run-x'),null);
assert.equal(safeReportURL('//evil.test/experiment-report?run=ap3:run-x','ap3:run-x'),null);
assert.equal(safeReportURL('/experiment-report?run=wrong','ap3:run-x'),null);
assert.equal(safeReportURL('/experiment-report?run=ap3%3Arun-x','ap3:run-x'),'/experiment-report?run=ap3%3Arun-x');
assert.deepEqual(flatten({model:{kind:'mlp',layers:[4,16,8,1]},seed:0}),[
 ['model.kind','mlp'],['model.layers.0',4],['model.layers.1',16],['model.layers.2',8],['model.layers.3',1],['seed',0]]);
const payload='<img src=x onerror=alert(1)>';
const t=table(['Feld','Wert'],[['name',payload]]);
assert(t.textContent.includes(payload));
assert(!walk(t).some(n=>n.tagName==='img'));
const g=chart('Test', [{name:'Netto',color:'#fff',points:[['A',1],['B',null],['C',-2]]}], 'Punkte','Zeit');
assert(g.textContent.includes('Punkte'));
const paths=walk(g).filter(n=>n.tagName==='path');
assert.equal(paths.length,1);
assert.equal((paths[0].attributes.d.match(/M/g)||[]).length,2,'null breaks curve instead of turning into zero');
assert(!paths[0].attributes.d.includes('NaN'));
const empty=chart('Leer',[{name:'Netto',color:'#fff',points:[['A',null]]}],'Punkte','Zeit');
assert(empty.textContent.includes('Keine Kurvendaten'));
""")
