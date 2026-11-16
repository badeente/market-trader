"""Self-contained human chart audit preview, separate from model input images."""
from base64 import b64encode
from html import escape


def _render_preview(manifest: dict, samples: list, labels: list, images: dict[str, bytes]) -> bytes:
    label_by_id = {label['sample_id']: label for label in labels}
    cards = []
    for row in samples:
        window = row['window']
        bars = window['candles']
        table = ''.join('<tr>' + ''.join('<td>' + escape(str(bar.get(field, ''))) + '</td>'
                            for field in ('open_time','open','high','low','close','available_at')) + '</tr>' for bar in bars)
        label = label_by_id[row['sample_id']]
        target = ''.join('<dt>'+escape(field)+'</dt><dd>'+escape(str(label[field]))+'</dd>'
                         for field in ('entry_at','exit_at','target_available_at','target_return'))
        uri = 'data:image/png;base64,' + b64encode(images[row['image_path']]).decode('ascii')
        cards.append('<article class="card"><h2>'+escape(row['split'])+' · '+escape(row['decision_at'])+'</h2>'
            '<p>'+escape(window['instrument'])+' · '+escape(row['sample_id'])+'</p>'
            '<img width="256" height="192" alt="Acht historische Kerzen, ohne Zielbeschriftung" src="'+uri+'">'
            '<p class="legend">Grün: Schluss über Eröffnung · Rot: darunter · Grau: unverändert. '
            'Preisachse nur aus diesem Eingabefenster; keine globale Skala.</p>'
            '<details><summary>Eingabekerzen kontrollieren</summary><div class="scroll"><table><thead><tr>'
            '<th>UTC-Eröffnung</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Verfügbar UTC</th>'
            '</tr></thead><tbody>'+table+'</tbody></table></div></details>'
            '<details class="labels"><summary>Zielwerte prüfen – nicht Teil des Eingabebildes</summary>'
            '<p>Diese später verfügbaren Angaben stehen getrennt in labels.json und sind keine Bildmerkmale.</p><dl>'
            +target+'</dl></details></article>')
    html = ('<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
        '<title>AP6 · Kausale Chartbilder</title><style>'
        '*{box-sizing:border-box}body{margin:0;background:#0b1020;color:#e6edf7;font:16px/1.5 system-ui,sans-serif}'
        'main{max-width:1180px;margin:auto;padding:24px}h1{font-size:2rem}h2{font-size:1.05rem}h2,p,dd,td{overflow-wrap:anywhere}'
        '.notice{border-left:4px solid #a78bfa;padding:12px;background:#171d30}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:18px}'
        '.card{min-width:0;padding:18px;border:1px solid #334155;border-radius:12px;background:#111827}.card img{display:block;max-width:100%;height:auto;image-rendering:pixelated;margin:auto}'
        '.legend{font-size:.85rem;color:#b8c5d7}summary{cursor:pointer;padding:10px 0}summary:focus-visible{outline:2px solid #a78bfa}'
        '.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:.8rem}th,td{text-align:left;padding:6px;white-space:nowrap;border-bottom:1px solid #334155}'
        'dt{font-weight:bold}dd{margin:0 0 8px}.labels{margin-top:8px;border-top:1px solid #475569}'
        '</style></head><body><main><h1>AP6 · Kausale Chartbilder</h1>'
        '<p class="notice">Vorschau zur Datenprüfung, kein Modelltraining und keine Ergebnisbewertung. '
        'Das PNG enthält ausschließlich acht damals bekannte Kerzen. Metadaten und Zielwerte dieser HTML-Seite sind nicht Teil des Eingabebildes. '
        'Gleiche Zeitfenster bedeuten nicht gleichen Informationsgehalt: Die Bilder zeigen mehr OHLC-Details als die vier numerischen Merkmale. 2025 bleibt reserviert.</p>'
        '<p>Rezept: '+escape(str(manifest['recipe_id']))+' · Bilder: '+str(len(samples))+'</p><div class="grid">'
        +''.join(cards)+'</div></main></body></html>')
    output = html.encode('utf-8')
    if len(output) > 32 * 1024 * 1024:
        raise ValueError('Preview exceeds 32 MiB')
    return output


def render_image_preview(bundle) -> bytes:
    """Capture verified bundle bytes coherently; do not expose arbitrary paths."""
    from pathlib import Path
    from hashlib import sha256
    import os
    import re
    import stat
    from market_trader.experiment_registry import _check_path, _decode, _encode
    root = _check_path(Path(bundle))
    from market_trader.image_dataset import verify_image_dataset
    manifest = verify_image_dataset(root)

    def capture(name, limit):
        if name not in ('samples.json', 'labels.json') and re.fullmatch(r'images/[0-9a-f]{64}\.png', name) is None:
            raise ValueError('Invalid preview artifact path')
        path = _check_path(root / name)
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError('Invalid or oversized preview artifact')
            data = stream.read(limit + 1)
        if len(data) > limit or 'sha256:' + sha256(data).hexdigest() != manifest['artifacts'].get(name):
            raise ValueError('Image bundle changed during preview capture')
        return data

    try:
        samples = _decode(capture('samples.json', 2 * 1024 * 1024))['samples']
        labels = _decode(capture('labels.json', 2 * 1024 * 1024))['labels']
        images = {row['image_path']: capture(row['image_path'], 200 * 1024) for row in samples}
        html = _render_preview(manifest, samples, labels, images)
        if _encode(verify_image_dataset(root)) != _encode(manifest):
            raise ValueError('Image bundle changed during preview rendering')
        return html
    except (KeyError, TypeError, RecursionError) as error:
        raise ValueError('Invalid image preview metadata') from error
