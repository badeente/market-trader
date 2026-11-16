"""AP6 image commands do not train models or change the running website."""
from market_trader.cli import main


def test_images_cli_is_discoverable(capsys):
    try:
        status=main(['images','--help'])
    except SystemExit as error:
        status=error.code
    assert status==0
    text=capsys.readouterr().out
    assert all(word in text for word in ('build','verify','preview'))


def test_invalid_image_bundle_has_a_concise_cli_error(tmp_path, monkeypatch, capsys):
    import sys
    from types import SimpleNamespace
    def invalid(path):
        raise ValueError('invalid synthetic image bundle')
    monkeypatch.setitem(sys.modules,'market_trader.image_dataset',SimpleNamespace(verify_image_dataset=invalid))
    assert main(['images','verify','--dataset',str(tmp_path/'bad')])==1
    error=capsys.readouterr().err
    assert 'images verify' in error and 'invalid synthetic image bundle' in error
    assert 'Traceback' not in error


def test_preview_dotdot_cannot_write_inside_immutable_bundle(tmp_path, monkeypatch):
    import market_trader.image_gallery as gallery
    bundle=tmp_path/'bundle';bundle.mkdir()
    sibling=tmp_path/'other';sibling.mkdir()
    monkeypatch.setattr(gallery,'render_image_preview',lambda path:b'synthetic preview fixture only')
    output=sibling/'..'/'bundle'/'preview.html'
    assert main(['images','preview','--dataset',str(bundle),'--output',str(output)])==1
    assert not (bundle/'preview.html').exists()


def test_images_cli_build_verify_preview_with_real_synthetic_artifacts(tmp_path,capsys):
    import json
    from tests.test_image_dataset import sources
    dataset,training=sources(tmp_path)
    bundle=tmp_path/'images'
    assert main(['images','build','--dataset',str(dataset),'--training-data',str(training),
                 '--cache',str(tmp_path/'cache'),'--per-split','2','--output',str(bundle)])==0
    result=json.loads(capsys.readouterr().out)
    assert result['sample_count']==4 and result['cache_misses']==4
    assert main(['images','verify','--dataset',str(bundle)])==0
    assert json.loads(capsys.readouterr().out)==result['manifest']
    preview=tmp_path/'preview.html'
    assert main(['images','preview','--dataset',str(bundle),'--output',str(preview)])==0
    import base64
    import re
    content=preview.read_text()
    embedded=re.findall(r'data:image/png;base64,([^\"]+)',content)
    samples=json.loads((bundle/'samples.json').read_text())['samples']
    assert len(embedded)==4
    assert [base64.b64decode(value,validate=True) for value in embedded]==[
        (bundle/sample['image_path']).read_bytes() for sample in samples]
    assert main(['images','preview','--dataset',str(bundle),'--output',str(preview)])==1
