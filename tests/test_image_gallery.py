"""Preview is a human audit artifact, never a neural-network input image."""
import base64


def test_preview_escapes_metadata_embeds_images_and_separates_labels():
    from market_trader.image_gallery import _render_preview
    row={'sample_id':'sha256:'+'1'*64,'split':'train','decision_at':'2021-01-04T10:00:00+00:00',
         'image_key':'sha256:'+'2'*64,'image_path':'images/'+'2'*64+'.png',
         'window':{'instrument':'<script>alert(1)</script>','candles':[
             {'open_time':'2021-01-04T09:45:00+00:00','close_time':'2021-01-04T10:00:00+00:00',
              'available_at':'2021-01-04T10:00:00+00:00','open':'1','high':'2','low':'1','close':'2'}]}}
    label={'sample_id':row['sample_id'],'entry_at':row['decision_at'],'exit_at':'2021-01-04T10:30:00+00:00',
           'target_available_at':'2021-01-04T10:45:00+00:00','target_return':'0.01'}
    html=_render_preview({'recipe_id':'sha256:'+'3'*64},[row],[label],{row['image_path']:b'synthetic bytes for formatter test only'}).decode()
    assert 'h2,p,dd,td{overflow-wrap:anywhere}' in html
    assert '<script>' not in html
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in html
    assert 'data:image/png;base64,'+base64.b64encode(b'synthetic bytes for formatter test only').decode() in html
    assert '<details class="labels">' in html
    assert 'nicht Teil des Eingabebildes' in html
    assert 'http://' not in html and 'https://' not in html and '<script' not in html


def test_public_preview_rejects_symlink_bundle_before_reading(tmp_path):
    import pytest
    from market_trader.image_gallery import render_image_preview
    source=tmp_path/'bundle'
    source.mkdir()
    link=tmp_path/'linked'
    link.symlink_to(source,target_is_directory=True)
    with pytest.raises(ValueError,match='symbolic'):
        render_image_preview(link)


def test_preview_rejects_mutation_after_capture(tmp_path,monkeypatch):
    import pytest
    import market_trader.image_gallery as gallery
    from market_trader.image_dataset import build_image_dataset
    from tests.test_image_dataset import sources
    dataset,training=sources(tmp_path)
    bundle=tmp_path/'images'
    build_image_dataset(dataset,training,bundle,cache=tmp_path/'cache',per_split=1)
    original=gallery._render_preview
    def mutate(*args):
        output=original(*args)
        (bundle/'labels.json').write_bytes(b'{}')
        return output
    monkeypatch.setattr(gallery,'_render_preview',mutate)
    with pytest.raises(ValueError,match='checksum'):
        gallery.render_image_preview(bundle)
