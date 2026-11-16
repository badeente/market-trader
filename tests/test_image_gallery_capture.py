"""Capture-coherence regression with an actual verified synthetic image bundle."""
import pytest


def test_mutation_between_verification_and_capture_is_rejected(tmp_path,monkeypatch):
    from tests.test_image_dataset import sources
    import market_trader.image_dataset as datasets
    from market_trader.image_gallery import render_image_preview
    dataset,training=sources(tmp_path)
    bundle=tmp_path/'images'
    datasets.build_image_dataset(dataset,training,bundle,cache=tmp_path/'cache',per_split=1)
    original=datasets.verify_image_dataset
    def changed_after_verify(path):
        manifest=original(path)
        (bundle/'labels.json').write_bytes(b'{}')
        return manifest
    monkeypatch.setattr(datasets,'verify_image_dataset',changed_after_verify)
    with pytest.raises(ValueError,match='changed during preview capture'):
        render_image_preview(bundle)
