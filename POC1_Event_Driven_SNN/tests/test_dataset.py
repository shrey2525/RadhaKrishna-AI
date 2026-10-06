from benchmarks.synthetic import SyntheticTemporalDataset

def test_dataset_shape():
    ds = SyntheticTemporalDataset(10, 20, 8, 3, 0.1)
    x, y = ds[0]
    assert x.shape == (20, 8)
    assert y.ndim == 0
