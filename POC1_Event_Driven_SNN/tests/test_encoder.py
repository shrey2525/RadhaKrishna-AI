import numpy as np
from encoders.delta import DeltaEncoder

def test_delta_encoder():
    x = np.array([[0., 0.], [0., 0.], [0.2, 0.]])
    s = DeltaEncoder(0.1).encode(x)
    assert s.shape == x.shape
    assert s[2, 0] == 1
