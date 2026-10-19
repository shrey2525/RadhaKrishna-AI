import torch
from core.memory import PersistentState
from models.lif import LIFCell

def test_persistent_state_decays_and_updates():
    m=PersistentState(2,decay=0.9); m.reset(1)
    s1=m.update(torch.tensor([[1.,0.]])); s2=m.update(torch.zeros(1,2))
    assert s2[0,0] < s1[0,0] and s2[0,0] > 0

def test_lif_spikes():
    c=LIFCell(2,tau=.5,threshold=1.)
    c.reset(1,'cpu')
    s=c(torch.tensor([[1.2,0.]]))
    assert s.shape==(1,2)
