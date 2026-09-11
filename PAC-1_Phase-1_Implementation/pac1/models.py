import torch
import torch.nn as nn

class StatelessMLP(nn.Module):
    def __init__(self,input_dim,n_actions,hidden=64):
        super().__init__(); self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),nn.Linear(hidden,n_actions))
    def forward(self,x): return self.net(x)

class ContextTransformer(nn.Module):
    def __init__(self,token_dim,n_actions,hidden=64,layers=2,heads=4,max_len=20):
        super().__init__(); self.proj=nn.Linear(token_dim,hidden); self.pos=nn.Parameter(torch.randn(1,max_len,hidden)*.02); self.max_len=max_len
        layer=nn.TransformerEncoderLayer(d_model=hidden,nhead=heads,dim_feedforward=hidden*2,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,num_layers=layers); self.head=nn.Linear(hidden,n_actions)
    def forward(self,x):
        x=x[:,-self.max_len:]; h=self.proj(x)+self.pos[:,:x.size(1)]; h=self.encoder(h); return self.head(h[:,-1])

class PersistentGRU(nn.Module):
    def __init__(self,token_dim,n_actions,hidden=64):
        super().__init__(); self.gru=nn.GRU(token_dim,hidden,batch_first=True); self.head=nn.Linear(hidden,n_actions)
    def forward(self,x,hidden=None):
        h,hidden=self.gru(x,hidden); return self.head(h[:,-1]),hidden
