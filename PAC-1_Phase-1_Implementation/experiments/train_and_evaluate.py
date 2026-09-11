import sys, json, random
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from config import Config
from environment.hidden_regime import HiddenRegimeEnv
from pac1.data import build_dataset
from pac1.models import StatelessMLP,ContextTransformer,PersistentGRU
from pac1.evaluate import evaluate_agent

def main():
    c=Config(); random.seed(c.seed); np.random.seed(c.seed); torch.manual_seed(c.seed); device='cuda' if torch.cuda.is_available() else 'cpu'
    def factory(): return HiddenRegimeEnv(c.n_contexts,c.n_actions,c.n_regimes,c.obs_noise,c.regime_change_prob,c.episode_length,np.random.randint(2**31-1))
    X,Y=build_dataset(factory,c.train_episodes); loader=DataLoader(TensorDataset(torch.tensor(X),torch.tensor(Y)),batch_size=c.batch_size,shuffle=True)
    td=c.n_contexts+c.n_actions+1
    models={'stateless':StatelessMLP(c.n_contexts,c.n_actions,c.hidden_size).to(device),'transformer':ContextTransformer(td,c.n_actions,c.hidden_size,c.transformer_layers,c.transformer_heads,c.context_window).to(device),'gru':PersistentGRU(td,c.n_actions,c.hidden_size).to(device)}
    losses={}
    for name,m in models.items():
        opt=torch.optim.Adam(m.parameters(),lr=c.learning_rate); loss_fn=nn.CrossEntropyLoss()
        print(f'Training {name}...', flush=True)
        for epoch in range(c.epochs):
            total=0
            for xb,yb in loader:
                xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
                if name=='stateless': logits=m(xb[:,:,:c.n_contexts].reshape(-1,c.n_contexts)); target=yb.reshape(-1)
                elif name=='transformer': logits=m(xb[:,-c.context_window:]); target=yb[:,-1]
                else: logits,_=m(xb); logits=logits[:,-1] if logits.dim()==3 else logits; target=yb[:,-1]
                loss=loss_fn(logits,target); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); total+=loss.item()
            losses[name]=total/len(loader)
            print(f'  {name} epoch {epoch+1}/{c.epochs} loss={losses[name]:.4f}', flush=True)
    results={n:evaluate_agent(factory(),m,n,c.eval_episodes,c.adaptation_window,c.recovery_fraction,device) for n,m in models.items()}
    out=Path(__file__).resolve().parents[1]/'results'; out.mkdir(exist_ok=True)
    for n,m in models.items(): torch.save(m.state_dict(),out/f'{n}.pt')
    payload={'config':c.__dict__,'device':device,'final_training_losses':losses,'evaluation':results}; (out/'results.json').write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
