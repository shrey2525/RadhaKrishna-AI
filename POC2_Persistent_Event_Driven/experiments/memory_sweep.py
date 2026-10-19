import json
import sys
from pathlib import Path

import yaml, torch, pandas as pd
from torch.utils.data import DataLoader

# Add the POC2 directory when this file is launched directly, e.g.
# `python experiments/memory_sweep.py` or `python /path/to/experiments/memory_sweep.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks import DelayedRecallDataset
from models import PersistentSNN, GRUClassifier
from experiments.common import accuracy, seed_everything

def evaluate(model, loader, is_snn=False):
    model.eval(); preds=[]; ys=[]
    with torch.no_grad():
        for x,y in loader:
            out = model(x)
            if out.ndim == 3: out = out[:, -1]
            preds.append(out.argmax(-1)); ys.append(y)
    pred=torch.cat(preds); y=torch.cat(ys)
    return accuracy(pred,y), getattr(model,'last_spike_count',0)

def main():
    with (PROJECT_ROOT / 'configs' / 'poc2.yaml').open() as config_file:
        cfg = yaml.safe_load(config_file)
    seed_everything(cfg['seed'])
    d=cfg['data']; m=cfg['model']; results=[]
    results_dir = PROJECT_ROOT / 'results' / 'latest'
    for gap in cfg['benchmark']['gaps']:
        seq=max(d['sequence_length'], gap+2)
        test=DelayedRecallDataset(d['test_samples'], seq, d['input_dim'], d['n_classes'], gap, d['distractor_probability'], cfg['seed']+gap)
        loader=DataLoader(test,batch_size=256)
        gru=GRUClassifier(d['input_dim'],m['hidden_dim'],d['n_classes'])
        snn=PersistentSNN(d['input_dim'],m['hidden_dim'],m['memory_dim'],d['n_classes'],m['tau_fast'],m['tau_slow'])
        # This sweep is intentionally an untrained capacity diagnostic unless checkpoints exist.
        for name,model in [('gru',gru),('persistent_snn',snn)]:
            path = results_dir / f'{name}.pt'
            if path.exists():
                try: model.load_state_dict(torch.load(path,map_location='cpu'))
                except RuntimeError: pass
            acc,spikes=evaluate(model,loader,name=='persistent_snn')
            results.append({'gap':gap,'model':name,'accuracy':acc,'spike_count_last_batch':spikes})
    results_dir.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(results); df.to_csv(results_dir / 'memory_sweep.csv',index=False)
    with (results_dir / 'memory_sweep.json').open('w') as output_file:
        json.dump(results, output_file, indent=2)
    print(df.to_string(index=False))

if __name__=='__main__': main()
