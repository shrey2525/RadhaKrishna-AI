import sys
from pathlib import Path

import yaml, torch
from torch import nn
from torch.utils.data import DataLoader

# Add the POC2 directory when this file is launched directly, e.g.
# `python experiments/train.py` or `python /path/to/experiments/train.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks import DelayedRecallDataset
from models import PersistentSNN, GRUClassifier
from experiments.common import seed_everything

def main():
    with (PROJECT_ROOT / 'configs' / 'poc2.yaml').open() as config_file:
        cfg = yaml.safe_load(config_file)
    seed_everything(cfg['seed'])
    d = cfg['data']; m = cfg['model']; tr = cfg['training']
    train = DelayedRecallDataset(d['train_samples'], d['sequence_length'], d['input_dim'], d['n_classes'], d['memory_gap'], d['distractor_probability'], cfg['seed'])
    loader = DataLoader(train, batch_size=tr['batch_size'], shuffle=True)
    models = {
        'gru': GRUClassifier(d['input_dim'], m['hidden_dim'], d['n_classes']),
        'persistent_snn': PersistentSNN(d['input_dim'], m['hidden_dim'], m['memory_dim'], d['n_classes'], m['tau_fast'], m['tau_slow'])
    }
    results_dir = PROJECT_ROOT / 'results' / 'latest'
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        opt = torch.optim.Adam(model.parameters(), lr=tr['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(tr['epochs']):
            model.train(); total = 0
            for x, y in loader:
                opt.zero_grad()
                out = model(x)
                if out.ndim == 3: out = out[:, -1]
                loss = loss_fn(out, y); loss.backward(); opt.step(); total += loss.item()
            print(f'{name:16s} epoch={epoch+1:02d} loss={total/len(loader):.4f}')
        torch.save(model.state_dict(), results_dir / f'{name}.pt')

if __name__ == '__main__': main()
