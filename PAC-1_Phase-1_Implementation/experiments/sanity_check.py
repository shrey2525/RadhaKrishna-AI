import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from environment.hidden_regime import HiddenRegimeEnv

env=HiddenRegimeEnv(seed=42); env.reset(); reward=0; changes=0
for _ in range(env.episode_length):
    _,r,done,info=env.step(0); reward+=r; changes+=int(info['regime_changed'])
    if done: break
print('PAC-1 environment sanity check')
print(f'Episode length: {env.episode_length}')
print(f'Fixed-action reward: {reward:.0f}/{env.episode_length}')
print(f'Regime changes: {changes}')
print('PASS: environment executes end-to-end.')
