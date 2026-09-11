import numpy as np

class HiddenRegimeEnv:
    def __init__(self, n_contexts=8, n_actions=4, n_regimes=4, obs_noise=0.10, regime_change_prob=0.03, episode_length=500, seed=42):
        self.n_contexts=n_contexts; self.n_actions=n_actions; self.n_regimes=n_regimes
        self.obs_noise=obs_noise; self.regime_change_prob=regime_change_prob; self.episode_length=episode_length
        self.rng=np.random.default_rng(seed)
        self.mapping=self.rng.integers(0,n_actions,size=(n_regimes,n_contexts))
    @property
    def obs_dim(self): return self.n_contexts
    def _observe(self, context):
        obs=np.zeros(self.n_contexts,dtype=np.float32); obs[context]=1
        if self.rng.random()<self.obs_noise:
            other=int(self.rng.integers(0,self.n_contexts-1))
            if other>=context: other+=1
            obs[:]=0; obs[other]=1
        return obs
    def reset(self):
        self.t=0; self.regime=int(self.rng.integers(0,self.n_regimes)); self.context=int(self.rng.integers(0,self.n_contexts))
        return self._observe(self.context)
    def oracle_action(self): return int(self.mapping[self.regime,self.context])
    def step(self, action):
        action=int(action); optimal=self.oracle_action(); reward=float(action==optimal)
        previous=self.regime; self.t+=1; changed=False
        if self.t<self.episode_length and self.rng.random()<self.regime_change_prob:
            candidates=[r for r in range(self.n_regimes) if r!=self.regime]; self.regime=int(self.rng.choice(candidates)); changed=True
        self.context=int(self.rng.integers(0,self.n_contexts))
        info={'regime':self.regime,'previous_regime':previous,'regime_changed':changed,'optimal_action':optimal}
        return self._observe(self.context),reward,self.t>=self.episode_length,info
