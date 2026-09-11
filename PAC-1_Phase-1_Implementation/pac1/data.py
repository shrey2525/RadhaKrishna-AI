import numpy as np

def make_token(obs,previous_action,previous_reward,n_actions):
    a=np.zeros(n_actions,dtype=np.float32); a[previous_action]=1
    return np.concatenate([obs.astype(np.float32),a,np.array([previous_reward],dtype=np.float32)])

def generate_episode(env):
    obs=env.reset(); tokens=[]; targets=[]; regimes=[]; changes=[]; prev_action=0; prev_reward=0.
    for _ in range(env.episode_length):
        tokens.append(make_token(obs,prev_action,prev_reward,env.n_actions)); targets.append(env.oracle_action())
        obs,reward,done,info=env.step(targets[-1]); regimes.append(info['regime']); changes.append(info['regime_changed']); prev_action=targets[-1]; prev_reward=reward
        if done: break
    return np.asarray(tokens,np.float32),np.asarray(targets,np.int64),np.asarray(regimes,np.int64),np.asarray(changes,bool)

def build_dataset(env_factory,episodes):
    xs=[]; ys=[]
    for _ in range(episodes):
        x,y,_,_=generate_episode(env_factory()); xs.append(x); ys.append(y)
    return np.stack(xs),np.stack(ys)
