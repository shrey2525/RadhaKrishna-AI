import numpy as np, torch

def rolling_mean(v,w):
    if len(v)<w:return np.array([])
    return np.convolve(v,np.ones(w)/w,mode='valid')

def adaptation_time(rewards,changes,window=20,recovery_fraction=.90):
    threshold=recovery_fraction; vals=[]
    for change in changes:
        post=rewards[change+1:]; rm=rolling_mean(post,window)
        found=np.where(rm>=threshold)[0]
        vals.append(int(found[0]+window) if len(found) else None)
    valid=[v for v in vals if v is not None]
    return {'n_changes':len(vals),'n_recovered':len(valid),'mean_adaptation_time':float(np.mean(valid)) if valid else None}

def evaluate_agent(env,agent,kind,episodes=100,window=20,recovery_fraction=.90,device='cpu'):
    agent.eval(); rewards_all=[]; details=[]
    for _ in range(episodes):
        obs=env.reset(); prev_action=0; prev_reward=0.; hidden=None; history=[]; rewards=[]; changes=[]
        for t in range(env.episode_length):
            token=np.concatenate([obs.astype(np.float32),np.eye(env.n_actions,dtype=np.float32)[prev_action],np.array([prev_reward],np.float32)])
            if kind=='stateless': logits=agent(torch.tensor(obs,dtype=torch.float32,device=device).unsqueeze(0))
            elif kind=='transformer':
                history.append(token); x=torch.tensor(np.asarray(history[-agent.max_len:],np.float32),device=device).unsqueeze(0); logits=agent(x)
            else:
                x=torch.tensor(token,dtype=torch.float32,device=device).view(1,1,-1); logits,hidden=agent(x,hidden)
            action=int(torch.argmax(logits,-1).item()); obs,reward,done,info=env.step(action); rewards.append(reward)
            if info['regime_changed']: changes.append(t)
            prev_action=action; prev_reward=reward
            if done: break
        rewards=np.asarray(rewards,np.float32); rewards_all.append(float(rewards.mean())); details.append(adaptation_time(rewards,changes,window,recovery_fraction))
    valid=[d['mean_adaptation_time'] for d in details if d['mean_adaptation_time'] is not None]
    return {'mean_reward':float(np.mean(rewards_all)),'std_reward':float(np.std(rewards_all)),'total_regime_changes':sum(d['n_changes'] for d in details),'recovered_changes':sum(d['n_recovered'] for d in details),'mean_adaptation_time':float(np.mean(valid)) if valid else None}
