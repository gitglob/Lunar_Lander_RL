import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from .network import PPOActorCritic

# PPO Agent
class PPO(pl.LightningModule):
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99, k_epochs=4, eps_clip=0.2):
        super(PPO, self).__init__()
        self.policy = PPOActorCritic(state_dim, action_dim)
        self.policy_old = PPOActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.lr = lr
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip

        self.memory = []
        self.log_interval = 20

    def forward(self):
        pass
    
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _ = self.policy_old.act(state)
        return action, log_prob
    
    def evaluate(self, states, actions):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        return self.policy.evaluate(states, actions)
    
    def compute_returns(self, next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns
    
    def update(self):
        rewards = [m['reward'] for m in self.memory]
        states = [m['state'] for m in self.memory]
        actions = [m['action'] for m in self.memory]
        log_probs = [m['log_prob'] for m in self.memory]
        masks = [m['mask'] for m in self.memory]
        next_state = self.memory[-1]['next_state']
        
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.policy.critic(next_state)
        returns = self.compute_returns(next_value, rewards, masks)
        
        returns = torch.FloatTensor(returns).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device).detach()
        values = self.policy.critic(torch.FloatTensor(states).to(self.device)).squeeze()
        advantage = returns - values
        
        for _ in range(self.k_epochs):
            action_logprobs, state_values, dist_entropy = self.evaluate(states, actions)
            
            ratios = torch.exp(action_logprobs - log_probs.detach())
            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, returns) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory = []

    def training_step(self, batch, batch_idx):
        state, action, reward, next_state, done = batch
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'mask': 1 - done,
            'log_prob': log_prob
        })

        if len(self.memory) >= self.log_interval:
            self.update()
            avg_reward = np.mean([m['reward'] for m in self.memory])
            self.log('reward', avg_reward)
            return avg_reward
    
    def configure_optimizers(self):
        return self.optimizer