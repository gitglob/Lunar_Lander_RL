import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from .network import PolicyNetwork, ValueNetwork

class PPO:
    def __init__(self, state_dim, action_dim, 
                 lr_a=0.001, lr_c=0.001, gamma=0.99, 
                 entropy_decay_rate=0, entropy_coef=0.01,
                 k_epochs=4, eps_clip=0.2):
        # Initialize the Policy (Actor) network
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.actor_old = PolicyNetwork(state_dim, action_dim)
        
        # Initialize value function (critic) network
        self.critic = ValueNetwork(state_dim)
        self.critic_old = ValueNetwork(state_dim)

        # Move policy and value function networks to CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_old.to(self.device)
        self.critic_old.to(self.device)

        # Initialize weights
        self.actor.apply(self.init_orthogonal)
        self.critic.apply(self.init_orthogonal)

        # Initially copy the weights from the current networks to the old networks
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())
    
        # Initialize the optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_c)

        # Define learning rate schedulers
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=1000, gamma=0.8)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=1000, gamma=0.8)
        
        # Initialize training parameters
        self.gamma = gamma        # Discount Factor
        self.k_epochs = k_epochs  # Number of epochs for updating policy
        self.eps_clip = eps_clip  # Clipping parameter for PPO

        # Entropy configuration
        self.entropy_coef = entropy_coef
        self.final_entropy_coef = 0.01 if entropy_decay_rate!=0 else entropy_coef
        self.entropy_decay_rate = entropy_decay_rate

        # GAE configuration
        self.lam = 0.95

        # Initialize logging parameters
        self.memory = []                         # Buffer to store experiences

        # Keep track of loss
        self.loss = None

    def act(self, state):
        # state: [x, y, v_x, v_y, theta, omega, grounded_left, grounded_right]
        # Select action based on current policy
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _ = self.actor_old.act(state)
        
        return action.item(), log_prob
    
    def evaluate(self, states):
        # Evaluate actions taken using current policy
        _, log_probs, entropy = self.actor.act(states)
        state_value = self.critic(states)

        return log_probs, torch.squeeze(state_value), entropy
    
    def compute_gae(self, rewards, values, masks, next_value):
        values = values + [next_value]
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            advantages.insert(0, gae)
        return advantages
    
    def compute_returns(self, next_value, rewards, masks):
        """Calculates the discounted sum of future rewards."""
        R = next_value
        returns = []

        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.append(R)

        returns = returns[::-1]
        return returns
    
    def update(self):
        """Update the policy based on collected experiences"""
        # Collect Experiences from the episode
        rewards = [m['reward'] for m in self.memory]
        # clipped_rewards = [np.clip(m['reward'], -1, 1) for m in self.memory]
        states = [m['state'] for m in self.memory]
        states = torch.stack(states).to(self.device)
        actions = [m['action'] for m in self.memory]
        actions = torch.stack(actions).to(self.device)
        log_probs = [m['log_prob'] for m in self.memory]
        log_probs = torch.stack(log_probs).to(self.device)
        masks = [m['mask'] for m in self.memory]

        # Extract the last action and state
        last_action = actions[-1].unsqueeze(0)
        if not isinstance(last_action, torch.Tensor):
            last_action = torch.LongTensor(last_action).to(self.device).unsqueeze(0)
        last_state = self.memory[-1]['next_state'].unsqueeze(0)
        if not isinstance(last_state, torch.Tensor):
            last_state = torch.FloatTensor(last_state).to(self.device).unsqueeze(0)

        # Compute the value of the first future state
        with torch.no_grad():
            future_value = self.critic(last_state).cpu()

        # Compute the returns for each time step in the episode using the rewards and masks
        returns = self.compute_returns(future_value, rewards, masks)
        returns = torch.FloatTensor(returns).to(self.device)
                
        # Compute the state values using the critic network
        values = self.critic(states).squeeze()
        
        # Compute the advantage by subtracting state values (estimated returns) from (actual) returns
        advantage = returns.detach() - values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # Initialize accumulators for logging
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        total_entropy = 0
        total_actor_grad_norms = 0
        total_critic_grad_norms = 0
        total_actor_clip_grad_norms = 0
        total_critic_clip_grad_norms = 0
        
        # Perform policy updates for k epochs
        for _ in range(self.k_epochs):
            # Evaluate the current policy with the states and actions
            action_logprobs, state_values, dist_entropy = self.evaluate(states)
            
            # Compute the ratio of the new and old action probabilities
            ratios = torch.exp(action_logprobs - log_probs.detach())
            
            # Compute the surrogate loss (PPO objective)
            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            
            # Compute the overall loss: the minimum of surrogate losses, value loss, and entropy bonus
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = nn.MSELoss()(state_values, returns)
            self.entropy_coef = max(self.final_entropy_coef, self.entropy_coef * self.entropy_decay_rate)
            entropy = self.entropy_coef * dist_entropy
            loss = actor_loss + critic_loss - entropy
            
            # Accumulate losses
            total_actor_loss += actor_loss.mean().item()
            total_critic_loss += critic_loss.mean().item()
            total_entropy += entropy.mean().item()
            total_loss += loss.mean().item()
            
            # Zero the gradients of the optimizer
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            # Backpropagate the loss
            loss.mean().backward()
            
            # Save the gradients before clipping
            total_actor_grad_norms += self.get_avg_grad(self.actor)
            total_critic_grad_norms += self.get_avg_grad(self.critic)
            
            # Clip gradients
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            
            # Save the gradients after clipping
            total_actor_clip_grad_norms += self.get_avg_grad(self.actor)
            total_critic_clip_grad_norms += self.get_avg_grad(self.critic)

            # Perform a single optimization step
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # Update learning rate
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # Compute averages over k_epochs
        avg_actor_loss = total_actor_loss / self.k_epochs
        avg_critic_loss = total_critic_loss / self.k_epochs
        self.avg_loss = total_loss / self.k_epochs
        self.avg_entropy = total_entropy / self.k_epochs
        avg_actor_grad_norms = total_actor_grad_norms / self.k_epochs
        avg_critic_grad_norms = total_critic_grad_norms / self.k_epochs
        avg_actor_clip_grad_norms = total_actor_clip_grad_norms / self.k_epochs
        avg_critic_clip_grad_norms = total_critic_clip_grad_norms / self.k_epochs
        
        # Get the current learning rates
        actor_lr = self.get_lr(self.actor_optimizer)
        critic_lr = self.get_lr(self.critic_optimizer)

        # Log dictionary
        self.log_dict = {
            "Reward": np.mean(rewards),
            "Loss": self.avg_loss,
            "Entropy": self.avg_entropy,
            "Actor Loss": avg_actor_loss,
            "Critic Loss": avg_critic_loss,
            "Actor Gradient": avg_actor_grad_norms,
            "Critic Gradient": avg_critic_grad_norms,
            "Clipped Actor Gradient": avg_actor_clip_grad_norms,
            "Clipped Critic Gradient": avg_critic_clip_grad_norms,
            "Actor Learning Rate": actor_lr,
            "Critic Learning Rate": critic_lr
        }
                
        # Update old policy and value function networks
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # Clear the memory buffer
        self.memory = []

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        # Store the experience in memory
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'mask': 1 - done,
            'log_prob': log_prob
        })

    @staticmethod
    def get_avg_grad(network):
        grad_norms = []
        for param in network.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        return avg_grad_norm

    @staticmethod
    def get_lr(optimizer):
        return optimizer.param_groups[0]["lr"]
        
    @staticmethod
    def init_orthogonal(layer, gain=1.0):
        if isinstance(layer, nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic optimizer"])
        self.entropy_decay_rate = checkpoint["entropy decay rate"]

    def save(self, filename):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor optimizer": self.actor_optimizer.state_dict(),
            "critic optimizer": self.critic_optimizer.state_dict(),
            "entropy decay rate": self.entropy_decay_rate
        }, filename)
