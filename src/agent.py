import torch
import torch.nn as nn
import torch.optim as optim
from .network import PolicyNetwork, ValueNetwork

class PPO:
    def __init__(self, state_dim, action_dim, 
                 lr=0.001, gamma=0.99, entropy_coef=0.01,
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
        self.lr = lr
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
        
        # Initialize training parameters
        self.gamma = gamma        # Discount Factor
        self.k_epochs = k_epochs  # Number of epochs for updating policy
        self.eps_clip = eps_clip  # Clipping parameter for PPO

        # Entropy configuration
        self.entropy_coef = entropy_coef

    def act(self, state):
        """Select action based on current policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _ = self.actor_old.act(state)
        
        return action.item(), log_prob
    
    def criticize(self, state):
        """Predict the value of a state using the current critic"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.critic_old.forward(state)
        
        return value

    def compute_returns(self, rewards, masks, next_value):
        """Calculate the discounted returns"""
        R = next_value
        returns = []

        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)

        return torch.tensor(returns).to(self.device)
        
    def update(self, states, log_probs, values, rewards, next_states, masks):
        """Update the policy based on collected experiences"""
        states = states.to(self.device).detach()
        log_probs = log_probs.to(self.device).detach()
        values = values.to(self.device).detach()
        next_states = next_states.to(self.device).detach()
        rewards = rewards.to(self.device).detach()
        masks = masks.to(self.device).detach()

        with torch.no_grad():
            # Compute the returns
            future_value = self.critic(next_states[-1])
            returns = self.compute_returns(rewards, masks, future_value)

            # Compute advantages
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
            # Actor loss
            _, new_log_probs, dist_entropy = self.actor.act(states)
            # Compute the ratio of the new and old action probabilities
            ratios = torch.exp(new_log_probs - log_probs)
            # Compute the surrogate (actor) loss (PPO objective)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            new_values = self.critic(states).flatten()
            critic_loss = nn.MSELoss()(new_values, returns)

            # Compute the entropy loss
            entropy = (self.entropy_coef * dist_entropy).mean()
            
            # Overall loss
            loss = actor_loss + 0.5*critic_loss - entropy
            
            # Accumulate losses
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy += entropy
            total_loss += loss
            
            # Zero the gradients of the optimizer
            self.optimizer.zero_grad()
            
            # Backpropagate the loss
            loss.backward()
            
            # Save the gradients before clipping
            total_actor_grad_norms += self.get_avg_grad(self.actor)
            total_critic_grad_norms += self.get_avg_grad(self.critic)
            
            # Save the gradients after clipping
            total_actor_clip_grad_norms += self.get_avg_grad(self.actor)
            total_critic_clip_grad_norms += self.get_avg_grad(self.critic)

            # Perform a single optimization step
            self.optimizer.step()

        # Compute averages over k_epochs
        avg_actor_loss = total_actor_loss / self.k_epochs
        avg_critic_loss = total_critic_loss / self.k_epochs
        self.avg_loss = total_loss / self.k_epochs
        self.avg_entropy = total_entropy / self.k_epochs
        avg_actor_grad_norms = total_actor_grad_norms / self.k_epochs
        avg_critic_grad_norms = total_critic_grad_norms / self.k_epochs
        avg_actor_clip_grad_norms = total_actor_clip_grad_norms / self.k_epochs
        avg_critic_clip_grad_norms = total_critic_clip_grad_norms / self.k_epochs

        # Log dictionary
        self.log_dict = {
            "Reward": rewards.mean(),
            "Loss": self.avg_loss,
            "Entropy": self.avg_entropy,
            "Actor Loss": avg_actor_loss,
            "Critic Loss": avg_critic_loss,
            "Actor Gradient": avg_actor_grad_norms,
            "Critic Gradient": avg_critic_grad_norms,
            "Clipped Actor Gradient": avg_actor_clip_grad_norms,
            "Clipped Critic Gradient": avg_critic_clip_grad_norms,
            "Learning Rate": self.get_lr(),
        }
                
        # Update old policy and value function networks
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # Clear the memory buffer
        self.memory = []

    @staticmethod
    def get_avg_grad(network):
        grad_norms = []
        for param in network.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        return avg_grad_norm

    def anneal_lr(self, frac):
        self.optimizer.param_groups[0]["lr"] = self.lr * frac

    def set_lr(self, lr):
        self.optimizer.param_groups[0]["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
        
    @staticmethod
    def init_orthogonal(layer, std=2.0, bias=0.0):
        if isinstance(layer, nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, std)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, bias)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save(self, filename):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, filename)
