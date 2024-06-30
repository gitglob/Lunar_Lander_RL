import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from src.utils import calculate_step_statistics
from .network import ActorCriticNetwork

class PPO:
    def __init__(self, state_dim=8, action_dim=4, 
                 lr=0.001, gamma=0.99, entropy_coef=0.01,
                 k_epochs=4, eps_clip=0.2, grad_clip=False,
                 vf_coef=0.5, vloss_clip=True, use_gae=True, gae_lam=0.95):
        # Initialize the Policy (Actor) network
        self.actor_critic = ActorCriticNetwork(state_dim, action_dim)

        # Move policy and value function networks to CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic.to(self.device)
  
        # Initialize the optimizer
        self.lr = lr
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        
        # Initialize training parameters
        self.gamma = gamma           # Discount Factor
        self.k_epochs = k_epochs     # Number of epochs for updating policy
        self.eps_clip = eps_clip     # Clipping parameter for PPO
        self.grad_clip = grad_clip   # Clip gradient or not
        self.vf_coef = vf_coef       # Value Function Loss coefficient
        self.vloss_clip = vloss_clip # Clip value loss or not
        self.use_gae = use_gae       # Flag to use or not GAE
        self.gae_lam = gae_lam       # GAE coef

        # Entropy configuration
        self.entropy_coef = entropy_coef

    def train(self):
        self.actor_critic.train()

    def eval(self):
        self.actor_critic.eval()

    def best_act(self, state):
        """Select action based on current policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor_critic.best_act(state)
        
        return action.item()

    def act(self, state):
        """Select action based on current policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _ = self.actor_critic.act(state)
            value = self.actor_critic.value(state)
        
        return action.item(), log_prob.item(), value.flatten().item()

    def compute_returns(self, rewards, dones, values, next_value, next_done):
        """Calculate the discounted returns"""
        returns = torch.zeros_like(values).to(self.device)

        for t in reversed(range(len(returns))):
            if t == len(returns) - 1:
                next_return = next_value 
                next_mask = (~next_done).int() 
            else:
                next_return = values[t+1]
                next_mask = (~dones[t+1]).int()
            returns[t] = rewards[t] + self.gamma * next_return * next_mask

        advantages = returns - values
        
        return returns, advantages
        
    def compute_gae(self, rewards, dones, values, next_value, next_done):
        """Compute the Generalized Advantage Estimation (GAE)"""
        # Initialize the advantages tensor
        advantages = torch.zeros_like(values).to(self.device)
        gae = 0
        
        # Iterate in reverse to compute GAE
        for t in reversed(range(len(advantages))):
            if t == len(advantages) - 1:
                next_value = next_value 
                next_mask = (~next_done).int() 
            else:
                next_value = values[t+1]
                next_mask = (~dones[t+1]).int()

            delta = rewards[t] + self.gamma * next_value * next_mask - values[t]
            gae = delta + self.gamma * self.gae_lam * next_mask * gae
            advantages[t] = gae

        returns = advantages + values
        
        return advantages, returns

    def compute_kl_divergence(self, old_log_probs, new_log_probs):
        """Compute the KL divergence between old and new log probabilities."""
        log_ratio = new_log_probs - old_log_probs
        kl_divergence = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
        return kl_divergence.mean()

    def max_entropy(self, action_dim=4):
        return torch.log(torch.tensor(action_dim, dtype=torch.float32)).item()

    def update(self, states, actions, log_probs, values, rewards, dones, next_state, next_done):
        """Update the policy based on collected experiences"""
        states = states.to(self.device).detach()
        actions = actions.to(self.device).detach()
        log_probs = log_probs.to(self.device).detach()
        values = values.to(self.device).detach()
        rewards = rewards.to(self.device).detach()
        dones = dones.to(self.device).detach()
        next_state = next_state.to(self.device).detach()
        next_done = next_done.to(self.device).detach()

        with torch.no_grad():
            # Compute returns and advantages
            next_value = self.actor_critic.value(next_state)
            if self.use_gae:
                advantages, returns = self.compute_gae(rewards, dones, values, next_value, next_done)
            else:
                returns, advantages = self.compute_returns(rewards, dones, values, next_value, next_done)
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Initialize accumulators for logging
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        total_relative_entropy = 0
        total_ac_grad_norms = 0
        total_ac_clip_grad_norms = 0
        total_kl_divergence = 0
        total_residual_variance = 0

        # Save current parameters
        old_actor_critic_params = [param.clone() for param in self.actor_critic.parameters()]
        
        # Perform policy updates for k epochs
        batch_values = torch.zeros_like(values)
        batch_target_values = torch.zeros_like(values)
        for _ in range(self.k_epochs):
            # Use current policy to get the probability of the memory actions
            action_logits = self.actor_critic.policy(states)
            dist = Categorical(logits=action_logits)
            dist_entropy = dist.entropy()
            new_log_probs = dist.log_prob(actions)

            # Compute the ratio of the new and old action probabilities
            ratios = torch.exp(new_log_probs - log_probs)

            # Compute the surrogate (actor) loss (PPO objective)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Use the current value network to get the new values
            new_values = self.actor_critic.value(states).flatten()

            # Critic loss
            if self.vloss_clip:
                diff_unclipped = (new_values - returns) ** 2
                values_clipped = values + torch.clamp(new_values - values, -self.eps_clip, self.eps_clip)
                diff_clipped = (values_clipped - returns) ** 2
                
                max_diff = torch.max(diff_unclipped, diff_clipped)
                
                critic_loss = self.vf_coef * max_diff.mean()
            else:
                critic_loss = self.vf_coef * ((new_values - returns) ** 2).mean()

            # Compute the entropy loss
            entropy = (self.entropy_coef * dist_entropy).mean()
            
            # Overall loss
            loss = actor_loss + critic_loss - entropy

            with torch.no_grad():
                # Keep the predicted values
                batch_values += new_values
                batch_target_values += returns

                # Compute KL divergence
                kl_divergence = self.compute_kl_divergence(log_probs, new_log_probs)
                total_kl_divergence += kl_divergence.item()

                # Compute residual variance
                residual_variance = ((returns - new_values).var() / returns.var()).item()
                total_residual_variance += residual_variance

                # Compute relative policy entropy
                policy_entropy = dist_entropy.mean().item()
                max_ent = self.max_entropy()
                relative_entropy = policy_entropy / max_ent
                total_relative_entropy += relative_entropy
                
                # Accumulate losses
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy_loss += entropy
                total_loss += loss
            
            # Zero the gradients of the optimizer
            self.optimizer.zero_grad()
            
            # Backpropagate the loss
            loss.backward()
            
            # Save the gradients before clipping
            total_ac_grad_norms += self.get_avg_grad(self.actor_critic)
            
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)

            # Save the gradients after clipping
            total_ac_clip_grad_norms += self.get_avg_grad(self.actor_critic)

            # Perform a single optimization step
            self.optimizer.step()

        # Calculate step statistics
        abs_max_ac, mean_square_value_ac = calculate_step_statistics(old_actor_critic_params, 
                                                                     self.actor_critic.parameters())

        # Compute averages over k_epochs
        avg_actor_loss = total_actor_loss / self.k_epochs
        avg_critic_loss = total_critic_loss / self.k_epochs
        avg_entropy_loss = total_entropy_loss / self.k_epochs
        avg_loss = total_loss / self.k_epochs
        avg_relative_entropy = total_relative_entropy / self.k_epochs
        avg_ac_grad_norms = total_ac_grad_norms / self.k_epochs
        avg_ac_clip_grad_norms = total_ac_clip_grad_norms / self.k_epochs
        avg_kl_divergence = total_kl_divergence / self.k_epochs
        avg_residual_variance = total_residual_variance / self.k_epochs
        batch_values = batch_values / self.k_epochs
        batch_target_values = batch_target_values / self.k_epochs

        # Log dictionary
        self.log_dict = {
            "Values/Min": batch_values.min(),
            "Values/Max": batch_values.max(),
            "Values/Mean": batch_values.mean(),
            "Values/Std": batch_values.std(),
            "Target Values/Min": batch_target_values.min(),
            "Target Values/Max": batch_target_values.max(),
            "Target Values/Mean": batch_target_values.mean(),
            "Target Values/Std": batch_target_values.std(),
            "Advantages/Min": advantages.min(),
            "Advantages/Max": advantages.max(),
            "Advantages/Mean": advantages.mean(),
            "Advantages/Std": advantages.std(),
            "Loss/Actor": avg_actor_loss,
            "Loss/Critic": avg_critic_loss,
            "Loss/Entropy": avg_entropy_loss,
            "Loss/Total": avg_loss,
            "Gradient/AC": avg_ac_grad_norms,
            "Gradient/Clipped AC": avg_ac_clip_grad_norms,
            "Network/Abs Max": abs_max_ac,
            "Network/MSE": mean_square_value_ac,
            "Debug/Entropy": avg_relative_entropy,
            "Debug/KL Divergence": avg_kl_divergence,
            "Debug/Residual Variance": 1 - avg_residual_variance,
            "Config/Learning Rate": self.get_lr(),
        }
                
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
        
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr = self.optimizer.param_groups[0]["lr"]

    def save(self, filename):
        torch.save({
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, filename)
