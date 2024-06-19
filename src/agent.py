import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .network import PolicyNetwork, ValueNetwork

class PPO:
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99, k_epochs=4, eps_clip=0.2):
        # Initialize the Policy (Actor) network
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        
        # Initialize value function (critic) network
        self.value_function = ValueNetwork(state_dim)
        self.value_function_old = ValueNetwork(state_dim)

        # Move policy and value function networks to CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.value_function.to(self.device)
        self.policy_old.to(self.device)
        self.value_function_old.to(self.device)

        # Initially copy the weights from the current networks to the old networks
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_function_old.load_state_dict(self.value_function.state_dict())
    
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize training parameters
        self.lr = lr              # Learning Rate
        self.gamma = gamma        # Discount Factor
        self.k_epochs = k_epochs  # Number of epochs for updating policy
        self.eps_clip = eps_clip  # Clipping parameter for PPO

        # Initialize logging parameters
        self.memory = []          # Buffer to store experiences
        self.log_interval = 20    # Interval for logging

        # Keep track of loss
        self.loss = None

    def act(self, state):
        # state: [x, y, v_x, v_y, theta, omega, grounded_left, grounded_right]
        # Select action based on current policy
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _ = self.policy_old.act(state)
        
        return action.item(), log_prob
    
    def evaluate(self, states, actions):
        # Evaluate actions taken using current policy
        _, log_probs, entropy = self.policy.act(states)
        state_value = self.value_function(states, actions)

        return log_probs, torch.squeeze(state_value), entropy
    
    def compute_returns(self, next_value, rewards, masks):
        """Calculates the discounted sum of future rewards."""
        # Initialize R with the value of the next state (used for bootstrapping)
        R = next_value
        # Initialize an empty list to store the returns
        returns = []

        # Loop over the rewards in reverse order
        for step in reversed(range(len(rewards))):
            # Update R using the reward at the current step and the previously computed return
            R = rewards[step] + self.gamma * R * masks[step]
            # Insert the updated return at the beginning of the returns list
            returns.insert(0, R)

        # Return the list of computed returns
        return returns
    
    def update(self):
        """Update the policy based on collected experiences"""
        ## Collect Experiences from the episode
        # Extract rewards from memory
        rewards = [m['reward'] for m in self.memory]
        # Extract states from memory
        states = [m['state'] for m in self.memory]
        # Extract actions from memory
        actions = [m['action'] for m in self.memory]
        # Extract log probabilities from memory
        log_probs = [m['log_prob'] for m in self.memory]
        # Extract masks (indicating whether the episode continues) from memory
        masks = [m['mask'] for m in self.memory]
        # Extract the last action
        if len(actions) == 0:
            breakpoint()
        last_action = actions[-1].unsqueeze(0)
        if not isinstance(last_action, torch.Tensor):
            last_action = torch.LongTensor(last_action).to(self.device).unsqueeze(0)
        # Extract the next state from the last memory entry
        next_state = self.memory[-1]['next_state'].unsqueeze(0)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)

        ## Compute the discounted returns for each state in the episode
        # Compute the value of the next state using the critic network, without tracking gradients
        with torch.no_grad():
            next_value = self.value_function(next_state, last_action)  # Get value of next state
        
        # Compute the returns for each time step in the episode using the rewards and masks
        returns = self.compute_returns(next_value, rewards, masks)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Stack log probabilities into a single tensor and move to the appropriate device, detaching from the computation graph
        log_probs = torch.stack(log_probs).to(self.device).detach()
        
        # Compute the state values using the critic network
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions)
        values = self.value_function(states, actions).squeeze()
        
        ## Calculate the advantage for each state-action pair
        # Compute the advantage by subtracting state values (estimated returns) from (actual) returns
        advantage = returns - values
        
        ## Perform policy updates for k epochs
        for _ in range(self.k_epochs):
            # Evaluate the current policy with the states and actions
            action_logprobs, state_values, dist_entropy = self.evaluate(states, actions)
            
            # Compute the ratio of the new and old action probabilities
            ratios = torch.exp(action_logprobs.detach() - log_probs.detach())
            
            # Compute the surrogate loss (PPO objective)
            surr1 = ratios * advantage.detach()
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage.detach()
            
            # Compute the overall loss: the minimum of surrogate losses, value loss, and entropy bonus
            self.loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, returns) - 0.01 * dist_entropy

            # Zero the gradients of the optimizer
            self.optimizer.zero_grad()
            
            # Backpropagate the loss
            self.loss.mean().backward()
            
            # Perform a single optimization step
            self.optimizer.step()
        
        # Update old policy and value function networks
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_function_old.load_state_dict(self.value_function.state_dict())

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

        # Update the policy if the memory buffer is full
        if len(self.memory) >= self.log_interval:
            self.update()
            avg_reward = np.mean([m['reward'] for m in self.memory])

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value_function.load_state_dict(checkpoint['value_function'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save(self, filename):
        torch.save({
            'policy': self.policy.state_dict(),
            'value_function': self.value_function.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename)
