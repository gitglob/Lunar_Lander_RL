import torch

# Custom DataLoader
class EnvDataLoader(torch.utils.data.IterableDataset):
    def __init__(self, env, agent, n_steps):
        """
        Initializes the custom DataLoader.
        
        Args:
            env: The environment instance (e.g., a Gym environment).
            agent: The agent that interacts with the environment.
            n_steps: Number of steps to run in the environment for each iteration.
        """
        self.env = env          # Store the environment instance
        self.agent = agent      # Store the agent instance
        self.n_steps = n_steps  # Store the number of steps for each iteration
    
    def __iter__(self):
        """
        Provides an iterator to generate experience tuples.
        
        Yields:
            Tuple containing state, action, reward, next_state, done, and log_prob.
        """
        state = self.env.reset()[0]  # Reset the environment to get the initial state
        for _ in range(self.n_steps):
            action, log_prob = self.agent.act(state)                 # Get action and log probability from the agent
            next_state, reward, done, truncated, info = self.env.step(action)      # Take a step in the environment
            yield state, action, reward, next_state, done, log_prob  # Yield the experience tuple
            state = next_state if not done else self.env.reset()[0]  # Update the state, reset if episode is done
