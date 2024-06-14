import torch

# Custom DataLoader
class EnvDataLoader(torch.utils.data.IterableDataset):
    def __init__(self, env, agent, n_steps):
        self.env = env
        self.agent = agent
        self.n_steps = n_steps
    
    def __iter__(self):
        state = self.env.reset()
        for _ in range(self.n_steps):
            action, log_prob = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            yield state, action, reward, next_state, done, log_prob
            state = next_state if not done else self.env.reset()
            