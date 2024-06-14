import gym
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from src.dataloader import EnvDataLoader
from src.agent import PPO

# Callback for testing
class TestCallback(Callback):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    
    def on_validation_epoch_end(self, trainer, pl_module):
        state = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = self.agent.act(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        trainer.logger.experiment.log({'test_reward': total_reward})

def main():
    # Training
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, action_dim)

    wandb_logger = WandbLogger(project='lunar-lander-ppo')
    trainer = pl.Trainer(max_epochs=1000, logger=wandb_logger, callbacks=[TestCallback(env, agent)])

    dataloader = torch.utils.data.DataLoader(EnvDataLoader(env, agent, n_steps=2000), batch_size=None)

    trainer.fit(agent, dataloader)

if __name__ == "__main__":
    main()
