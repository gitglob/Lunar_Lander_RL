import os
import gymnasium as gym
import torch
import torch.optim as optim
import wandb
from src.dataloader import EnvDataLoader
from src.agent import PPO


def test_agent(env, agent):
    """Test the agent in the given environment and log the total reward."""
    env.render()            # Render the environment to display the graphical interface
    state = env.reset()[0]  # Reset the environment to get the initial state
    done = False            # Flag to check if the episode has ended
    total_reward = 0        # Variable to accumulate the total reward
    while not done:
        action, _ = agent.act(state)               # Get the action from the agent
        state, reward, done, truncated, info = env.step(action)  # Take a step in the environment
        total_reward += reward                      # Accumulate the reward
    return total_reward

def train(env, agent, n_episodes, n_steps, log_interval, save_path):
    """Train the PPO agent in the given environment."""
    optimizer = optim.Adam(agent.policy.parameters(), lr=agent.lr)
    for episode in range(n_episodes):
        state = env.reset()[0]
        for step in range(n_steps):
            action, log_prob = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done, log_prob)
            state = next_state
            
            if done:
                break

        if (episode + 1) % log_interval == 0:
            total_reward = test_agent(env, agent)
            agent.save(save_path)
            print(f'Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}')
            # Log metrics to wandb
            wandb.log({'Total Reward': total_reward})

def main():
    # Initialize the Lunar Lander environment 
    # human or rgb_array render for visualization
    env = gym.make('LunarLander-v2', render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize wandb
    wandb.init(project="lunar-lander-ppo", entity="gitglob")

    # Initialize the Agent
    agent = PPO(state_dim, action_dim)

    # Load the model if a checkpoint exists
    save_path = 'checkpoints/ppo_model.pth'
    if os.path.exists(save_path):
        agent.load(save_path)
        print(f'Model loaded from {save_path}')

    # Training parameters
    n_episodes = 1000
    n_steps = 2000
    log_interval = 20

    # Train the agent
    train(env, agent, n_episodes, n_steps, log_interval, save_path)

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
