import torch
import gymnasium as gym
from src.normalizer import NormalizedEnv
from src.agent import PPO


def test(env, agent, episodes):
    """Test the agent in the given environment and log the total reward."""
    avg_reward = 0
    for e in range(episodes):
        state = env.reset()[0]
        state = torch.FloatTensor(state)
        done, truncated = False, False
        total_reward = 0
        while not (done or truncated):
            action = agent.best_act(state)
            state, reward, done, truncated, info = env.step(action)
            state = torch.FloatTensor(state)
            total_reward += reward
        
        avg_reward += total_reward
        print(f"Total reward: {total_reward}")
    print(f"Average reward over {episodes} runs: {avg_reward/episodes}   (>200 means success)")

def main():
    # Initialize wandb
    model = "r0.0.1"
    model_path = "checkpoints/ppo_model_" + str(model) + ".pth"

    # Initialize the Agent
    agent = PPO()

    # Load the model
    agent.load(model_path)

    # Initialize gym environment
    env = gym.make("LunarLander-v2", render_mode="human")
    env = NormalizedEnv(env, ret=False)

    # Run inference x times with the saved model
    episodes = 1
    test(env, agent, episodes)
    

if __name__ == "__main__":
    main()