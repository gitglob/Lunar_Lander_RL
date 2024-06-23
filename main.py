import os
import gymnasium as gym
import wandb
from src.agent import PPO
from src.running_stats import RunningStats
from src.discounted_rewards import DiscountedRewards


def test(agent, running_stats, render_mode=None):
    """Test the agent in the given environment and log the total reward."""
    env_test = gym.make("LunarLander-v2", render_mode=render_mode)
    state = env_test.reset()[0]  # Reset the environment to get the initial state
    state = running_stats.normalize(state)  # Normalize the initial state
    if render_mode:
        env_test.render()
    done = False            # Flag to check if the episode has ended
    total_reward = 0        # Variable to accumulate the total reward
    while not done:
        action, _ = agent.act(state)               # Get the action from the agent
        state, reward, done, truncated, info = env_test.step(action)  # Take a step in the environment
        state = running_stats.normalize(state)  # Normalize the initial state
        total_reward += reward                      # Accumulate the reward
    env_test.close()
    return total_reward

def train(env_train, agent, n_episodes, 
          n_steps, save_every, batch_size, 
          log_interval, save_path, running_stats):
    """Train the PPO agent in the given environment."""
    # Initialize the Discounted Rewards
    discounted_rewards = DiscountedRewards()

    # Iterate over the number of episodes
    timestep = 0
    update_counter = 0
    for episode in range(n_episodes):
        state = env_train.reset()[0]
        running_stats.update(state)
        norm_state = running_stats.normalize(state)
        for step in range(n_steps):
            timestep += 1 
            # Take an action on the real environment with the current policy
            action, log_prob = agent.act(norm_state)
            # Get the rewards from the real environment
            next_state, reward, done, truncated, info = env_train.step(action)

            # Normalize state and reward
            norm_reward = discounted_rewards.update(reward)
            running_stats.update(next_state)
            norm_next_state = running_stats.normalize(next_state)

            # Store the following:
            # state: the current real state
            # action: the action that the current policy proposed
            # reward: the current real reward
            # next state: the real next state (state + action -> next_state)
            # done: if the episode ends
            # log_prob: the policy's probability for each action
            agent.store_transition(state, action, 
                                   norm_reward, norm_next_state, 
                                   done, log_prob)
            norm_state = norm_next_state
            
            # Break if the episode is over
            if done or timestep % batch_size == 0:
                break

        # Update the agent's actor and critic networks
        if timestep % batch_size == 0:
            running_stats.reset()
            discounted_rewards.reset()
            update_counter += 1
            agent.update()
            if update_counter % save_every == 0:
                agent.save(save_path)

        # Log metrics every n episodes
        if (episode + 1) % log_interval == 0 and update_counter > 0:
            total_reward = test(agent, running_stats)
            print(f"Episode {episode + 1}/{n_episodes},",
                  f"Reward: {total_reward}")
            agent.log_dict["Episode Reward"] = total_reward
            wandb.log(agent.log_dict)


def main():
    # Initialize the Lunar Lander environment 
    env_train = gym.make("LunarLander-v2", render_mode=None)
    state_dim = env_train.observation_space.shape[0]
    action_dim = env_train.action_space.n

    # Initialize wandb
    run_id = "v11-norm_states_disc_rewards_clip_lr_anneal2"
    wandb.init(project="lunar-lander-ppo", entity="gitglob", resume='allow', id=run_id)

    # Set and log the hyperparameters
    lr_a=0.001
    lr_c=0.001
    gamma=0.99
    entropy_coef = 0.01
    entropy_decay_rate = 0
    k_epochs=4
    eps_clip=0.2
    wandb.config.update({
        "lr_a": lr_a,
        "lr_c": lr_c,
        "gamma": gamma,
        "entropy_decay_rate": entropy_decay_rate,
        "entropy_coef": entropy_coef,
        "eps_clip": eps_clip,
        "k_epochs": k_epochs
    }, allow_val_change=True)

    # Initialize the Agent
    agent = PPO(state_dim, action_dim, 
                lr_a, lr_c, gamma, 
                entropy_decay_rate, entropy_coef,
                k_epochs, eps_clip)

    # Load the model if a checkpoint exists
    save_path = "checkpoints/ppo_model_" + str(run_id) + ".pth"
    if os.path.exists(save_path):
        agent.load(save_path)
        print(f"Model loaded from {save_path}")

    # Training parameters
    n_episodes = 20000
    n_steps = 500
    batch_size = 100
    save_every = 5
    log_interval = 25

    # Initialize state running stats
    running_stats = RunningStats()

    # Train the agent
    train(env_train, agent, n_episodes, 
          n_steps, save_every, batch_size, 
          log_interval, save_path, running_stats)

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
