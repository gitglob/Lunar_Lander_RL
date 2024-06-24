import os
import wandb
import torch
import gymnasium as gym
from src.utils import normalize_and_clip_rewards, corr
from src.agent import PPO


def test(agent, render_mode=None):
    """Test the agent in the given environment and log the total reward."""
    env_test = gym.make("LunarLander-v2", render_mode=render_mode)

    state = env_test.reset()[0]  # Reset the environment to get the initial state
    if render_mode:
        env_test.render()
    done, truncated = False, False    # Flag to check if the episode has ended
    total_reward = 0                  # Variable to accumulate the total reward
    while not (done or truncated):
        action, _ = agent.act(state)  # Get the action from the agent
        state, reward, done, truncated, info = env_test.step(action)
        total_reward += reward        # Accumulate the reward
    env_test.close()
    return total_reward

def train(env_train, agent, 
          num_updates, batch_size, 
          save_path, lr_anneal):
    """Train the PPO agent in the given environment."""
    # Get initial state
    state = env_train.reset()[0]

    # Iterate over the requested updates
    done, truncated, episode_counter = False, False, 0
    for update in range(num_updates):
        # Anneal Learning Rate
        if lr_anneal:
            frac = 1.0 - (update - 1.0) / num_updates
            agent.anneal_lr(frac)

        # Run an episode
        batch_states = torch.empty((batch_size, len(state)))
        batch_action_counts = torch.zeros((4))
        batch_rewards = torch.empty((batch_size))
        batch_next_states = torch.empty((batch_size, len(state)))
        batch_masks = torch.empty((batch_size))
        batch_log_probs = torch.empty((batch_size))
        batch_values = torch.empty((batch_size))
        for step in range(batch_size):
            # Reset environment if the episode terminated
            if done or truncated:
                episode_counter += 1
                state = env_train.reset()[0]
            
            # Take an action on the real environment with the current actor's policy
            batch_states[step] = torch.FloatTensor(state)
            action, log_prob = agent.act(state)
            action_counts = torch.bincount(torch.tensor([action]), minlength=4)
            batch_action_counts += action_counts
            batch_log_probs[step] = log_prob
            
            # Get the estimated values of the current step using the critic
            value = agent.criticize(state)
            batch_values[step] = value

            # Get the rewards from the real environment
            next_state, reward, done, truncated, info = env_train.step(action)
            batch_next_states[step] = torch.FloatTensor(next_state)
            batch_rewards[step] = reward
            batch_masks[step] = 1 - done

            # Advance the state
            state = next_state

        # Normalize rewards
        batch_rewards = normalize_and_clip_rewards(batch_rewards)

        # Identify terminal states
        terminal_indices = (batch_masks == 0).nonzero(as_tuple=False).squeeze()

        # Calculate terminal and penultimate terminal correlations
        if len(terminal_indices) > 1:
            terminal_correlation = corr(batch_values[terminal_indices], batch_rewards[terminal_indices])
            penultimate_terminal_correlation = corr(batch_values[terminal_indices - 1], batch_rewards[terminal_indices])
        else:
            terminal_correlation = 0
            penultimate_terminal_correlation = 0

        # Increase the update counter and update agent
        agent.update(batch_states, 
                     batch_log_probs, 
                     batch_values,
                     batch_rewards,
                     batch_next_states, 
                     batch_masks)

        # Save the model every few updates
        if update % 5 == 0:
            agent.save(save_path)

        # Run inference to calculate the total reward
        total_reward = test(agent)
        agent.log_dict["Rewards/Min"] = batch_rewards.min()
        agent.log_dict["Rewards/Max"] = batch_rewards.max()
        agent.log_dict["Rewards/Mean"] = batch_rewards.mean()
        agent.log_dict["Rewards/Std"] = batch_rewards.std()
        agent.log_dict["Values/Min"] = batch_values.min()
        agent.log_dict["Values/Max"] = batch_values.max()
        agent.log_dict["Values/Mean"] = batch_values.mean()
        agent.log_dict["Values/Std"] = batch_values.std()
        agent.log_dict["Episode Reward"] = total_reward
        agent.log_dict["Debug/Terminal Correlation"] = terminal_correlation
        agent.log_dict["Debug/Penultimate Terminal Correlation"] = penultimate_terminal_correlation
        wandb.log(agent.log_dict)
        action_data = {f"action_{i}_count": batch_action_counts[i] 
                       for i in range(len(batch_action_counts))}
        wandb.log(action_data)

        # Log reward every few updates
        if update % 25 == 0:
            print(f"Episode {episode_counter}, Update {update}/{num_updates},",
                f"\tReward: {total_reward}")


def main():
    # Initialize the Lunar Lander environment 
    env_train = gym.make("LunarLander-v2", 
                         max_episode_steps=200,
                         render_mode=None)

    state_dim = env_train.observation_space.shape[0]
    action_dim = env_train.action_space.n

    # Initialize wandb
    run_id = "v20.0.14-debug"
    wandb.init(project="lunar-lander-ppo", 
               entity="gitglob", 
               resume='allow', 
               id=run_id)

    # Set and log the hyperparameters
    lr=1e-5
    lr_anneal = True
    gamma=0.99
    entropy_coef = 0.001
    k_epochs=4
    eps_clip=0.2
    wandb.config.update({
        "lr": lr,
        "gamma": gamma,
        "entropy_coef": entropy_coef,
        "eps_clip": eps_clip,
        "k_epochs": k_epochs
    }, allow_val_change=True)

    # Initialize the Agent
    agent = PPO(state_dim, action_dim, 
                lr, gamma, entropy_coef,
                k_epochs, eps_clip)

    # Load the model if a checkpoint exists
    save_path = "checkpoints/ppo_model_" + str(run_id) + ".pth"
    if os.path.exists(save_path):
        agent.load(save_path)
        print(f"Model loaded from {save_path}")

    # Training parameters
    num_updates = 5000
    batch_size = 1024#128

    # Train the agent
    train(env_train, agent, 
          num_updates, batch_size, 
          save_path, lr_anneal)

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
