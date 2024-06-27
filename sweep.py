import wandb
import torch
import gymnasium as gym
from src.utils import corr
from src.normalizer import NormalizedEnv
from src.agent import PPO


def test(env, agent):
    """Test the agent in the given environment and log the total reward."""
    state = env.reset()  # Reset the environment to get the initial state
    done, truncated = False, False    # Flag to check if the episode has ended
    total_reward = 0                  # Variable to accumulate the total reward
    while not (done or truncated):
        action, _, _ = agent.act(state)         # Get the action from the agent
        state, reward, done, truncated, info = env.step(action)
        total_reward += info["real_reward"]  # Accumulate the reward

    return total_reward

def train(env, agent, num_updates, batch_size, lr_anneal):
    """Train the PPO agent in the given environment."""   
    # Iterate over the requested updates
    done, truncated, episode_counter = False, False, 0
    for update_counter in range(num_updates):
        # Anneal Learning Rate
        if lr_anneal:
            frac = 1.0 - (update_counter - 1.0) / num_updates
            agent.anneal_lr(frac)

        # Get initial state
        state = env.reset()

        # Run an episode
        episode_reward = 0
        episode_length = 0
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
                wandb.log({"Episode/Length": episode_length})
                wandb.log({"Episode/Counter": episode_counter})
                wandb.log({"Episode/Train Reward": episode_reward})
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_counter += 1
            
            # Take an action on the real environment with the current actor's policy
            batch_states[step] = torch.FloatTensor(state)
            action, log_prob, value = agent.act(state)
            action_counts = torch.bincount(torch.tensor([action]), minlength=4)
            batch_action_counts += action_counts
            batch_log_probs[step] = log_prob
            batch_values[step] = value

            # Get the rewards from the real environment
            next_state, reward, done, truncated, info = env.step(action)
            batch_next_states[step] = torch.FloatTensor(next_state)
            batch_rewards[step] =  reward
            episode_reward += info["real_reward"]
            batch_masks[step] = 1 - done
            episode_length += 1

            # Advance the state
            state = next_state

        # Identify terminal states
        terminal_indices = (batch_masks == 0).nonzero(as_tuple=False).squeeze()

        # Calculate terminal and penultimate terminal correlations
        if terminal_indices.dim() != 0 and len(terminal_indices) > 1:
            terminal_correlation = corr(batch_values[terminal_indices], batch_rewards[terminal_indices])
            penultimate_terminal_correlation = corr(batch_values[terminal_indices - 1], batch_rewards[terminal_indices])

        # Increase the update counter and update agent
        agent.update(batch_states, 
                     batch_log_probs, 
                     batch_values,
                     batch_rewards,
                     batch_next_states, 
                     batch_masks)

        # Run inference to calculate the total reward
        total_reward = test(env, agent)
        agent.log_dict["States/Min"] = batch_states.min()
        agent.log_dict["States/Max"] = batch_states.max()
        agent.log_dict["States/Mean"] = batch_states.mean()
        agent.log_dict["States/Std"] = batch_states.std()
        agent.log_dict["Rewards/Min"] = batch_rewards.min()
        agent.log_dict["Rewards/Max"] = batch_rewards.max()
        agent.log_dict["Rewards/Step"] = batch_rewards.mean()
        agent.log_dict["Rewards/Std"] = batch_rewards.std()
        agent.log_dict["Episode/Test Reward"] = total_reward
        agent.log_dict["Debug/Terminal Correlation"] = terminal_correlation
        agent.log_dict["Debug/Penultimate Terminal Correlation"] = penultimate_terminal_correlation
        wandb.log(agent.log_dict)
        action_data = {f"action/count_{i}": batch_action_counts[i] 
                       for i in range(len(batch_action_counts))}
        wandb.log(action_data)

        # Log reward every few updates
        if update_counter % 25 == 0:
            print(f"Episode {episode_counter}, Update {update_counter}/{num_updates},",
                f"\tReward: {total_reward}")

def train_wrapper():
    wandb.init()
    config = wandb.config

    # Initialize the Lunar Lander environment 
    env = gym.make("LunarLander-v2", render_mode=None)
    env = NormalizedEnv(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Standard agentparameters
    k_epochs = 4
    eps_clip = 0.2
    gamma = 0.99
    # Sweep parameters
    lr = config.lr
    entropy_coef = config.entropy_coef
    grad_clip = config.grad_clip

    # Initialize the Agent
    agent = PPO(state_dim, action_dim, 
                lr, gamma, entropy_coef,
                k_epochs, eps_clip, grad_clip)

    # Train parameters
    num_updates = 500
    batch_size = config.batch_size
    lr_anneal = config.lr_anneal

    # Train agent
    train(env, agent, num_updates, batch_size, lr_anneal)


def main():
    # Initialize wandb
    project_name = "lunar-lander-ppo"
    
    # Set the sweep
    sweep_config = {
        'method': 'random',  # or 'grid', 'bayes'
        'metric': {
            'name': 'Episode/Train Reward',
            'goal': 'maximize'
        },
        'parameters': {
            "batch_size": {"values": [16, 128, 512]},
            'lr': {"max": 0.1, "min": 0.0001},
            'entropy_coef': {'values': [0, 0.1, 0.01, 0.001]},
            'grad_clip': {'values': [True, False]},
            'lr_anneal': {'values': [True, False]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, 
                           project=project_name)

    # Train the agent
    wandb.agent(sweep_id, train_wrapper, count=200)


if __name__ == "__main__":
    main()
