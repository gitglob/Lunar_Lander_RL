import os
import numpy as np
import random
import wandb
import torch
import gymnasium as gym
from src.utils import corr
from src.agent import PPO
from src.normalizer import NormalizedEnv


def test(agent, env):
    """Test the agent in the given environment and log the total reward."""
    agent.eval()
    state = env.reset()  # Reset the environment to get the initial state
    done, truncated = False, False    # Flag to check if the episode has ended
    total_reward = 0                  # Variable to accumulate the total reward
    while not (done or truncated):
        action, _, _ = agent.act(state)      # Get the action from the agent
        state, reward, done, truncated, info = env.step(action)
        total_reward += info["real_reward"]  # Accumulate the reward

    return total_reward

def train(env, agent, 
          num_updates, batch_size, 
          save_path, lr_anneal):
    """Train the PPO agent in the given environment."""
    # Iterate over the requested updates
    done, truncated, episode_counter = False, False, 0
    terminal_correlation = 0
    penultimate_terminal_correlation = 0
    for update_counter in range(num_updates):
        # Anneal Learning Rate
        if lr_anneal:
            frac = 1.0 - (update_counter - 1.0) / num_updates
            agent.anneal_lr(frac)

        # Get initial state
        state = env.reset()
        done, truncated = False, False

        # Run batch_size steps
        episode_reward = 0
        episode_length = 0
        batch_states = torch.zeros((batch_size, len(state)))
        batch_action_counts = torch.zeros((4))
        batch_rewards = torch.zeros((batch_size))
        batch_next_states = torch.zeros((batch_size, len(state)))
        batch_masks = torch.zeros((batch_size))
        batch_log_probs = torch.zeros((batch_size))
        batch_values = torch.zeros((batch_size))
        agent.train()
        for step in range(batch_size):
            # Reset environment if the episode terminated
            if done or truncated:
                wandb.log({"Episode/Length": episode_length})
                wandb.log({"Episode/Counter": episode_counter})
                wandb.log({"Episode/Train Reward": episode_reward})
                state = env.reset()
                done, truncated = False, False
                episode_reward = 0
                episode_length = 0
                episode_counter += 1
            
            batch_masks[step] = 1 - (done or truncated)
            
            # Take an action on the real environment with the actor's policy
            # and evaluate it with the critic
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
        total_reward = test(agent, env)
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
            agent.save(save_path)

def train_wrapper(run_id):
    # Initialize the Lunar Lander environment 
    env = gym.make("LunarLander-v2", render_mode=None)
    env = NormalizedEnv(env)

    # Seeding
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    env.action_space.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Standard agentparameters
    k_epochs = 4
    eps_clip = 0.2
    # Sweep parameters
    config = wandb.config
    lr = config.lr
    gamma = config.gamma
    entropy_coef = config.entropy_coef
    grad_clip = config.grad_clip
    vf_coef = config.vf_coef
    vloss_clip = config.vloss_clip
    use_gae = config.use_gae
    gae_lam = config.gae_lam

    # Initialize the Agent
    agent = PPO(state_dim, action_dim, 
                lr, gamma, entropy_coef,
                k_epochs, eps_clip, 
                grad_clip, vf_coef, vloss_clip,
                use_gae, gae_lam)

    # Load the model if a checkpoint exists
    save_path = "checkpoints/ppo_model_" + str(run_id) + ".pth"
    if os.path.exists(save_path):
        agent.load(save_path)
        print(f"Model loaded from {save_path}")

    # Training parameters
    num_updates = 5000
    batch_size = config.batch_size
    lr_anneal = config.lr_anneal

    # Train agent
    train(env, agent, 
          num_updates, batch_size, 
          save_path, lr_anneal)


def main():
    # Initialize wandb
    project_name = "lunar-lander-ppo"

    # Initialize wandb
    run_id = "v23.0.0"
    wandb.init(project=project_name, 
               entity="gitglob", 
               resume='allow', 
               id=run_id)
    
    # Sweep parameters
    wandb.config.update({
        "batch_size": 1024,#128,
        "lr": 2.5e-5,#0.01587,
        "gamma": 0.99,
        "entropy_coef": 0.01,#.1,
        "vf_coef": 0.5,
        "vloss_clip": True,
        "grad_clip": True,
        "lr_anneal": True,
        "use_gae": True,
        "gae_lam": 0.95
    }, allow_val_change=True)

    # Train the agent
    train_wrapper(run_id)
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
