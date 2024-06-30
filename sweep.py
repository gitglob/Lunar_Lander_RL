import os
import wandb
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from src.normalizer import NormalizedEnv
from src.agent import PPO
from src.utils import get_save_subdir


def test(test_env, agent):
    """Test the agent in the given environment and log the total reward."""
    agent.eval()
    state = test_env.reset()[0]
    state = torch.FloatTensor(state)
    done, truncated = False, False
    total_reward = 0
    while not (done or truncated):
        action = agent.best_act(state)
        state, reward, done, truncated, info = test_env.step(action)
        state = torch.FloatTensor(state)
        total_reward += reward
    
    return total_reward

def train(env, agent,
          num_updates, batch_size, 
          save_path, lr_anneal,
          train_video_folder, run_id):
    """Train the PPO agent in the given environment."""    
    # Iterate over the requested updates
    episode_counter = 0
    env = RecordVideo(env, disable_logger=True, video_folder=train_video_folder, episode_trigger=lambda episode_counter: episode_counter % 100 == 0)
    env = gym.wrappers.AutoResetWrapper(env)
    env = NormalizedEnv(env)
    done = True
    for update_counter in range(num_updates):
        # Anneal Learning Rate
        if lr_anneal:
            frac = 1.0 - (update_counter - 1.0) / num_updates
            agent.anneal_lr(frac)

        # Run batch_size steps
        batch_states = torch.empty((batch_size, 8))
        batch_actions = torch.empty((batch_size))
        batch_rewards = torch.empty((batch_size))
        batch_dones = torch.empty((batch_size), dtype=torch.bool)
        batch_log_probs = torch.empty((batch_size))
        batch_values = torch.empty((batch_size))
        agent.train()
        for step in range(batch_size):
            # Check if we need to reset the environment
            if done:
                # Get initial state
                state = env.reset()[0]
                state = torch.FloatTensor(state)
                done = torch.BoolTensor([False])
                episode_length = 0
                episode_reward = 0
                episode_counter += 1
                episode_reward = 0
                episode_action_counts = [0, 0, 0, 0]

            # Save the observations and done masks
            batch_states[step] = state
            batch_dones[step] = done

            # Take an action on the real environment with the actor's policy
            # and evaluate it with the critic
            action, log_prob, value = agent.act(state)
            batch_actions[step] = action
            episode_action_counts[action] += 1
            batch_log_probs[step] = log_prob
            batch_values[step] = value

            # Get the rewards from the real environment
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.FloatTensor(next_state)
            next_done = torch.BoolTensor([terminated or truncated])
            batch_rewards[step] =  reward
            episode_reward += info["real_reward"]
            episode_length += 1

            # Advance the state
            state = next_state
            done = next_done

            # If episode just terminated, log the episode metrics
            if done:
                episode_log = {
                    "Episode/Train Reward": episode_reward,
                    "Episode/Length": episode_length,
                    "action/count_0": episode_action_counts[0],
                    "action/count_1": episode_action_counts[1],
                    "action/count_2": episode_action_counts[2],
                    "action/count_3": episode_action_counts[3],
                }
                wandb.log(episode_log)

        # Increase the update counter and update agent
        agent.update(batch_states, 
                     batch_actions,
                     batch_log_probs, 
                     batch_values,
                     batch_rewards, 
                     batch_dones,
                     next_state,
                     next_done)
        
        # Run inference to calculate the total reward
        win_counter = 0
        test_counter = 0
        total_reward = 0
        test_video_folder = get_save_subdir(f"videos/test/{run_id}")
        test_env = gym.make("LunarLander-v2", render_mode="rgb_array")
        test_env = RecordVideo(test_env, disable_logger=True, video_folder=test_video_folder)
        test_env = NormalizedEnv(test_env, ret=False)
        while test_counter < 10:
            total_reward += test(test_env, agent)
            test_counter += 1
            if total_reward >= 200:
                win_counter += 1
            else:
                break
        test_env.close()
        avg_test_reward = total_reward / test_counter

        # Log metrics
        batch_log = {
            "States/Min": batch_states.min(),
            "States/Max": batch_states.max(),
            "States/Mean": batch_states.mean(),
            "States/Std": batch_states.std(),
            "Rewards/Min": batch_rewards.min(),
            "Rewards/Max": batch_rewards.max(),
            "Rewards/Step": batch_rewards.mean(),
            "Rewards/Std": batch_rewards.std(),
            "Episode/Test Reward": avg_test_reward,
            "Episode/# wins": win_counter
        }
        batch_log.update(agent.log_dict)
        wandb.log(batch_log, commit=False)

        # Log reward every few updates
        if update_counter % 25 == 0:
            print(f"Update {update_counter}/{num_updates}, Episode {episode_counter}",
                  f"\tReward: {avg_test_reward}")
            print(f"\tAction counts: {episode_action_counts}   ({episode_length})")
            agent.save(save_path)

        # Finish if we consistently get more than 250 reward
        if win_counter >= 7:
            print("\n\n\t\t\tEnvironment solved!")
            break

def train_wrapper(run_id):
    # Initialize the Lunar Lander environment 
    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    # Initialize saving directories
    train_video_folder = get_save_subdir(f"videos/train/{run_id}")
    print(f"Train Videos will be saved in: {train_video_folder}")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Parameters
    config = wandb.config
    k_epochs = config.k_epochs
    lr = config.lr
    gamma = config.gamma
    entropy_coef = config.entropy_coef
    grad_clip = config.grad_clip
    vf_coef = config.vf_coef
    vloss_clip = config.vloss_clip
    use_gae = config.use_gae
    gae_lam = config.gae_lam
    eps_clip = config.eps_clip

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
        print(f"Model loaded from {save_path}\n")
    else:
        print("No existing model...\n")

    # Training parameters
    num_updates = 1000
    batch_size = config.batch_size
    lr_anneal = config.lr_anneal

    # Train agent
    train(env, agent, 
          num_updates, batch_size, 
          save_path, lr_anneal,
          train_video_folder, run_id)


def main():
    # Initialize wandb
    project_name = "lunar-lander-ppo"
    
    # Set the sweep
    sweep_config = {
        'method': 'random',  # or 'grid', 'bayes'
        'metric': {
            'name': 'Episode/Test Reward',
            'goal': 'maximize'
        },
        'parameters': {
            "batch_size": {"values": [2048, 1024*10]},
            "lr_anneal": {"value": True},
            "lr": {"value": 0.003743},#{"min": 0.0003, "max": 0.02},
            "k_epochs": {"value": 4},
            "use_gae": {"value": True},
            "gae_lam": {"value": 0.9},
            "gamma": {"value": 0.99},
            "eps_clip": {"value": 0.25},
            "entropy_coef": {"values": [0, 0.1, 0.01]},
            "vloss_clip": {"value": False},
            "vf_coef": {"value": 0.5},
            "grad_clip": {"value": True}
        }
    }
    sweep_id = wandb.sweep(sweep_config, 
                           project=project_name)

    # Train the agent
    wandb.agent(sweep_id, train_wrapper, count=200)


if __name__ == "__main__":
    main()
