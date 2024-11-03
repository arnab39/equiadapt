from re import A
from tracemalloc import stop
import gym
import os
import hydra
import omegaconf
import wandb
import random
import math

from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from zmq import device

from prepare.gym_cartpole import CartpoleWrapper
from collections import deque

from typing import List, Tuple
from omegaconf import DictConfig
from network import DQN
from utils import ReplayMemory, Transition, load_envs

from tqdm import tqdm


# Setup the environment using a wrapper
def setup_environment(env_hyperparams):
    if env_hyperparams["name"] == "cartpole":
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        env = CartpoleWrapper(env, env_hyperparams)
    return env

# Action selection , if stop training == True, only exploitation
def select_action(dqn, state, steps_done, exp_hyperparams, stop_training):
    dqn.eval()
    sample = random.random()
    eps_threshold = exp_hyperparams["eps_end"] + (exp_hyperparams["eps_start"]- exp_hyperparams["eps_end"]) * \
        math.exp(-1. * steps_done / exp_hyperparams["eps_decay"])
    # print('Epsilon = ', eps_threshold, end='\n')
    if sample > eps_threshold or stop_training:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            q_values = dqn(state)
            action = q_values.max(1)[1].item()
    else:
        action = random.randrange(dqn.num_actions)
    dqn.train()
    return action

def optimize_model(
    memory: ReplayMemory, 
    dqn: DQN, 
    optimizer: optim.Optimizer, 
    target_dqn: DQN, 
    batch_size: int, 
    gamma: float) -> None:
    """
    Optimize the DQN model using the given memory replay buffer.

    Args:
        memory (ReplayMemory): The replay memory buffer.
        dqn (DQN): The DQN model.
        optimizer (optim.Optimizer): The optimizer for updating the model parameters.
        target_dqn (DQN): The target DQN model.
        batch_size (int): The batch size for training.
        gamma (float): The discount factor for future rewards.
    """
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).unsqueeze(1)
    
    device = state_batch.device
    
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), 
        device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )
    
    reward_batch = torch.cat(batch.reward).type(torch.FloatTensor).to(device)
    
    state_action_values = dqn(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=action_batch.device)
    next_state_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    wandb.log({'Loss:': loss.item()})

    optimizer.zero_grad()
    loss.backward()
    for param in dqn.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    

def train_rl(hyperparams: DictConfig) -> None:
    # set system environment variables for wandb
    if hyperparams["wandb"]["use_wandb"]:
        print("Using wandb for logging...")
        os.environ["WANDB_MODE"] = "online"
    else:
        print("Wandb disabled for logging...")
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DIR"] = hyperparams["wandb"]["wandb_dir"]
    os.environ["WANDB_CACHE_DIR"] = hyperparams["wandb"]["wandb_cache_dir"]
    
    # initialize wandb
    wandb.init(
        config=OmegaConf.to_container(hyperparams, resolve=True),
        entity=hyperparams["wandb"]["wandb_entity"],
        project=hyperparams["wandb"]["wandb_project"],
        dir=hyperparams["wandb"]["wandb_dir"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = setup_environment(hyperparams["env"])
    env.reset()
    init_screen = env.get_screen().to(device)
    
    _, inchannels, screen_height, screen_width = init_screen.shape
    FRAMES = hyperparams["env"]["frames"]
    input_shape = (FRAMES * inchannels, screen_height, screen_width)
    print("Screen height: ", screen_height," | Width: ", screen_width)

    # Get number of actions from gym action space
    num_actions = env.num_actions
    
    exp_hyperparams = hyperparams["experiment"]
    
    dqn = DQN(input_shape, num_actions).to(device)
    target_dqn = DQN(input_shape, num_actions).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    optimizer = optim.RMSprop(dqn.parameters())
    # optimizer = optim.Adam(dqn.parameters(), lr=exp_hyperparams["learning_rate"])
    memory = ReplayMemory(exp_hyperparams["replay_memory_size"])
    mean_last = deque([0] * exp_hyperparams['last_episodes_num'], exp_hyperparams['last_episodes_num'])
    
    stop_training = False
    count_final = 0   
    steps_done = 0
    episode_durations = []
    num_episodes = exp_hyperparams["num_episodes"]

    # Wrap your range function with tqdm for a progress bar
    for i_episode in tqdm(range(num_episodes), desc="Training Episodes"):
    # for i_episode in range(exp_hyperparams["num_episodes"]):
        # Initialize the environment and state
        env.reset()
        init_screen = env.get_screen().to(device)
        screens = deque([init_screen] * FRAMES, FRAMES)
        state = torch.cat(list(screens), dim=1)

        for t in count():

            # Select and perform an action
            action = select_action(dqn, state, steps_done, exp_hyperparams, stop_training)

            state_variables, _, done, _, _ = env.step(action)
            steps_done += 1

            # Observe new state
            screens.append(env.get_screen().to(device))
            next_state = torch.cat(list(screens), dim=1) if not done else None

            # Reward modification for better stability
            x, x_dot, theta, theta_dot = state_variables
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            reward = torch.tensor([reward], device=device)
            if t >= exp_hyperparams["end_score"] - 1:
                reward = reward + 20
                done = 1
            else: 
                if done:
                    reward = reward - 20 

            # Store the transition in memory
            action = torch.tensor([action], device=device)
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if done:
                episode_durations.append(t + 1)
                mean_last.append(t + 1)
                mean = 0
                wandb.log({'Episode duration': t+1 , 'Episode number': i_episode})
                for i in range(exp_hyperparams['last_episodes_num']):
                    mean = mean_last[i] + mean
                mean = mean/exp_hyperparams['last_episodes_num']
                if mean < exp_hyperparams['training_stop'] and stop_training == False:
                    optimize_model(memory, dqn, optimizer, target_dqn, 
                                    exp_hyperparams["batch_size"], 
                                    exp_hyperparams["gamma"])
                else:
                    stop_training = True
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % exp_hyperparams['target_update'] == 0:
            target_dqn.load_state_dict(dqn.state_dict())
        if stop_training == True:
            count_final += 1
            if count_final >= 100:
                break
    
    print('Training Complete')
    env.close()
    
# load the variables from .env file
load_envs()
    
@hydra.main(config_path=str("./configs/"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    train_rl(cfg)

if __name__ == '__main__':
    main()
