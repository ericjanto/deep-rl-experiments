import gymnasium as gym
from typing import List, Tuple

import numpy as np

from experiments.experiment4.agents import DDPG
from experiments.experiment4.evaluate_ddpg import evaluate
from experiments.experiment5.train_ddpg \
    import RACETRACK_CONFIG

RENDER = False

CONFIG = RACETRACK_CONFIG

if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    # filename1 = "DDPG--racetrack-v0--policy_learning_rate-0.01_critic_learning_rate-0.01_critic_hidden_size-[32, 32, 32]_policy_hidden_size-[32, 32, 32]_gamma-tensor(0.7000)_tau-tensor(0.1000)--1.pt"
    # filename2 = "DDPG--racetrack-v0--policy_learning_rate-0.01_critic_learning_rate-0.01_critic_hidden_size-[32, 32, 32]_policy_hidden_size-[32, 32, 32]_gamma-tensor(0.8495)_tau-tensor(0.1000)--0.pt"
    # filename3 = "DDPG--racetrack-v0--policy_learning_rate-0.01_critic_learning_rate-0.01_critic_hidden_size-[32, 32, 32]_policy_hidden_size-[32, 32, 32]_gamma-tensor(0.8495)_tau-tensor(1.)--1.pt"

    # new_files = ["DDPG--racetrack-v0--policy_learning_rate-0.001_critic_learning_rate-0.01_critic_hidden_size-[64, 64, 64]_policy_hidden_size-[64, 64, 64]_gamma-0.8495_tau-0.1_batch_size-64_buffer_capacity-10000000--0.pt",]
    # new_files = ["DDPG--racetrack-v0--policy_learning_rate-0.001_critic_learning_rate-0.01_critic_hidden_size-[64, 64, 64]_policy_hidden_size-[64, 64, 64]_gamma-0.8495_tau-0.1_batch_size-64_buffer_capacity-10000000--1.pt",]
                #  "DDPG--racetrack-v0--policy_learning_rate-0.001_critic_learning_rate-0.001_critic_hidden_size-[64, 64, 64]_policy_hidden_size-[64, 64, 64]_gamma-0.8495_tau-0.1_batch_size-64_buffer_capacity-10000000--0.pt"]

    returns = evaluate(env, CONFIG)
    print(returns)
    print(np.max(returns))
    env.close()
