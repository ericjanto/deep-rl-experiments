-- Deep RL Experiments --
This repository contains five reinforcement learning experiments:

| experiments/ | rl area             | description                           |
| ------------ | ------------------- | ------------------------------------- |
| experiment1  | dynamic programming | value and policy iteration            |
| experiment2  | tabular rl          | Q-learning; on-policy first-visit MC  |
| experiment3  | deep rl             | DQN; REINFORCE                        |
| experiment4  | continuous deep rl  | DDPG                                  |
| experiment5  | misc                | fine-tuning and hyperparameter search |

-- Setup --
conda create -n rl_course python=3.7
conda activate rl_course
pip install -e .