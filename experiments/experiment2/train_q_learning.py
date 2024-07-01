import gymnasium as gym
from tqdm import tqdm
from csv import writer

from experiments.constants import EX2_QL_CONSTANTS as CONSTANTS
from experiments.experiment2.agents import QLearningAgent
from experiments.experiment2.utils import evaluate
from experiments.util.hparam_sweeping import generate_hparam_configs

CONFIG = {
    "eval_freq": 1000, # keep this unchanged
    "alpha": 0.05,
    "epsilon": 0.9,
    "gamma": 0.99,
}

CONFIG.update(CONSTANTS)

HPARAMS = {
    "gamma": [0.99, 0.8],
}

RENDER = False # FALSE FOR FASTER TRAINING / TRUE TO VISUALIZE ENVIRONMENT DURING EVALUATION
SWEEP = False # TRUE TO SWEEP OVER POSSIBLE HYPERPARAMETER CONFIGURATIONS
NUM_SEEDS_SWEEP = 10 # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGHTS = False # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED

SWEEP_RESULTS_FILE = "Q-Learning-sweep-results.csv"

def q_learning_eval(
        env,
        config,
        q_table,
        render=False):
    """
    Evaluate configuration of Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    if render:
        eval_env = gym.make(CONFIG["env"], render_mode="human")
    else:
        eval_env = env
    return evaluate(eval_env, eval_agent, config["eval_eps_max_steps"], config["eval_episodes"])


def train(env, config):
    """
    Train and evaluate Q-Learning on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table
    """
    agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    for eps_num in tqdm(range(1, config["total_eps"]+1)):
        obs, _ = env.reset()
        episodic_return = 0
        t = 0

        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            act = agent.act(obs)
            n_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            agent.learn(obs, act, reward, n_obs, done)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = q_learning_eval(env, config, agent.q_table)
            tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])

    if SWEEP:
        config_list, swept_params = generate_hparam_configs(CONFIG, HPARAMS)
        results = []

        if SWEEP_RESULTS_FILE:
            with open(SWEEP_RESULTS_FILE, 'w', newline='') as file:
                w = writer(file)
                w.writerow(["hparams", "last_mean_return"])

        for config in config_list:
            hparams_values = '_'.join([':'.join([key, str(config[key])]) for key in swept_params])
            print("\nStarting new run...")
            all_last_returns = []
            for i in range(NUM_SEEDS_SWEEP):
                print(f"\nTraining iteration: {i+1}/{NUM_SEEDS_SWEEP}")
                _, evaluation_return_means, _, _ = train(env, config)

                all_last_returns.append(evaluation_return_means[-1])

                if SWEEP_RESULTS_FILE:
                    with open(SWEEP_RESULTS_FILE, 'a', newline='') as file:
                        w = writer(file)
                        w.writerow([hparams_values, evaluation_return_means[-1]])
            # calc mean of last returns and std dev
            mean_last_return = sum(all_last_returns) / len(all_last_returns)
            std_dev_last_return = (sum([(x - mean_last_return) ** 2 for x in all_last_returns]) / len(all_last_returns)) ** 0.5
            print(f"hparams: {hparams_values}, mean last return: {mean_last_return}, std dev last return: {std_dev_last_return}")
    else:
         total_reward, _, _, q_table = train(env, CONFIG)