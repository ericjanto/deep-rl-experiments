"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import os
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def rl2024_dir():
    path_base = os.path.dirname(os.path.dirname(__file__))
    rl2024_path = os.path.join(path_base, "rl2024")
    return rl2024_path

def test_experiment1(rl2024_dir):
    ex1_path = os.path.join(rl2024_dir, "experiment1")
    init_path = os.path.join(ex1_path, "__init__.py")
    assert os.path.isfile(init_path)
    mdp_solver_path = os.path.join(ex1_path, "mdp_solver.py")
    assert os.path.isfile(mdp_solver_path)

def test_experiment2(rl2024_dir):
    ex2_path = os.path.join(rl2024_dir, "experiment2")
    init_path = os.path.join(ex2_path, "__init__.py")
    assert os.path.isfile(init_path)
    agents_path = os.path.join(ex2_path, "agents.py")
    assert os.path.isfile(agents_path)
    train_mc_path = os.path.join(ex2_path, "train_monte_carlo.py")
    assert os.path.isfile(train_mc_path)
    train_q_path = os.path.join(ex2_path, "train_q_learning.py")
    assert os.path.isfile(train_q_path)

def test_experiment3(rl2024_dir):
    ex3_path = os.path.join(rl2024_dir, "experiment3")
    init_path = os.path.join(ex3_path, "__init__.py")
    assert os.path.isfile(init_path)
    agents_path = os.path.join(ex3_path, "agents.py")
    assert os.path.isfile(agents_path)
    train_dqn_path = os.path.join(ex3_path, "train_dqn.py")
    assert os.path.isfile(train_dqn_path)
    train_reinforce_path = os.path.join(ex3_path, "train_reinforce.py")
    assert os.path.isfile(train_reinforce_path)

def test_experiment4(rl2024_dir):
    ex4_path = os.path.join(rl2024_dir, "experiment4")
    init_path = os.path.join(ex4_path, "__init__.py")
    assert os.path.isfile(init_path)
    agents_path = os.path.join(ex4_path, "agents.py")
    assert os.path.isfile(agents_path)
    train_ddpg_path = os.path.join(ex4_path, "train_ddpg.py")
    assert os.path.isfile(train_ddpg_path)
    racetrack_params_path = os.path.join(ex4_path, "racetrack_latest.pt")
    assert os.path.isfile(racetrack_params_path)

def test_experiment5(rl2024_dir):
    ex5_path = os.path.join(rl2024_dir, "experiment5")
    init_path = os.path.join(ex5_path, "__init__.py")
    assert os.path.isfile(init_path)
    train_ddpg_path = os.path.join(ex5_path, "train_ddpg.py")
    assert os.path.isfile(train_ddpg_path)
    racetrack_params_path = os.path.join(ex5_path, "racetrack_hparam_latest.pt")
    assert os.path.isfile(racetrack_params_path)

def test_answer_sheet(rl2024_dir):
    answer_sheet_path = os.path.join(rl2024_dir, "answer_sheet.py")
    assert os.path.isfile(answer_sheet_path)