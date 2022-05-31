from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
from pandas._libs.tslibs.period import freq_to_dtype_code

from tensorforce.agents import PPOAgent
from tensorforce.agents import DoubleDQNAgent
from tensorforce.agents import RandomAgent
from tensorforce.agents import A2CAgent
from tensorforce.agents import DPGAgent, DQNAgent

from tensorforce.execution import Runner
from env.gymWrapper import create_btc_env

import os
import sys
import argparse

# Global variable
FREQ = 10
EP = 100
LR = 1e-4
DISCOUNT = 0.9999
BASEDIR = "/content/drive/MyDrive/tf_deep_rl_trader"
NAME = None

def create_network_spec():
    network_spec = dict(
        type='auto',
        size=512,
        depth=3
    )
    return 'auto'

def create_baseline_spec():
    baseline_spec = dict(   # lstm-dnn-dnn인데 dnn-dnn-lstm으로 바뀜.
        type='auto',
        size=32,
        depth=2,
        rnn=20
    )
    return 'auto'

def set_agent(agent_type, environment, network_spec, baseline_spec):
    agent = None
    environment.actions()['num_values'] = environment.actions()['num_actions']
    del environment.actions()['num_actions']

    if (agent_type == 'ppo'):
        agent = PPOAgent(
        max_episode_timesteps=16000,
        batch_size = 32,
        states=environment.states(),
        actions=environment.actions(),

        discount=DISCOUNT,
        network=network_spec,

        exploration=dict(
                type='linear',
                unit='episodes',
                num_steps=EP,
                initial_value=1.0,
                final_value=0,
            ),
        
        update_frequency=FREQ, #custom, update_mode[frequency] -> update_frequency
        
        baseline=baseline_spec, # baseline=dict(type='custom', network=baseline_spec),
        baseline_optimizer=dict(
          optimizer='adam',
          learning_rate=LR,
          multi_step=5,
        ),
        # PGLRModel
        likelihood_ratio_clipping=0.2,
        subsampling_fraction=0.2,  # 0.1
        multi_step=10, # custom,optimization_steps -> multi_step instead
        summarizer = dict(directory='/content/drive/MyDrive/tf_deep_rl_trader/record',
                          filename= f"{NAME}",
                          max_summaries = 5 #default 5
                          # summaries="all" 
                          ),
        config = dict(device='GPU')
    )

    elif (agent_type == 'ddqn'):
        agent = DoubleDQNAgent(
            # required
            states=environment.states(),
            actions=environment.actions(),
            batch_size = 32,
            memory=50000,
            # Environment
            max_episode_timesteps=16000,
            # network
            network=network_spec,
            # optimization
            update_frequency=FREQ,
            learning_rate = LR,
            # Reward estimation
            discount=DISCOUNT,
            # Traget network
            #Preprocessing
            #exploration
            exploration=dict(
                type='linear',
                unit='episodes',
                num_steps=EP,
                initial_value=1.0,
                final_value=0,
            ),
            summarizer = dict(directory='/content/drive/MyDrive/tf_deep_rl_trader/record',
                          filename= f"{NAME}",
                          max_summaries = 5 #default 5
                          # summaries="all" 
                          ),
            # Regularization
            #parallel interactions
            #config, saver, sumeraizaer, tracking, recorder
        )
    elif (agent_type == 'a2c'):
        agent = A2CAgent(
            states = environment.states(),
            actions = environment.actions(),
            max_episode_timesteps=16000,
            batch_size=32,

            discount = 0.9999,
            exploration=dict(
                type='linear',
                unit='episodes',
                num_steps=EP,
                initial_value=1.0,
                final_value=0,
            ),
            summarizer = dict(directory='/content/drive/MyDrive/tf_deep_rl_trader/record',
                          filename= f"{NAME}",
                          max_summaries = 5 #default 5
                          # summaries="all" 
                          ),
            update_frequency=FREQ
        )
    elif (agent_type == 'dpg'):
        agent = DPGAgent(
            states = environment.states(),
            actions = environment.set_dpg_actions(),
            batch_size=32,
            memory=50000,
            summarizer = dict(directory='/content/drive/MyDrive/tf_deep_rl_trader/record',
                          filename= f"{NAME}",
                          max_summaries = 5 #default 5
                          # summaries="all" 
                          ),
            update_frequency=FREQ
        )
    elif (agent_type == 'random'):
        agent = RandomAgent(
            max_episode_timesteps=16000,
            states=environment.states(),
            actions=environment.actions(),
        )

    elif (agent_type == 'dqn'):
        agent = DQNAgent(
            states = environment.states(),
            actions = environment.actions(),
            memory=50000,
            batch_size=32,

            discount = DISCOUNT,
            exploration=dict(
                type='linear',
                unit='episodes',
                num_steps=EP,
                initial_value=1.0,
                final_value=0,
            ),
            summarizer = dict(directory='/content/drive/MyDrive/tf_deep_rl_trader/record',
                          filename= f"{NAME}",
                          max_summaries = 5 #default 5
                          # summaries="all" 
                          ),
            update_frequency=FREQ
        )
    return agent


def main():
    # create environment for train and test
    PATH_TRAIN = BASEDIR + "/data/train/"
    PATH_TEST = BASEDIR+ "/data/test/"
    TIMESTEP = 30 # window size

    # get parameters
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    parser.add_argument('--agent', choices=['ppo', 'a2c', 'dpg', 'ddqn', 'random', 'dqn'])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--exploration', type=float, default=0.3)
    parser.add_argument('--episode', type=int, default=30)
    parser.add_argument('--discount', type=float, default=0.9999)
    parser.add_argument('--freq', type=int, default=1)
    parser.add_argument('--ep_len', type=int, default=500)
    args = parser.parse_args()

    global EP, LR, FREQ, DISCOUNT, NAME
    EP = args.episode*3
    LR = args.lr
    FREQ = args.freq*3
    DISCOUNT = args.discount
    NAME = f"{args.agent}_freq{args.freq}_epoch{args.episode}_eplen{args.ep_len}_{args.lr}_{args.discount}"

    # set environments
    environment = create_btc_env(infoname = NAME, window_size=TIMESTEP, path=BASEDIR, ep_len = args.ep_len, train=True)
    test_environment = create_btc_env(infoname = NAME, window_size=TIMESTEP, path=BASEDIR, ep_len = args.ep_len, train=False)

    network_spec = create_network_spec()
    baseline_spec = create_baseline_spec()

    # agent
    # option1: initial create
    agent = set_agent(agent_type=args.agent, environment=environment, network_spec= network_spec, baseline_spec= baseline_spec)
    # # option2: load model
    # agent = Agent.load(directory=BASEDIR + f"trained/trained_0.001", format='numpy',environment=environment)
    
    # train
    train_runner = Runner(agent=agent, environment=environment)
    train_runner.run(num_episodes=EP) #, num_timesteps=16000, num_updates = FREQ) #, episode_finished=episode_finished)
    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        ep=train_runner.num_episodes,
        ar=np.mean(train_runner.episode_returns[-100:])
        )
    )

    # test
    test_runner = Runner(agent=agent,environment=test_environment)
    test_runner.run(num_episodes=1)
    
    # save
    # agent.save(directory=BASEDIR + f"/trained/{NAME}", format='numpy', append='episodes')
    # agent.close()


if __name__ == '__main__':
    main()