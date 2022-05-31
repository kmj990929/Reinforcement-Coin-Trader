import process_data
import pandas as pd
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import deque
import random

# position constant
LONG = 0
SHORT = 1
FLAT = 2

# action constant
BUY = 0
SELL = 1
HOLD = 2

# data constant
RISE = 110
FALL = 10550
BOX = 13130

class OHLCVEnv(gym.Env):

    def __init__(self, infoname, window_size, path, ep_len, train=True, show_trade=True):
        self.train= train
        self.ep_len = ep_len
        self.show_trade = show_trade
        self.path = path
        self.actions = ["LONG", "SHORT", "FLAT"]
        self.fee = 0.0005
        self.leverage = 10
        self.n = 0
        self.seed()
        self.file_list = []

        self.infoname = infoname

        self.datapath=path+"/data/test/"
        if(train):
            self.datapath = path+"/data/train/"

        # n_features
        self.window_size = window_size

        # load_csv
        self.load_from_csv()
        self.ep_count=0

        self.n_features = self.df.shape[1]
        self.shape = (self.window_size, self.n_features+4)

        # defines action space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def load_from_csv(self):
        names = [x.name for x in Path(self.datapath).iterdir() if x.is_file()]
        raw_df= pd.read_csv(self.datapath + names.pop())
        extractor = process_data.FeatureExtractor(raw_df)
        self.source_df = extractor.add_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features

        ## selected manual fetuares
        feature_list = [
            'bar_hc',
            'bar_ho',
            'bar_hl',
            'bar_cl',
            'bar_ol',
            'bar_co', 'close']
        self.source_df.dropna(inplace=True) # drops Nan rows
        self.closingPrices = self.source_df['close'].values
        self.source_df = self.source_df[feature_list].values

        start_point = set()
        while len(start_point) < 3:
            random_point = random.randint(0, self.source_df.shape[0] - self.ep_len - self.window_size -1)
            start_point.add(random_point)
        self.file_list = list(start_point)
        # self.file_list = [FALL, RISE, BOX]

        self.df = self.source_df.copy()
        self.source_cp = self.closingPrices.copy()


    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normalize_frame(self, frame):
        offline_scaler = StandardScaler()
        observe = frame[..., :-4]
        observe = offline_scaler.fit_transform(observe)
        agent_state = frame[..., -4:]
        temp = np.concatenate((observe, agent_state), axis=1)
        return temp

    def step(self, action):
        s, r, d, i = self._step(action)
        self.state_queue.append(s)
        return self.normalize_frame(np.concatenate(tuple(self.state_queue))), r, d, i

    def _step(self, action):
        if self.portfolio <= 0:
            self.done = True
            self.reward =  -10000000 
        if self.done:
            return self.state, self.reward, self.done, {}

        self.reward = 0

        self.action = HOLD  # hold
        if action == BUY: # buy
            if self.position == FLAT: # if previous position was flat
                self.position = LONG # update position to long
                self.action = BUY # record action as buy
                self.entry_price = self.closingPrice # maintain entry price

                # long 진입
                borrow = self.krw_balance*self.leverage #씨드머니
                self.n = float(borrow/self.entry_price)  #코인개수
                self.n_long += 1
                    
            elif self.position == SHORT: # if previous position was short
                # 수익 = 거래한 코인 개수*(가격 이익) - 원래 있던 현금
                self.exit_price = self.closingPrice
                self.reward = self.n*(self.entry_price - self.exit_price)
                seed_money = self.krw_balance
                self.krw_balance += self.reward
                self.reward = self.reward / seed_money
                self.position = FLAT    # long 한 번이 종료되었으므로 flat으로 전환
                self.action = BUY
                self.entry_price = 0 # clear entry price
        
        elif action == SELL: # vice versa for short trade
            if self.position == FLAT:
                self.position = SHORT   # short의 시작
                self.action = SELL
                self.entry_price = self.closingPrice

                # short 진입
                borrow = self.krw_balance*self.leverage #빌린 액수(빌렸다가)
                self.n = float(borrow/self.entry_price)  #빌린코인개수
                self.n_short += 1 # record number of short
                    
            elif self.position == LONG:
                # 수익 = 거래한 코인 개수*(가격 이익) - 원래 있던 현금 
                self.exit_price = self.closingPrice

                self.reward = self.n * (self.exit_price - self.entry_price)
                seed_money = self.krw_balance
                self.krw_balance += self.reward
                self.reward = self.reward / seed_money
                self.position = FLAT    # long 한 번이 종료되었으므로 flat으로 전환
                self.action = SELL
                self.entry_price = 0

        if(self.position == LONG):
            temp_reward = self.n * (self.closingPrice - self.entry_price)
            new_portfolio = self.krw_balance + temp_reward
            self.reward = temp_reward / self.krw_balance
        elif(self.position == SHORT):
            temp_reward = self.n * (self.entry_price - self.closingPrice)
            new_portfolio = self.krw_balance + temp_reward
            self.reward = temp_reward / self.krw_balance
        else:
            temp_reward = 0
            new_portfolio = self.krw_balance

        self.portfolio = new_portfolio
        self.current_tick += 1
        if(self.show_trade and self.current_tick%100 == 0):
            print("Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
        self.history.append((self.action, self.current_tick, self.closingPrice, self.portfolio, self.reward, self.n, self.position))
        self.state = self.updateState()
        info = {'portfolio':np.array([self.portfolio]),
                                                    "history":self.history,
                                                    "n_trades":{'long':self.n_long, 'short':self.n_short}}
                                    
        if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
            self.done = True
            self.reward = self.get_profit() # return reward at end of the game
            if(self.train == False):
                np.array([info]).dump(
                    '{4}/info/{3}_{0}_LS_{1}_{2}.info'.format(self.portfolio,
                                                                self.n_long,
                                                                self.n_short,
                                                                self.infoname,
                                                                self.path))
        return self.state, self.reward, self.done, info

    def get_profit(self):
        return self.reward

    def reset(self):
        self.ep_count += 1
        if self.train:
            start_point = self.file_list[self.ep_count%3]
            self.df = self.source_df[start_point:start_point+self.ep_len+1][:]
            self.closingPrices = self.source_cp[start_point:start_point+self.ep_len+1][:]
        else:
            self.df = self.source_df
            self.closingPrices = self.source_cp
        self.current_tick = 0

        print("start episode ... {0} at {1}" .format(self.file_list[self.ep_count%3], self.current_tick))

        # positions
        self.n_long = 0
        self.n_short = 0

        # clear internal variables
        self.history = [] # keep buy, sell, hold action history
        self.krw_balance = 100 * 10000 # initial balance, u can change it to whatever u like
        self.portfolio = float(self.krw_balance) # (coin * current_price + current_krw_balance) == portfolio
        self.profit = 0
        self.closingPrice = self.closingPrices[self.current_tick]

        self.action = HOLD
        self.position = FLAT
        self.done = False

        self.state_queue = deque(maxlen=self.window_size)
        self.state = self.preheat_queue()
        return self.state


    def preheat_queue(self):
        while(len(self.state_queue) < self.window_size):
            # rand_action = random.randint(0, len(self.actions)-1)
            rand_action = 2
            s, r, d, i= self._step(rand_action)
            self.state_queue.append(s)
        return self.normalize_frame(np.concatenate(tuple(self.state_queue)))

    def updateState(self):
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]
        self.closingPrice = float(self.closingPrices[self.current_tick])
        prev_position = self.position
        one_hot_position = one_hot_encode(prev_position,3)
        profit = self.get_profit()
        # append two
        state = np.concatenate((self.df[self.current_tick], one_hot_position, [profit]))
        return state.reshape(1,-1)