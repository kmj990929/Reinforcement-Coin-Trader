from pathlib import Path
import copy
import process_data

BUY = 0
SELL = 1
HOLD = 2

LONG = 0
SHORT = 1
FLAT = 2

def load_from_csv(): #TFTraderEnv2.py
    raw_df= pd.read_csv("/content/drive/MyDrive/tf_deep_rl_trader/data/test/pureKRW-BTC_5_20220508.csv")
    extractor = process_data.FeatureExtractor(raw_df)
    source_df = extractor.add_bar_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features
    
    return source_df


def make_action(df, idx, direction):
  if idx < 4:
    return HOLD
  
  diff = df['close'].iloc[idx-1] - df['close'].iloc[idx-7]
  if diff > 0:
    if direction:
      return BUY
    else:
      return SELL

  else:
    if direction:
      return SELL
    else:
      return BUY
  

class SimpleAgent():
  def __init__(self, df):
    self.df = df
    self.krw_balance = 1000000
    self.portfolio = float(self.krw_balance)
    self.position = FLAT
    self.n = 0
    self.leverage = 10
    self.n_long = 0
    self.n_short = 0
    #self.ep_count = 0
    self.train = False

    self.reset()

  def reset(self):
        #self.ep_count += 1
        self.current_tick = 0

        #print("start episode ... {0} at {1}" .format(self.file_list[self.ep_count%3], self.current_tick))

        # positions
        self.n_long = 0
        self.n_short = 0

        # clear internal variables
        self.history = [] # keep buy, sell, hold action history
        self.krw_balance = 100 * 10000 # initial balance, u can change it to whatever u like
        self.portfolio = float(self.krw_balance) # (coin * current_price + current_krw_balance) == portfolio

        self.action = HOLD
        self.position = FLAT
        self.done = False

  def print_info(self, info):
      self.done = True
      np.array([info]).dump(
              '{4}/info/{3}_{0}_LS_{1}_{2}.info'.format(self.portfolio,
                                                          self.n_long,
                                                          self.n_short,
                                                          "simple_ver",
                                                          "/content/drive/MyDrive/tf_deep_rl_trader"))

  def _step(self, idx, action):

        if idx != 0 and self.portfolio <= 0:
            self.done = True
        if self.done:
            return self.done, True

        self.reward = 0

        # action comes from the agent
        # 0 buy, 1 sell, 2 hold
        # single position can be opened per trade
        # valid action sequence would be
        # LONG : buy - hold - hold - sell
        # SHORT : sell - hold - hold - buy
        # FLAT : position change
        # invalid action sequence is just considered hold
        # (e.g.) "buy - buy" would be considred "buy - hold"

        closingPrice = self.df['close'].iloc[idx]
        self.action = HOLD  # hold
        if action == BUY: # buy
            if self.position == FLAT: # if previous position was flat
                # long 진입
                self.position = LONG # update position to long
                self.action = BUY # record action as buy
                self.entry_price = closingPrice # maintain entry price

                borrow = self.krw_balance*self.leverage #씨드머니
                self.n = float(borrow/self.entry_price)  #코인개수
                self.n_long += 1
                    
            elif self.position == SHORT: # if previous position was short
                # 수익 = 거래한 코인 개수*(가격 이익) - 원래 있던 현금
                # short의 끝
                self.exit_price = closingPrice
                self.reward = self.n*(self.entry_price - self.exit_price)
                self.krw_balance += self.reward
                self.position = FLAT    # long 한 번이 종료되었으므로 flat으로 전환
                self.action = BUY
                self.entry_price = 0 # clear entry price
        
        elif action == SELL: # vice versa for short trade
            if self.position == FLAT:
                # short 진입
                self.position = SHORT   # short의 시작
                self.action = SELL
                self.entry_price = closingPrice

                borrow = self.krw_balance*self.leverage #빌린 액수(빌렸다가)
                self.n = float(borrow/self.entry_price)  #빌린코인개수
                self.n_short += 1 # record number of short
                    
            elif self.position == LONG:
                # long의 끝
                # 수익 = 거래한 코인 개수*(가격 이익) - 원래 있던 현금 
                self.exit_price = closingPrice

                self.reward = self.n * (self.exit_price - self.entry_price)
                self.krw_balance += self.reward
                self.position = FLAT    # long 한 번이 종료되었으므로 flat으로 전환
                self.action = SELL
                self.entry_price = 0

        if(self.position == LONG):
            temp_reward = self.n * (closingPrice - self.entry_price)
            new_portfolio = self.krw_balance + temp_reward
        elif(self.position == SHORT):
            temp_reward = self.n * (self.entry_price - closingPrice)
            new_portfolio = self.krw_balance + temp_reward
        else:
            temp_reward = 0
            new_portfolio = self.krw_balance

        self.portfolio = new_portfolio
        self.current_tick += 1
        if(self.current_tick%100 == 0):
            print("Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
        self.history.append((self.action, self.current_tick, closingPrice, self.portfolio, self.reward, self.n, self.position))
        
        if idx < 50:

          print(idx, self.reward, end=" ")
          if self.position == 0:
            print("LONG", end =" ")
          elif self.position == 1:
            print("SHORT", end = " ")
          else:
            print("FLAT", end = " ")
          
          if self.action == 0:
            print("BUY")
          elif self.action == 1:
            print("SELL")
          else:
            print("HOLD")

        info = {'portfolio':np.array([self.portfolio]),
                                                    "history":self.history,
                                                    "n_trades":{'long':self.n_long, 'short':self.n_short}}
                                    
        if (self.current_tick >= self.df.shape[0]):
          print("here")
          self.print_info(info)
            
        return self.done, False


df = load_from_csv()
agent = SimpleAgent(df)

for i in range(16000):
  action = make_action(df, i, True)
  done, check = agent._step(i, action)
  if done:
    if check:
      print("break")
    break

