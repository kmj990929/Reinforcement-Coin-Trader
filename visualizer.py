import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

basepath = "/content/Reinforcement-Coin-Trader"
FILENAME = basepath + "/info/dqn_freq1_epoch10_eplen500_0.0001_0.9999_433538.04642786185_LS_606_460.info"
info = np.load(FILENAME, allow_pickle=True).all()
portfolio = [data[3] for data in info['history']]
action = [data[0] for data in info['history']]
coin_num = [data[5] for data in info['history']]
position = [data[6] for data in info['history']]

plt.figure(figsize=(15,10))


# positions = show_position(position)
path = basepath + "/data/test"
file_list = os.listdir(path)
for file in file_list:
  if file.endswith(".csv"):
    df = pd.read_csv(path+'/'+file, thousands=',', converters={'time': lambda x: str(x)})
    # open, high, low, close, volume, value
    # df = df.iloc[0:100]
    plt.subplot(2,1,1)
    plt.plot(df.open, label="open")
    plt.plot(df.high, label="high")
    plt.plot(df.low, label="low")
    plt.plot(df.close, label="close")
    plt.ylabel("CoinChart", fontsize=20)
plt.legend(fontsize=10, loc="upper left")
plt.title("ddqn_trained", loc="right",pad=20)

x = list(range(0,16000))
colors = ['tomato', 'darkturquoise', 'darkseagreen']

plt.subplot(2,1,2)

i = np.argmax(portfolio) 
j = np.argmin(portfolio) 
plt.plot(portfolio)
plt.ylabel('Portfolio', fontsize=20)
plt.plot([i, j], [portfolio[i], portfolio[j]], 'o', color='Red', markersize=10)
plt.axhline(1000000, linestyle='-', color='gray')

# redraw flat
flats = []
shorts = []
longs = []
prev_pos = 3
start_idx = 0
end_idx = 0
# count pos
for idx, pos in enumerate(position):
    if pos == prev_pos:
        end_idx = idx
        continue

    if idx == len(position)-1:
        pos = prev_pos+1

    if prev_pos == 0:    #long ended
        longs.append((start_idx, end_idx))
    elif prev_pos == 1: # short ended
        shorts.append((start_idx, end_idx))
        prev_pos = 0
    elif prev_pos == 2: # flat ended
        flats.append((start_idx, end_idx))
    prev_pos = pos
    start_idx = idx

poses = [longs, shorts, flats]
labels= ["long", "short"]

for i in range(2):
    for j, idxes in enumerate(poses[i]):
        start, end = idxes
        if j == 0:
            plt.fill_between(x[start:end], portfolio[start:end], label = labels[i], color=colors[i], alpha=0.2)
        else:
            plt.fill_between(x[start:end], portfolio[start:end], color=colors[i], alpha=0.2)
plt.legend(fontsize=10,loc="upper left")
#fig.patch.set_facecolor('xkcd:white')  # 배경 색 하얗게.
plt.show()