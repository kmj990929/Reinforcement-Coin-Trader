# Development of a cryptocurrency(coin) investment program using reinforcement learning

강화학습 알고리즘 (A2C, PPO, DQN, DDQN)으로 학습한 코인 투자 모델 개발

###  멤버
[김민주](https://github.com/kmj990929), [김태연](https://github.com/tykim5931), [채지영](https://github.com/chaejiyeong)  

## Getting Started
실험은 Google Colab 환경에서 진행할 것을 추천합니다.
1. Google Colab 환경에서 프로젝트를 생성합니다.
2. 다음 명령어를 사용하여 라이브러리를 설치합니다.  
이 때 Warning이 발생할 수 있으나, 실행에는 문제가 없습니다.
```
!pip install tensorforce -U
!pip install tensorflow==1.15.0
!pip install Keras==2.1.2
!pip install keras-rl==0.4.2
!pip install tensorflow-gpu==1.14.0
!pip install talib-binary
!pip install pandas==1.3.5
!pip install gym==0.9.6
!pip install keras==2.6.*
!pip install tensorflow==2.6.0
!pip install matplotlib==2.2.2
```
3. Github 코드를 가져옵니다.
```
https://github.com/kmj990929/Reinforcement-Coin-Trader.git
```
4. `coin_trader.py` 를 실행합니다.
```
# 예시 실행 코드
!python ./Reinforcement-Coin-Trader/coin_trader.py --agent ppo  --episode 3  
```  

## 실행 옵션
coin_trader 실행 시 다양한 옵션으로 모델의 parameter를 변경할 수 있습니다. 기타 파라미터(epsilon 등)은 coin_trader.py 파일에서 직접 수정해야 합니다.  

* --agent : 강화학습 알고리즘을 선택하는 옵션; ppo, a2c, dpg, ddqn, random, dqn 중 선택
* --lr : learning rate; float type; 기본값 0.0001
* --episode : epoch; int type; 기본값 30
* --discount : discount factor; float type; 기본값 0.9999
* --freq : update frequency; int type; 기본값 1


## 그래프 출력
* coin_trader.py가 정상적으로 종료되면 info 폴더에 파일이 생성됩니다.  
(예시 파일은 info/sample 폴더에서 확인할 수 있습니다.)
해당 파일명을 복사하여 visualizer.py의 FILENAME을 수정한 후, `visualizer.py`를 실행하면 해당 모델의 학습 결과를 출력할 수 있습니다.  
(그래프 예시는 sample_result 폴더에서 확인할 수 있습니다.) 
* record 폴더에도 새로운 폴더가 생성되는데, 이 폴더의 이름을 넣고 아래 코드를 실행하면 tensorboard에서 해당 모델의 학습 결과 (loss 등)을 확인할 수 있습니다.
```
%load_ext tensorboard
%tensorboard --logdir="/content/Reinforcement-Coin-Trader/record/a2c_freq1_epoch10_eplen500_0.0001_0.5"
``` 


## 참고 리소스
* Reinforcement Algorithm 구현을 위해 [Tensorforce Library](https://github.com/tensorforce/tensorforce)를 사용했습니다.
* Environment 구현 및 전체 프로젝트 구성에 [tf_deep_rl_trader 프로젝트](https://github.com/miroblog/tf_deep_rl_trader)를 참조했습니다.
