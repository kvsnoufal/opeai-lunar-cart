import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
from gym.wrappers import Monitor
import glob
import io
import base64
from pyvirtualdisplay import Display

from gym import wrappers
# from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers,models
import os
from tensorflow.keras import backend as K

tf.config.experimental_run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# strategy = tf.distribute.MirroredStrategy()

print("*****************")
print(gpus)
print("*****************")
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
from gym.wrappers import Monitor
import glob
import io
import base64
from pyvirtualdisplay import Display

from gym import wrappers
# from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers,models
import os
from tensorflow.keras import backend as K

tf.config.experimental_run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# strategy = tf.distribute.MirroredStrategy()

print("*****************")
print(gpus)
print("*****************")
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

MEMORYLEN=int(1e5)
MEMORYLEN=int(10000)
BATCHSIZE=32
EPOCHS=1
# UPDATE_EVERY = 4


class DQNAgent():
    def __init__(self,actions=6,obs=(4,80,80)):
        self.actions=actions
        self.observations=obs
        self.model=self.load_model()
        self.target_model=self.load_model()
        
        self.copy_weights()

        self.memory=deque(maxlen=MEMORYLEN)
        self.gamma=0.99
        self.patience=0
        
           
    def play(self,observation,epsilon):
        if (len(self.memory)<BATCHSIZE):
            
            action=np.random.randint(low=0,high=self.actions)
            return action
        else:
            if np.random.random()>epsilon:
#                 print("model")
                action=self.model_predictions(observation)
            else:
                action=np.random.randint(low=0,high=self.actions)
            return action
            
    def step(self,state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        if ((len(self.memory)>=BATCHSIZE) & (np.random.random() < 0.25 )):
            self.train_model()
        pass
 
    
    def train_model(self):
        rnd_indices = np.random.choice(len(self.memory), size=BATCHSIZE)
        data=np.array(self.memory)[rnd_indices]
        np.random.shuffle(data)
        
        state, action, reward, next_state, done=np.stack(data[:,0]),np.stack(data[:,1]),np.stack(data[:,2]),np.stack(data[:,3]),np.stack(data[:,4])
        qnext_max=np.max(self.target_model.predict(next_state),axis=1)
        qnext_max=reward+ self.gamma*qnext_max*(1-done)
        qtable_to_update=self.target_model.predict(state)
        for indx,qs in enumerate(qtable_to_update):
            qtable_to_update[indx,action[indx]]=qnext_max[indx]
        self.model.fit(state,qtable_to_update,epochs=1,verbose=0)
        self.patience+=1
        if self.patience==100:
            self.copy_weights()
            self.patience=0
        
        pass
    def model_predictions(self,observation):
        pred=self.model.predict(observation.reshape(1,4,80,80))
        pred=np.argmax(pred)
        return pred
        
    def load_model(self):
        num_input = layers.Input(shape=(self.observations))
        x=layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', data_format="channels_first")(num_input)
        x=layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                     data_format='channels_first')(x)
        x=layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                     data_format='channels_first')(x)
        x=layers.Flatten()(x)
        x = layers.Dense(128,activation="relu")(x)
        y = layers.Dense(self.actions, activation="linear")(x)
        model = models.Model(inputs=num_input, outputs=y)
        model.compile(loss="mse",optimizer=tf.keras.optimizers.Adam(lr=0.01,decay=0.01))
        model.summary()
        return model
    def copy_weights(self):
        self.target_model.set_weights(self.model.get_weights()) 


def proc_(state):
    state=0.299*state[:,:,0] + 0.587*state[:,:,1] + \
                    0.114*state[:,:,2]
    return state[35:195:2, ::2]
import time
from tqdm import tqdm
starttime=time.time()
scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores
n_episodes=5000
agent=DQNAgent()

eps_start=1.0
eps_end=0.02
eps_decay=0.997


eps = eps_start
env=gym.make('PongNoFrameskip-v4')
eps_history=[]
best_score=-9999
for i_episode in range(1, n_episodes+1):
    # print("Epsilon: {}".format(eps))
    state = env.reset()
    score = 0
    
    state_m=deque(maxlen=4)
    if(len(state_m))==0:
        for i in range(4):
            state_m.append(np.zeros(proc_(state).shape))

    next_state_m=state_m.copy()
    next_state_m.append(proc_(state))
    state_m.append(proc_(state))
    
    
    while True:
        action = agent.play(np.array(state_m),eps)
        next_state, reward, done, _ = env.step(action)
        next_state_m.append(proc_(next_state))
        if reward<0:
            reward=-50
            done=True
        else:
            reward=1

        agent.step(state_m, action, reward, next_state_m, done)
        
        state = next_state
        state_m.append(proc_(state))
        score += reward
        if done:
            break 
    
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps)
    eps_history.append(eps)
    if score>best_score:
        agent.model.save_weights("./weightfolder/best_wt_pong.h5")
    if i_episode % 10 == 0:
            print("Time: {}".format(time.time()-starttime))
            print('\rEpisode {}\tAverage Score: {:.2f} Epsilon {}'.format(i_episode, np.mean(scores_window),eps))
    if np.mean(scores_window)>=400.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            
            break
endtime= time.time() 
print(endtime-starttime)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
# plt.show()
plt.savefig("deepQpong.png")
agent.model.save_weights("./weightfolder/final_wt_pong.h5")