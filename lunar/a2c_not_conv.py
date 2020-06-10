# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
import os
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from IPython import display
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
from gym.wrappers import Monitor
import glob
import io
import base64
# from pyvirtualdisplay import Display

from gym import wrappers
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers,models
import os
from tensorflow.keras import backend as K
# get_ipython().run_line_magic('matplotlib', 'inline')
tf.config.experimental_run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# strategy = tf.distribute.MirroredStrategy()

print("*****************")
print(gpus)
print("*****************")

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
# virtual_display = Display(visible=0, size=(1400, 900))
# virtual_display.start()


# %%
# env = gym.make('LunarLander-v2')
# obs = env.reset()
# print(obs.shape)
# s=env.render(mode = 'rgb_array')
# # s.shape
# env.close()


# %%
import matplotlib.animation as animation

env = gym.make('LunarLander-v2')
# env = gym.wrappers.Monitor(env,"./notefiles/test",force=True)
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]

scores = []

rewards=[]

obs = env.reset()
cum_reward = 0
frames = []
# fig = plt.figure()
#         obs = env.reset()
#         episode_reward = 0
while True:
    # f=plt.imshow(env.render(mode = 'rgb_array'),animated=True)
    # frames.append([f])

#         q_values = agent.get_q(obs)
    action = env.action_space.sample()

    obs, reward, done, _ = env.step(action)
#             print(t,action,reward)
    cum_reward += reward
#     print(reward)
    rewards.append(reward)
#     print(done)
    if done:
        break
scores.append(cum_reward)
env.close()


# %%
MEMORYLEN=int(1e5)
# MEMORYLEN=int(10000)
MEMORYLEN=int(64)
BATCHSIZE=64
EPOCHS=1


class DQNAgent():
    def __init__(self,actions=4,obs=8):
        self.actions=actions
        self.observations=obs
        self.actor,self.critic,self.policy=self.load_model()
        
        
#         self.copy_weights()

        self.memory=deque(maxlen=MEMORYLEN)
        self.gamma=0.99
        self.patience=0
        
           
    def play(self,observation,epsilon):
        if (len(self.memory)<BATCHSIZE):
            
            action=np.random.randint(low=0,high=self.actions)
            return action
        else:
            action=self.choose_action(observation)
            return action
    
    def choose_action(self,observation):
        state=observation[np.newaxis,:]
        action_probs=self.policy.predict(state)[0]
        action=np.random.choice(self.actions,p=action_probs)
        return action
            
    def step(self,state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        if ((len(self.memory)>=BATCHSIZE) ):
            self.train_model()
        pass
 
    
    def train_model(self):
        
        rnd_indices = np.random.choice(len(self.memory), size=BATCHSIZE)
        data=np.array(self.memory)[rnd_indices]
        np.random.shuffle(data)
        
        state, action, reward, next_state, done=np.stack(data[:,0]),np.stack(data[:,1]),np.stack(data[:,2]),np.stack(data[:,3]),np.stack(data[:,4])
#         state=state[np.newaxis,:]
#         next_state=next_state[np.newaxis,:]
        
        critic_output_state=self.critic.predict(state).flatten()
        critic_output_next_state=self.critic.predict(next_state).flatten()
        
        target=reward+self.gamma*critic_output_next_state*(1-done)
        delta=target-critic_output_state
        
        
        actions=np.zeros([BATCHSIZE,self.actions])
        actions[np.arange(BATCHSIZE),action]=1
        
        self.actor.fit([state,delta],actions,verbose=0)
        self.critic.fit(state,target,verbose=0)
                
#         self.patience+=1
#         if self.patience==10:
#             self.copy_weights()
#             self.patience=0
        
        pass
    def model_predictions(self,observation):
        pred=self.model.predict(observation.reshape(1,-1))
        pred=np.argmax(pred)
        return pred
        
    def load_model(self):
        num_input = layers.Input(shape=(self.observations, ))
        delta=layers.Input(shape=[1])
        dense1=layers.Dense(64,activation='relu')(num_input)
        dense2=layers.Dense(32,activation='relu')(dense1)
        probs=layers.Dense(self.actions,activation='softmax')(dense2)
        values=layers.Dense(1,activation='linear')(dense2)
        def custom_loss(y_true,y_pred):
            out=K.clip(y_pred,1e-8,1-1e-8)
            log_lik=y_true*K.log(out)
            return K.sum(-log_lik*delta)
        
        actor=models.Model(inputs=[num_input,delta],outputs=[probs])
        actor.compile(loss=custom_loss,optimizer=tf.keras.optimizers.Adam(lr=0.001,decay=0.005))
        
        critic=models.Model(inputs=[num_input],outputs=[values])
        critic.compile(loss="mse",optimizer=tf.keras.optimizers.Adam(lr=0.001,decay=0.005))
        
        policy=models.Model(inputs=[num_input],outputs=[probs])
        
        return actor,critic,policy
 


# %%
import time
from tqdm import tqdm
starttime=time.time()
scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores
n_episodes=4000
agent=DQNAgent()

max_t=500
eps_start=1.0
eps_end=0.2
eps_decay=0.995

eps = eps_start
env=gym.make('LunarLander-v2')
eps_history=[]
for i_episode in range(1, n_episodes+1):
    state = env.reset()
    score = 0
    for i_ in range(1,max_t+1):
        action = agent.play(state,eps)
        next_state, reward, done, _ = env.step(action)
        if reward==-100:
            reward=-100
        if reward==100:
            reward=35
        agent.step(state, action, reward, next_state, done)
        
        state = next_state
        score += reward
        if done:
            break 
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps)
    eps_history.append(eps)
    if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            agent.policy.save("weightsfolder/policy_.h5")
    if np.mean(scores_window)>=190.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            
            break
endtime= time.time() 
print(endtime-starttime)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# %%


