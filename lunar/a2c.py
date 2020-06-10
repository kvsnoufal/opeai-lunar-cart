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


def query_environment(name):
  env = gym.make(name)
  spec = gym.spec(name)
  print(f"Action Space: {env.action_space}")
  print(f"Observation Space: {env.observation_space}")
  print(f"Max Episode Steps: {spec.max_episode_steps}")
  print(f"Nondeterministic: {spec.nondeterministic}")
  print(f"Reward Range: {env.reward_range}")
  print(f"Reward Threshold: {spec.reward_threshold}")

MEMORYLEN=int(1)
BATCHSIZE=1
EPOCHS=1
# UPDATE_EVERY = 4


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
        
        # rnd_indices = np.random.choice(len(self.memory), size=BATCHSIZE)
        # data=np.array(self.memory)[rnd_indices]
        # np.random.shuffle(data)
        data=np.array(self.memory)
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
    
        
    def load_model(self):
        num_input = layers.Input(shape=(self.observations, ))
        delta=layers.Input(shape=[1])
        dense1=layers.Dense(128,activation='relu')(num_input)
        dense2=layers.Dense(64,activation='relu')(dense1)
        probs=layers.Dense(self.actions,activation='softmax')(dense2)
        values=layers.Dense(1,activation='linear')(dense2)
        def custom_loss(y_true,y_pred):
            out=K.clip(y_pred,1e-8,1-1e-8)
            log_lik=y_true*K.log(out)
            return K.sum(-log_lik*delta)
        
        actor=models.Model(inputs=[num_input,delta],outputs=[probs])
        actor.compile(loss=custom_loss,optimizer=tf.keras.optimizers.Adam(lr=0.00001,decay=0.00005))
        
        critic=models.Model(inputs=[num_input],outputs=[values])
        critic.compile(loss="mse",optimizer=tf.keras.optimizers.Adam(lr=0.00001,decay=0.00005))
        
        policy=models.Model(inputs=[num_input],outputs=[probs])
        
        return actor,critic,policy



def gen_video(ag,epoch):
    eps=0
    env = gym.make("LunarLander-v2")
    env = gym.wrappers.Monitor(env,"a2cvid/{}".format(epoch))
    # action_space = env.action_space.n
    # observation_space = env.observation_space.shape[0]
    
    # scores = []

    agent=ag
    
    obs = env.reset()

    frames = []
    cum_reward=0
    while True:
        action = agent.play(obs,eps)

        obs, reward, done, _ = env.step(action)

        cum_reward += reward
        
        if done:
            print(reward,done)
            break
    env.close()    
    print("TOTAL REWARDS: {}".format(cum_reward))




import time
from tqdm import tqdm
starttime=time.time()
scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores
n_episodes=4000

agent=DQNAgent()

max_t=500
eps_start=1.0
eps_end=0.15
eps_decay=0.995



eps = eps_start
env=gym.make('LunarLander-v2')
eps_history=[]
best_score_so_far=-9999
for i_episode in range(1, n_episodes+1):
    state = env.reset()
    score = 0
    for i_ in range(1,max_t+1):
        action = agent.play(state,eps)
        next_state, reward, done, _ = env.step(action)
        # if reward==-100:
        #     reward=-35
        # if reward==100:
        #     reward=35
        agent.step(state, action, reward, next_state, done)
        
        state = next_state
        score += reward
        if done:
            # print(i_,reward,score)
                           
            break 
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps)
    eps_history.append(eps)
    if i_episode % 100 == 0:
        agent.policy.save_weights("./weightsfolder/Lunar_a2cweights_{}.h5".format(i_episode))
    if score>best_score_so_far:
        best_score_so_far=score
        agent.policy.save_weights("./weightsfolder/Lunar_a2cbest_weights_.h5")
        print("saving best weights {}".format(score))
        gen_video(agent,i_episode)
    if i_episode % 10 == 0:
            print('\nEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
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
plt.savefig("luna2cFT.png")
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(eps_history)), eps_history)
plt.ylabel('Epsilone')
plt.xlabel('Episode #')
plt.savefig("epsilona2cLunarFT.png")
# plt.show()  
agent.policy.save_weights("./weightsfolder/Lunar_a2c_weights.h5")