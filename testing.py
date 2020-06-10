import gym
from gym.wrappers import Monitor
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
import glob
import io
import base64
from gym import wrappers
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers,models

def query_environment(name):
  env = gym.make(name)
  spec = gym.spec(name)
  print(f"Action Space: {env.action_space}")
  print(f"Observation Space: {env.observation_space}")
  print(f"Max Episode Steps: {spec.max_episode_steps}")
  print(f"Nondeterministic: {spec.nondeterministic}")
  print(f"Reward Range: {env.reward_range}")
  print(f"Reward Threshold: {spec.reward_threshold}")

env=gym.make('CarRacing-v0')
# print(env.reset())
# print("done")

class randomAgent():
    def __init__(self,actions=3,obs=8,random=True):
        self.actions=actions
        self.random=random
        self.observations=8
        
    def play(self,observation):
        if self.random:
            action=np.random.randint(low=0,high=self.actions)
            # return env.action_space.sample()
            return [0,1,0]
        else:
            pass
    def step(self,state, action, reward, next_state, done):
        pass
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
starttime=time.time()
scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores
n_episodes=1
agent=randomAgent()
env=gym.make('CarRacing-v0')
for i_episode in range(1, n_episodes+1):
    state = env.reset()
    score = 0
    fig = plt.figure()
    ims=[]
    while True:
        im =plt.imshow(state, animated=True)
        ims.append([im])
        action = agent.play(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break 
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            
            break
endtime= time.time() 
print(endtime-starttime)
ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                repeat_delay=11110)

ani.save('dynamic_images_010_low.gif',writer=animation.PillowWriter())
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# # plt.show()    
# plt.savefig("test_save.png")