
# %%
from IPython import get_ipython

# %%
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
import os
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
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()


# %%
# env = gym.make('LunarLander-v2')
# obs = env.reset()
# print(obs.shape)
# s=env.render(mode = 'rgb_array')
# s.shape
# env.close()


# %%
import matplotlib.animation as animation

env = gym.make('LunarLander-v2')
env = gym.wrappers.Monitor(env,"./notefiles/12")
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

print(scores)
# %%


