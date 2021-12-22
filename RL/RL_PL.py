import os
import gym
import time
import math
from gym.spaces.dict import Dict
import numpy as np
from gym.core import Env
from PL_LunarLander import LunarLander
import PL_LunarLander
import matplotlib.pyplot as plt
from gym import spaces

# simulation parameters
dt = 0.005
mass = 0.52
starting_height = 1.71
deviation = 0.05
render = False

""" Thrust Curve Code (SRM)"""
def engineCurve(x,a,b,c,d,e,f,g,h,i,j,k,l,m,n):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 +  h*x**7 + i*x**8 +  j*x**9 + k*x**10 +  l*x**11 + m*x**12 +  n*x**13

def update_plot(datx, daty, velx, vely, act):
        plt.figure(figsize=(8,8))
        plt.plot(datx, daty, color='b')
        plt.xlabel("t")
        plt.ylabel("vx")
        plt.show()
        plt.xlabel("t")
        plt.ylabel("vy")
        plt.plot(velx, vely, color='r')
        plt.show()
        plt.xlabel("t")
        plt.ylabel("actuator")
        plt.plot(datx, act, color='r')
        plt.show()

param = [8.55737293*10**-5,  2.16185198*10**0,  1.03338808*10**3, -3.27246718*10**4,
    5.61765793*10**5, -4.58971034*10**6,  1.31742637*10**7,  4.09887150*10**7,
    -3.07523806*10**8,  1.29853726*10**8,  1.93955863*10**9, -3.51515455*10**8,
    -1.38401836*10**10,  1.86375875*10**10]

params = []
for i in param:
    params.append(i*8)

class SRM_PL_RL(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(SRM_PL_RL, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    
    #define engine stuff
    self.prev_shaping = None
    self.engine_on = False
    self.thrust = 0
    self.Tburn = 0
    self.powers = False
    self.done = False
    self.actuator = 0

    self.action_space = spaces.Box(low=-5*3.14/180,high=5*3.14/180,shape=(1,),dtype=np.float32)
    # observation space
    self.observation_space = spaces.Box(
        -np.inf, np.inf, shape=(6,), dtype=np.float32
    )

    self.hit_ground = False
    self.soft_landing = False

    # initiate dynamics
    self.acc = [0,0,0]
    self.vel = [0,0,-10]
    self.pos = [0,0,starting_height+np.random.uniform(-deviation, deviation)]

    self.pos_x = [self.pos[0]]
    self.pos_y = [self.pos[2]]

    self.velx = [self.vel[0]]
    self.vely = [self.vel[2]]

    self.tl = [0]
    self.act = [0]

    self.t = 0

    self.state = [self.pos[0],
        self.pos[2],
        self.vel[0],
        self.vel[2],
        0,
        0
    ]

  def step(self, action):
    #action = np.clip(action, -1, +1).astype(np.float32)
    # Execute one time step within the environment
    self.t += dt
    self.tl.append(self.t)
    if abs(action - self.actuator) < 5/17*3.15/180:
        self.actuator = action
    else:
        self.actuator = self.actuator + action/abs(action)*1/17*5*3.15/180

    self.act.append(self.actuator)

    #if (self.engine_on or self.powers )and self.Tburn < 0.32:
    if self.Tburn < 0.32:
        self.thrust = engineCurve(self.Tburn, *params)
        self.Tburn += dt
        self.powers = True
    else:
        self.thrust = 0

    #update dynamics
    self.acc = [ self.thrust * math.sin(self.actuator) / mass, 
                    0, 
                    self.thrust * math.cos(self.actuator) / mass - 9.81]
    m = 0
    for i in self.vel:
        self.vel[m] = i + self.acc[m] * dt
        m+=1
    
    m = 0
    for i in self.pos:
        self.pos[m] = i + self.vel[m] * dt
        m+=1

    self.pos_x.append(self.pos[0])
    self.pos_y.append(self.pos[2])

    self.velx.append(self.vel[0])
    self.vely.append(self.vel[2])

    if self.pos[2] <= 0 and np.linalg.norm(self.vel) > 1.5:
        self.hit_ground = True
    if self.pos[2] <= 0 and np.linalg.norm(self.vel) <= 1.5:
        self.soft_landing = True

    state = [self.pos[0],
        self.pos[2],
        self.vel[0],
        self.vel[2],
        0,
        0
    ]
    assert len(state) == 6

    reward = 0
    shaping = (
        #-20 * np.sqrt(state[0] * state[0] + state[1] * state[1])
        #- 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
        )#    - 100 * abs(state[4])
    #)
    if self.prev_shaping is not None:
        reward = shaping - self.prev_shaping
    self.prev_shaping = shaping
    
    self.done = False
    # criteria for rough landing, total speed more than 1 m/s, height = 0 m
    if self.hit_ground:
        self.done = True
        reward = -300 * abs(np.linalg.norm(self.vel)/1.5)
        if render:
            update_plot(self.tl, self.pos_y, self.tl, self.vely, self.act)
    if self.soft_landing:
        self.done = True
        reward = +100 * (abs(1/np.linalg.norm(self.vel)))
        if render:
            update_plot(self.tl, self.pos_y, self.tl, self.vely, self.act)

    return np.array(state, dtype=np.float32), reward, self.done, {}
  
  def reset(self):
    # Reset the state of the environment to an initial state
    self.t = 0
    self.prev_shaping = None
    self.tl = [0]
    self.act = [0]
    self.actuator = 0

    self.engine_on = False
    self.thrust = 0
    self.Tburn = 0
    self.powers = False

    self.hit_ground = False
    self.soft_landing = False

    # initiate dynamics
    self.acc = [0,0,0]
    self.vel = [0,0,-10]
    self.pos = [0,0,starting_height+np.random.uniform(-deviation, deviation)]

    self.pos_x = [self.pos[0]]
    self.pos_y = [self.pos[2]]

    self.velx = [self.vel[0]]
    self.vely = [self.vel[2]]

    self.t = 0

    self.state = [self.pos[0],
        self.pos[2],
        self.vel[0],
        self.vel[2],
        0,
        0
    ]
    return self.state
  
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    return