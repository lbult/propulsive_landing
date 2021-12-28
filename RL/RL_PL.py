import os
import gym
import time
import math
from gym.spaces.dict import Dict
import numpy as np
from gym.core import Env
import matplotlib.pyplot as plt
from gym import spaces
from support_code import SRM, Render

# simulation parameters
dt = 0.01 # [s]
vehicle_mass = 0.52 # [kg]
starting_height = 2.5 # [m]
deviation = 0.05 # [m]
initial_velocity = -13 #[m/s]
render = True
engine_type = "F-15-0"
COM_dist = 0.1 # [m]
Ixx = 5*10**-3 # kg/m^2
 

class SRM_PL_RL(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, render):
    super(SRM_PL_RL, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    
    self.engine = SRM(engine_type, 0, dt)
    self.total_mass = vehicle_mass + self.engine.total_mass

    #define engine stuff
    self.prev_shaping = None
    self.engine_on = False
    self.thrust = 0
    self.Tburn = 0
    self.powers = False
    self.done = False
    self.actuator = 0
    self.renderr = Render

    self.action_space = spaces.Box(low=-5*3.14/180,high=5*3.14/180,shape=(1,),dtype=np.float32)
    # observation space
    self.observation_space = spaces.Box(
        -np.inf, np.inf, shape=(6,), dtype=np.float32
    )

    self.hit_ground = False
    self.soft_landing = False

    # initiate translational dynamics
    self.acc = [0,0,0]
    self.vel = [0,0,initial_velocity]
    self.pos = [0,0,starting_height+np.random.uniform(-deviation, deviation)]

    self.pos_x = [self.pos[0]]
    self.pos_y = [self.pos[2]]

    self.velx = [self.vel[0]]
    self.vely = [self.vel[2]]

    # initiate rotational dynamics
    self.ang_acc = 0
    self.ang_vel = 0
    self.ang_pos = 0

    self.tl = [0]
    self.act = [0]

    self.ang_track = [0]
    self.ang_vel_track = [0]

    self.t = 0

    self.state = [self.pos[0],
        self.pos[2],
        self.vel[0],
        self.vel[2],
        self.ang_pos,
        self.ang_vel
    ]

  def step(self, action):
    #action = np.clip(action, -1, +1).astype(np.float32)
    
    # Execute one time step within the environment
    # Update engine time as well
    self.t += dt
    self.engine.t = self.t
    self.total_mass = self.engine.total_mass + vehicle_mass
    self.tl.append(self.t)

    # Update engine thrust, mass etc.
    if self.Tburn < self.engine.t_burn:
        self.thrust = self.engine.thrust_force
        self.Tburn += dt
        self.powers = True
    else:
        self.thrust = 0
        self.Tburn += dt

    if abs(action - self.actuator) < 5/17*3.15/180:
        self.actuator = action
    else:
        self.actuator = self.actuator + action/abs(action)*1/17*5*3.15/180

    self.act.append(self.actuator)

    # update rotational dynamics
    self.ang_acc = self.thrust * math.sin(self.actuator+self.ang_pos) * COM_dist / Ixx
    self.ang_vel = self.ang_vel + self.ang_acc * dt
    self.ang_pos = self.ang_pos + self.ang_vel * dt

    # update translational dynamics
    self.acc = [ self.thrust * math.sin(self.actuator+self.ang_pos) / self.total_mass, 
                    0, 
                    self.thrust * math.cos(self.actuator+self.ang_pos) / self.total_mass - 9.81]
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

    self.ang_track.append(self.ang_pos)
    self.ang_vel_track.append(self.ang_vel)

    if self.pos[2] <= 0 and self.Tburn > self.engine.t_burn and np.linalg.norm(self.vel) > 0.5:
        self.hit_ground = True
    if self.pos[2] <= 0 and self.Tburn > self.engine.t_burn and np.linalg.norm(self.vel) <= 0.5:
        self.soft_landing = True

    state = [self.pos[0],
        self.pos[2],
        self.vel[0],
        self.vel[2],
        self.ang_pos,
        self.ang_vel
    ]
    assert len(state) == 6

    reward = 0
    shaping = (-100*abs(state[0]) - 100*abs(state[4])
            -20*abs(state[5])
        #-20 * np.sqrt(state[0] * state[0] + state[1] * state[1])
        #- 20 * np.sqrt(state[2] * state[2] + state[3] * state[3])
        #)    - 100 * abs(state[4])
    )
    if self.prev_shaping is not None:
        reward = shaping - self.prev_shaping
    self.prev_shaping = shaping

    self.done = False
    # criteria for rough landing, total speed more than 1 m/s, height = 0 m
    if self.hit_ground:
        self.done = True
        reward = -100 * abs(np.linalg.norm(self.vel)/3)
        if self.renderr:
            Render(self.tl, [self.pos_x, self.pos_y, self.velx, self.vely, self.ang_track, self.ang_vel], self.act, reward)
    if self.soft_landing:
        self.done = True
        reward = +100 * (abs(3/np.linalg.norm(self.vel)))
        if self.renderr:
            Render(self.tl, [self.pos_x, self.pos_y, self.velx, self.vely, self.ang_track, self.ang_vel], self.act, reward)

    return np.array(state, dtype=np.float32), reward, self.done, {}
  
  def reset(self):
    # Reset the state of the environment to the initial state

    self.engine = SRM(engine_type, 0, dt)
    self.total_mass = vehicle_mass + self.engine.total_mass

    #define engine stuff
    self.prev_shaping = None
    self.engine_on = False
    self.thrust = 0
    self.Tburn = 0
    self.powers = False
    self.done = False
    self.actuator = 0

    self.hit_ground = False
    self.soft_landing = False

    # initiate dynamics
    self.acc = [0,0,0]
    self.vel = [0,0,initial_velocity]
    self.pos = [0,0,starting_height+np.random.uniform(-deviation, deviation)]

    # initiate rotational dynamics
    self.ang_acc = 0
    self.ang_vel = 0
    self.ang_pos = 0

    self.pos_x = [self.pos[0]]
    self.pos_y = [self.pos[2]]

    self.velx = [self.vel[0]]
    self.vely = [self.vel[2]]

     # initiate rotational dynamics
    self.ang_acc = 0
    self.ang_vel = 0
    self.ang_pos = 0

    self.ang_track = [0]
    self.ang_vel_track = [0]

    self.tl = [0]
    self.act = [0]

    self.t = 0

    self.state = [self.pos[0],
        self.pos[2],
        self.vel[0],
        self.vel[2],
        self.ang_pos,
        self.ang_vel
    ]
    return self.state
  
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    return