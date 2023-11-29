import numpy as np
import gym
from gym import spaces
import copy
import math
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from scipy.integrate import solve_ivp
device = torch.device("cuda:0")

class LorenzEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    #Define the action space and observation space in the init function
    def __init__(self, sigma, rho, beta, dt): #, render_mode=None, size=5):
        #Parameters
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        #Equilibrium Values
        #self.Ftarget = np.array([math.sqrt(self.beta*(self.rho - 1)), math.sqrt(self.beta*(self.rho - 1)), (self.rho - 1)])
        #self.Ftarget = [abs(math.sqrt(72)), abs(math.sqrt(72)), 27]
        self.Ftarget = [-8.42, -8.42, 27]
        #Observation Space
        self.observation_space = spaces.Box(low = np.array([-20, -15, 0]), high = np.array([20, 15, 45]), shape = (3,), dtype = np.float32) #No downstream numpy
        #NUMPY can't convert yet
        #Action Space
        self.action_space = spaces.Box(low = -50, high = 50, shape = (1,), dtype = np.float32) #No downstream numpy
        #NUMPY can't convert yet

    #Lorenz System Eqns - dF = np.array([dX, dY, dZ])
    def lorenz(self, s, a):
        #print('before lorenz step', s,a)
        s = s.to(torch.float32)
        dF = [(self.sigma*(s[1] - s[0]))+ a,
            (s[0]*(self.rho - s[2]) - s[1]),
            (s[0]*s[1] - self.beta*s[2])]
        #list comprehension to multiply a float by a list
        dF = [x*self.dt for x in dF]
        dF = torch.tensor(dF)
        #print('deta lorenz values', dF)
        return dF

    def onlylorenz(self, s):
        s = s.to(torch.float32)
        dFL = [(self.sigma*(s[1] - s[0])),
            (s[0]*(self.rho - s[2]) - s[1]),
            (s[0]*s[1] - self.beta*s[2])]
        #list comprehension to multiply a float by a list
        dFL = [x*self.dt for x in dFL]
        dFL = torch.tensor(dFL)
        return dFL

#RESET
    # The initial conditions will need to be randomly generated
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #self.F0 = np.array([np.random.uniform(-21, 21), np.random.uniform(-16, 16), np.random.uniform(-1, 46)])
        self.F0 = [-0.8949, -0.4012, 23.1023]
        self.F0 = torch.tensor(self.F0)
        return self.F0

#STEP
    def step(self, a, s):
        a = torch.tensor(a)
        s = torch.tensor(s)
        FB4Step = copy.deepcopy(s)
        s += torch.tensor(self.lorenz(s, a))#, device = 'cuda')
        #print('state after step', s,a)

        # Reward Paradigm 1                                               #This solved the CPU / CUDA data problems
        if (math.sqrt(((s[0]-self.Ftarget[0])**2)) < math.sqrt(((FB4Step[0]-self.Ftarget[0])**2))):
            reward = 1
        elif (math.sqrt(((s[0]-self.Ftarget[0])**2)) > math.sqrt(((FB4Step[0]-self.Ftarget[0])**2))):
            reward = -1
        else: reward = 10
        
        # # Reward Paradigm 2
        # if (math.sqrt(((s[0]-self.Ftarget[0])**2)) + math.sqrt(((s[1]-self.Ftarget[1])**2)) + math.sqrt(((s[2]-self.Ftarget[2])**2))) >= .25:
        #     reward = 10/(math.sqrt(((s[0]-self.Ftarget[0])**2)) + math.sqrt(((s[1]-self.Ftarget[1])**2)) + math.sqrt(((s[2]-self.Ftarget[2])**2)))
        # elif (math.sqrt(((s[0]-self.Ftarget[0])**2)) + math.sqrt(((s[1]-self.Ftarget[1])**2)) + math.sqrt(((s[2]-self.Ftarget[2])**2))) < .25:
        #     reward = 100

        # # Reward Paradigm 3
        # if (math.sqrt(((s[0]-self.Ftarget[0])**2))) >= .25:
        #     reward = 5/(math.sqrt(((s[0]-self.Ftarget[0])**2)))
        # elif (math.sqrt(((s[0]-self.Ftarget[0])**2))) < .25:
        #     reward = 100

        

        terminated = True if (s[0] <= -20 or s[0] >= 20 or s[1] <= -15 or s[1] >= 15 or s[2] <= 0 or s[2] >= 45) else False # or state == self.Ftarget) else False
        #print('Terminated')
        truncated = False
        observation = s
        return observation, reward, terminated, truncated