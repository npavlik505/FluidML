# DDPG for lorenz system
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

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)



device = torch.device("cuda:0")

class LorenzEnv(gym.Env):

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
        #Action Space
        self.action_space = spaces.Box(low = -50, high = 50, shape = (1,), dtype = np.float32) #No downstream numpy

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
    

    #Define actor and critic NN.
# Actor produces single action; The state is inputed, the action is out made continuous by multiplying max acion with tanh(NN output)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        #self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s): #Changed l1 and l2 from F.relu() to tanh
        s = torch.tanh(self.l1(s))
        #s = torch.tanh(self.l2(s))
        a = self.max_action * torch.tanh(self.l3(s))  # [-max,max]
        #print('self.max_action:', self.max_action)
        #print(torch.tanh(self.l3(s)))
        return a

# Critic produces single value (Q value); The state AND action is inputed, the output represents the value of taking the action-state pair
class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

# Replay buffer stores 1000 state, action, reward, next state, change in weights (S,A,R,S_,dw) data sets
class ReplayBuffer(object):
    #Specifies max number of SARSdW collected (max_size) and creates matrices to store the collected data
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = torch.zeros((self.max_size, state_dim))
        self.a = torch.zeros((self.max_size, action_dim))
        self.r = torch.zeros((self.max_size, 1))
        self.s_ = torch.zeros((self.max_size, state_dim))
    #The method to store the data
    def store(self, s, a, r, s_):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    #Out of the stored data, which is on length self.size, a batch_size number of SARS_dw samples are randomly collected
    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_

#This the policy gradient algorithm, notice this uses the actor and critic classes made earlier
#The hyperparameters are defined, the actor & critc NN are defined as attributes and their Target NN are created
#Lastly, the optimizer, Adam, is selected to adjust the NN weights and the MSELoss is selected for use in the backprop calc
class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.hidden_width = 8  # The number of neurons in hidden layers of the neural network
        self.batch_size = 50 #100  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.MseLoss = nn.MSELoss()
    # An action is chosen by feeding the state into the actor NN which outputs the action a... refreshingly simple :)
    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).data.numpy().flatten()
        #a = torch.unsqueeze(torch.tensor(a, dtype=torch.float), 0)
        return a

    # We use our sample method, previously defined, to select the SARS_dw samples
    def learn(self, replay_buffer):
        # print(replay_buffer.is_cuda)
        # print(type(replay_buffer))
        # replay_buffer.to(device)
        # print(type(replay_buffer))
        batch_s, batch_a, batch_r, batch_s_= replay_buffer.sample(self.batch_size) # Sample a batch

        # Compute the target Q. This is done with no_grad so the target Q NN weights won't be adjusted every learning
        # Not exactly sure why apparently only two args required for critic_target but three required before it was deepcopied; Maybe
            # super() or the fact that the second argument is another method has something to do with it.
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * Q_

        # Compute the current Q and then the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()                     #THE CRITIC IS BEING OPTIMIZED: IS THE CRITIC TARGET BEING OPTIMIZED?

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)







device = torch.device("cuda:0")

#GLOBAL VARIABLES TO BE USED FOR TESTING (BEFORE PACKAGE INTEGRATION)
#DDPG Control Attributes
Episodes = 5
random_steps = 500
max_episode_steps = 500
update_freq = 5
Learnings = 2

#LorenzEnv Attributes
sigma = 10
rho = 28
beta = 8/3
dt = .001
#WHEN READY TO INTEGRATE INTO PACKAGE, ADD LorenzEnv PARAMETERS TO DDPGcontrol PARAMETERS


def DDPGcontrol(Episodes, random_steps, max_episode_steps, update_freq, Learnings):
    env = LorenzEnv(sigma, rho, beta, dt)

    testt1 = 0
    testt2 = 0

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration


#Note: Eventually Program the NN to be reinitialized at the beginning of each learning
    for total_learnings in range(Learnings):
        print("This is learning ", str(total_learnings + 1))
        testt1 = 0
        testt2 = 0
        LearningAveMSE = []
        state_data = []
        for Episode in range(Episodes):
            s = env.reset()
            state_data.clear()
            state_data.append(s)
            #Create test lorenz
            s_test = copy.deepcopy(s)
            #Figure for Plotting
            ######fig = plt.figure()
            # if Episode % 10 == 0:
                # fig = plt.figure()
                # ax = fig.add_subplot(1, 1, 1, projection='3d')
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel('Z')
            for episode_steps in range(max_episode_steps):
                #Test lorenz - get delta
                #if episode_steps == 250: print('Sample ICs:', s_test)
                #s_test += torch.tensor(env.onlylorenz(s_test))#, device = 'cuda')
                #Plotting
                #2D Plot
                #plt.plot(episode_steps,s[0], markersize=1, marker = '+', color = 'b')
                #2D Plot Test
                #plt.plot(episode_steps,s_test[0], markersize=1, marker = '+', color = 'c')
                #2D Plot X and Y
                #FOR PLOTTING
                if random_steps >= 0:
                    random_steps += -25
                # if Episode % 10 == 0:
                #     ss = s.to('cpu')
                #     ss = ss.numpy()
                #     #3D
                #     ax.scatter(ss[0],ss[1], ss[2], c=ss[1], s=1, marker = '+')

                if episode_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                    #a = torch.tensor(env.action_space.sample(), device = 'cuda')
                    a = env.action_space.sample()
                    a = torch.from_numpy(a)
                    #if episode_steps % 50 == 0: print('random', a)
                    #if episode_steps > 4999: print(Episode, 'end of random a selection')
                else:
                    # Add Gaussian noise to actions for exploration
                    #a = torch.tensor(agent.choose_action(s), device = 'cuda')
                    a = agent.choose_action(s)
                    a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
                    a = torch.from_numpy(a)
                    #if episode_steps % 50 == 0: print('chosen', a)
                #Action History Plot
                ########plt.plot(episode_steps,a, markersize=1, marker = '+', color = 'b')
                s_, r, terminated, truncated = env.step(a, s)
                state_data.append(s_)
                if episode_steps == 0:
                    print('Start of Episode ' + str(Episode+1))
                    EpAveMSE = (((abs(s[0] - env.Ftarget[0])+abs(s[1] - env.Ftarget[1])+abs(s[2] - env.Ftarget[2]))**2))
                else:
                    EpAveMSE = ((EpAveMSE*(episode_steps-1)) + (abs(s[0] - env.Ftarget[0])**2+abs(s[1] - env.Ftarget[1])**2+abs(s[2] - env.Ftarget[2])**2))/episode_steps
                if episode_steps == max_episode_steps-1:
                    #print('End of Episode ' + str(Episode+1))
                    LearningAveMSE.append(EpAveMSE)
                    if EpAveMSE == min(LearningAveMSE):
                        #BestStateData.clear()
                        Best_State_Data = state_data

                        #Deleteing previous file, method 1
                        Control = os.path.dirname(os.path.abspath(__file__))
                        FluidML = os.path.abspath(os.path.join(Control, '..'))
                        myfilepath1 = os.path.join(FluidML, 'BestParamsActor_Learning' + str(total_learnings+1) + '.pt')
                        myfilepath2 = os.path.join(FluidML, 'BestParamsCritic_Learning' + str(total_learnings+1) + '.pt')

                        if os.path.exists(myfilepath1):
                            os.remove(myfilepath1)
                            testt1 += 1
                            print("we've deleted " + str(testt1) + " actor files")
                        if os.path.exists(myfilepath2):
                            os.remove(myfilepath2)
                            testt2 += 1
                            print("we've deleted " + str(testt2) + " critic files")
                        
                        # #Deleteing prvoious file, method 2 (more pythonic, maybe not applicable)
                        # myfilepath = "File path goes here"
                        # try:
                        #     os.remove(myfilepath)
                        #     testt += 1
                        #     print("we've deleted " + str(testt) + "files")    
                        # except:

                        torch.save(agent.actor.state_dict(), 'BestParamsActor_Learning' + str(total_learnings+1) + '.pt')
                        torch.save(agent.critic.state_dict(), 'BestParamsCritic_Learning' + str(total_learnings+1) + '.pt')

                    #print('Episode Number:', Episode+1)
                    print('Episode Ave MSE:', EpAveMSE)
                    print()

                #if terminated:
                    #break
                replay_buffer.store(s, a, r, s_)  # Store the transition
                s = s_

                # Take 50 steps,then update the networks 50 times
                if episode_steps >= random_steps and episode_steps % update_freq == 0:
                    for _ in range(update_freq):
                        agent.learn(replay_buffer)
        #print("Best State Data", Best_State_Data)
                        
        print("End of Learning " + str(total_learnings + 1))
        print("Best State Data length", len(Best_State_Data))
        xx = torch.linspace(1,Episodes, int(Episodes))
        yy = LearningAveMSE

        plt.figure()
        plt.plot(xx, yy)
    plt.show()


# Testing the Definition
DDPGcontrol(Episodes, random_steps, max_episode_steps, update_freq, Learnings)






























# device = torch.device("cuda:0")

# if __name__ == '__main__':
#     env = LorenzEnv(sigma = 10, rho = 28, beta = 8/3, dt = .001)
#     env_evaluate = env  # When evaluating the policy, we need to rebuild an environment
#     number = 1

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     max_action = float(env.action_space.high[0])
#     print(max_action)

#     agent = DDPG(state_dim, action_dim, max_action)
#     replay_buffer = ReplayBuffer(state_dim, action_dim)


#     max_train_steps= 30
#     random_steps = 500

#     max_episode_steps = 5000
#     update_freq = 5
#     evaluate_num = 0
#     evaluate_rewards = []
#     evaluate_freq = 10  # Evaluate the policy every 'evaluate_freq' steps

#     noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration

#     AverageMSE = []
#     action_history = []
#     for total_steps in range(max_train_steps):
#         #s = torch.tensor(env.reset(), device = 'cuda')
#         #print(total_steps)
#         s = env.reset()
#         #Create test lorenz
#         s_test = copy.deepcopy(s)
#         episode_steps = 0
#         #Figure for Plotting
#         ######fig = plt.figure()
#         # if total_steps % 10 == 0:
#             # fig = plt.figure()
#             # ax = fig.add_subplot(1, 1, 1, projection='3d')
#             # ax.set_xlabel('X')
#             # ax.set_ylabel('Y')
#             # ax.set_zlabel('Z')
#         for episode_steps in range(max_episode_steps):
#             #Test lorenz - get delta
#             #if episode_steps == 250: print('Sample ICs:', s_test)
#             #s_test += torch.tensor(env.onlylorenz(s_test))#, device = 'cuda')
#             #Plotting
#             #2D Plot
#             #plt.plot(episode_steps,s[0], markersize=1, marker = '+', color = 'b')
#             #2D Plot Test
#             #plt.plot(episode_steps,s_test[0], markersize=1, marker = '+', color = 'c')
#             #2D Plot X and Y
#             #FOR PLOTTING
#             episode_steps += 1
#             if random_steps != 0:
#                 random_steps += -25
#             # if total_steps % 10 == 0:
#             #     ss = s.to('cpu')
#             #     ss = ss.numpy()
#             #     #3D
#             #     ax.scatter(ss[0],ss[1], ss[2], c=ss[1], s=1, marker = '+')

#             if episode_steps < random_steps:  # Take the random actions in the beginning for the better exploration
#                 #a = torch.tensor(env.action_space.sample(), device = 'cuda')
#                 a = env.action_space.sample()
#                 a = torch.from_numpy(a)
#                 #if episode_steps % 50 == 0: print('random', a)
#                 #if episode_steps > 4999: print(total_steps, 'end of random a selection')
#             else:
#                 # Add Gaussian noise to actions for exploration
#                 #a = torch.tensor(agent.choose_action(s), device = 'cuda')
#                 a = agent.choose_action(s)
#                 a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
#                 a = torch.from_numpy(a)
#                 #if episode_steps % 50 == 0: print('chosen', a)
#             #Action History Plot
#             ########plt.plot(episode_steps,a, markersize=1, marker = '+', color = 'b')
#             s_, r, terminated, truncated = env.step(a, s)
#             if episode_steps == 1:
#                  AveMSE = (((abs(s[0] - env.Ftarget[0])+abs(s[1] - env.Ftarget[1])+abs(s[2] - env.Ftarget[2]))**2))
#             else:
#                 AveMSE = ((AveMSE*(episode_steps-1)) + (abs(s[0] - env.Ftarget[0])**2+abs(s[1] - env.Ftarget[1])**2+abs(s[2] - env.Ftarget[2])**2))/episode_steps
#             if episode_steps == max_episode_steps:
#                 AverageMSE.append(AveMSE)
#                 print('Episode Number:', total_steps+1)
#                 print('Episode Ave MSE:', AveMSE)
#             #if terminated:
#                 #break
#             replay_buffer.store(s, a, r, s_)  # Store the transition
#             s = s_

#             # Take 50 steps,then update the networks 50 times
#             if episode_steps >= random_steps and episode_steps % update_freq == 0:
#                 for _ in range(update_freq):
#                     agent.learn(replay_buffer)
#     xx = torch.linspace(0,max_train_steps, int(max_train_steps))
#     yy = AverageMSE

#     plt.figure()
#     plt.plot(xx, yy)
#     plt.show()



