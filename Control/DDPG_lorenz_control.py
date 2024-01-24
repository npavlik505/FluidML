### DDPG for lorenz system

#Imports for DDPG
import numpy as np
import gym
from gym import spaces
import copy
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
#Imports for RainClouds plots
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid",font_scale=2)
import ptitprince as pt

#Add to python path
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)


#Lorenz63 Gym Environment
device = torch.device("cuda:0")
class LorenzEnv(gym.Env):

    #Define the action space and observation space in the init function
    def __init__(self, sigma, rho, beta, dt, Force_X, Force_Y, Force_Z): #, render_mode=None, size=5):
        #Parameters
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.Force_X = Force_X
        self.Force_Y = Force_Y
        self.Force_Z = Force_Z
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
        if self.Force_X == True:
            a_x = a
        else:
            a_x = 0
        if self.Force_Y == True:
            a_y = a 
        else:
            a_y = 0
        if self.Force_Z == True:
            a_z = a
        else: a_z = 0  
        dF = [(self.sigma*(s[1] - s[0])) + a_x,
            (s[0]*(self.rho - s[2]) - s[1]) + a_y,
            (s[0]*s[1] - self.beta*s[2]) + a_z]
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
        #a = torch.tensor(a)
        a = torch.clone(a) #Trying to get rid of python warnimg for using torch.tensor()
        #s = torch.tensor(s)
        s = torch.clone(s) #Trying to get rid of python warnimg for using torch.tensor()

        FB4Step = copy.deepcopy(s)
        #s += torch.tensor(self.lorenz(s, a))#, device = 'cuda')
        s += torch.clone(self.lorenz(s, a)) #Trying to get rid of python warnimg for using torch.tensor()
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
    
    def initialize_Actor_weights(self):
        for mod1 in self.modules():
            if isinstance(mod1, nn.Linear):
                nn.init.normal(mod1.weight)
                if mod1.bias is not None:
                    nn.init.constant_(mod1.bias, 0)

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
    
    def initialize_Critic_weights(self):
        for mod2 in self.modules():
            if isinstance(mod2, nn.Linear):
                nn.init.normal(mod2.weight)
                if mod2.bias is not None:
                    nn.init.constant_(mod2.bias, 0)

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
        batch_s = torch.clone(self.s[index]) #Trying to get rid of python warnimg for using torch.tensor()
        batch_a = torch.clone(self.a[index]) #Trying to get rid of python warnimg for using torch.tensor()
        batch_r = torch.clone(self.r[index]) #Trying to get rid of python warnimg for using torch.tensor()
        batch_s_ = torch.clone(self.s_[index]) #Trying to get rid of python warnimg for using torch.tensor()

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
        #s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        s = torch.unsqueeze(torch.clone(s), 0) #Trying to get rid of python warnimg for using torch.tensor()
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

# #GLOBAL VARIABLES TO BE USED FOR TESTING (BEFORE PACKAGE INTEGRATION)
# #DDPG Control Attributes
# Episodes = 5
# random_steps = 500 #Ususally at 500
# max_episode_steps = 5000 #Usually at 5000
# update_freq = 5
# Learnings = 2

# #LorenzEnv Attributes
# sigma = 10
# rho = 28
# beta = 8/3
# dt = .001
# #WHEN READY TO INTEGRATE INTO PACKAGE, ADD LorenzEnv PARAMETERS TO DDPGcontrol PARAMETERS


def DDPGcontrol(sigma, rho, beta, dt, Episodes, random_steps, max_episode_steps, update_freq, Learnings, Force_X, Force_Y, Force_Z):
    env = LorenzEnv(sigma, rho, beta, dt, Force_X, Force_Y, Force_Z)

    testt1 = 0
    testt2 = 0

    state_dim = env.observation_space.shape[0] #Dimension of the state space (in this case three continous values for x, y, and z of the Lorenz System)
    action_dim = env.action_space.shape[0] #Dimension of the action space (in this case continuous values between 0 and 50)
    max_action = float(env.action_space.high[0]) #The max forcing that can be applied

    agent = DDPG(state_dim, action_dim, max_action) #Initializes the DDPG algorithm (Class containing the Actor-Critic NN that chooses x forcing action)
    replay_buffer = ReplayBuffer(state_dim, action_dim) #Initializes the ReplayBuffer (stores s,a,s_,r sequences for batch learning)

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration


#Note: Eventually Program the NN to be reinitialized at the beginning of each learning
    for total_learnings in range(Learnings):
        print("This is learning ", str(total_learnings + 1))
        #testt1 and testt2 used when validating file management for actor and critic parameters (may delete eventually)
        testt1 = 0
        testt2 = 0

        #Initialize lists to store the average MSE for forcing and non-forcing cases
        LearningAveMSE = []
        LearningAveMSE_NF = []

        #Code below resets network parameters between Learnings, making them independent of each other
        if total_learnings > 0:
            agent.actor.initialize_Actor_weights
            agent.actor_target = copy.deepcopy(agent.actor)
            agent.critic.initialize_Critic_weights
            agent.critic_target = copy.deepcopy(agent.critic)
            print('we have ran the reinit code')

        for Episode in range(Episodes):
            #Generate initial value for episode (note: forced ICs = Unforced ICs)
            s = env.reset()
            state_data = torch.tensor(s)
 
            s_noforcing = torch.clone(s)
            state_data_noforcing = s_noforcing.view(1,3)

            #Decriments the random_steps value so fewer are taken each episode (Consider if random steps are actually needed)
            if random_steps >= 0:
                random_steps += -25

            for episode_steps in range(max_episode_steps):

                #Generate the unforced lorenz values
                s_noforcing += env.onlylorenz(s_noforcing)
                state_data_noforcing = torch.cat((state_data_noforcing, s_noforcing.view(1,3)), dim = 0)

                #Randomly select, or have the NN choose, an action for the current step (i.e. forcing value on x)
                if episode_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                    a = env.action_space.sample()
                    a = torch.from_numpy(a)
                else:
                    # Add Gaussian noise to actions for exploration
                    a = agent.choose_action(s)
                    a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
                    a = torch.from_numpy(a)
                

                s_, r, terminated, truncated = env.step(a, s)
                state_data = torch.cat([state_data.view(-1,3), s_.view(1,3)])

                #Calculate the average MSE for each X forced episode 
                if episode_steps == 0:
                    print('Start of Episode ' + str(Episode+1))
                    EpAveMSE = (((abs(s[0] - env.Ftarget[0])+abs(s[1] - env.Ftarget[1])+abs(s[2] - env.Ftarget[2]))**2))
                else:
                    EpAveMSE = ((EpAveMSE*(episode_steps-1)) + (abs(s[0] - env.Ftarget[0])**2+abs(s[1] - env.Ftarget[1])**2+abs(s[2] - env.Ftarget[2])**2))/episode_steps


                #Calculate the average MSE for each unforced episode 
                if episode_steps == 0:
                    EpAveMSE_NF = (((abs(s_noforcing[0] - env.Ftarget[0])+abs(s[1] - env.Ftarget[1])+abs(s[2] - env.Ftarget[2]))**2))
                else:
                    EpAveMSE_NF = ((EpAveMSE_NF*(episode_steps-1)) + (abs(s_noforcing[0] - env.Ftarget[0])**2+abs(s_noforcing[1] - env.Ftarget[1])**2+abs(s_noforcing[2] - env.Ftarget[2])**2))/episode_steps                

                #Executes after the final step in each episode    
                if episode_steps == max_episode_steps-1:
                    LearningAveMSE.append(EpAveMSE)
                    LearningAveMSE_NF.append(EpAveMSE_NF)
                    if EpAveMSE == min(LearningAveMSE):

                        #Store the State Data for the Forcing and Non-Forcing Cases as np arrays for plotting
                        BSD = np.array(state_data)
                        UFSD = np.array(state_data_noforcing)

                        #Create path to parameter file
                        Control = os.path.dirname(os.path.abspath(__file__))
                        FluidML = os.path.abspath(os.path.join(Control, '..'))
                        myfilepath1 = os.path.join(FluidML, 'BestParamsActor_Learning' + str(total_learnings+1) + '.pt')
                        myfilepath2 = os.path.join(FluidML, 'BestParamsCritic_Learning' + str(total_learnings+1) + '.pt')
                        #Delete parameter file for previous best performing policy
                        if os.path.exists(myfilepath1):
                            os.remove(myfilepath1)
                            testt1 += 1
                        if os.path.exists(myfilepath2):
                            os.remove(myfilepath2)
                            testt2 += 1
                        #Create parameter file for new best performing policy
                        torch.save(agent.actor.state_dict(), 'BestParamsActor_Learning' + str(total_learnings+1) + '.pt')
                        torch.save(agent.critic.state_dict(), 'BestParamsCritic_Learning' + str(total_learnings+1) + '.pt')

                    #Print the average Mean Squared Error for the Episode
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

        #Store the state data for the best performing policy of each Learning 
        Best_State_Data_Storage = {}
        Best_State_Data_Storage['Learning_' + str(total_learnings + 1)] = BSD
        Unforced_State_Data_Storage = {}
        Unforced_State_Data_Storage['Learning_' + str(total_learnings + 1)] = UFSD
        
        #Plot the forced state data
        plot_title = []
        if Force_X == True:
            plot_title.append('X')
        if Force_Y == True:
            plot_title.append('Y')
        if Force_Z == True:
            plot_title.append('Z')
        fig1 = plt.figure()
        forcing = fig1.add_subplot(111, projection = '3d')
        forcing.plot(BSD[:,0], BSD[:,1], BSD[:,2], label = ('Best' + ' '.join(plot_title) + 'Forcing Policy'))
        forcing.set_title('Best ' + ' '.join(plot_title) + ' Forcing Policy - Learning ' + str(total_learnings + 1))
        forcing.set_xlabel('X', labelpad = 10)
        forcing.set_ylabel('Y', labelpad = 10)
        forcing.set_zlabel('Z', labelpad = 10)

        #Plot the unforced state data
        fig2 = plt.figure()
        noforcing = fig2.add_subplot(111, projection = '3d')
        noforcing.plot(UFSD[:,0], UFSD[:,1], UFSD[:,2], label = 'Corresponding Unforced Lorenz')              
        noforcing.set_title('Corresponding Unforced Lorenz - Learning ' + str(total_learnings + 1))
        noforcing.set_xlabel('X', labelpad = 10)
        noforcing.set_ylabel('Y', labelpad = 10)
        noforcing.set_zlabel('Z', labelpad = 10)

        #Plot the MSE for Forced and Unforced lorenz
        xx = torch.linspace(1,Episodes, int(Episodes))
        plt.figure()
        plt.plot(xx, LearningAveMSE, label = ' '.join(plot_title) + ' Forced MSE', color = 'blue')
        plt.plot(xx, LearningAveMSE_NF, label = 'Unforced MSE', color = 'red')
        plt.xlabel('Episode')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('MSE for Learning ' + str(total_learnings + 1))
        plt.legend()

        #Plot the spatial fluctuations (half-eye (density + interval) plots)
        #Forcing
        Xequil = env.Ftarget[0]
        Yequil = env.Ftarget[1]
        Zequil = env.Ftarget[2]    
        X_flux_forcing = []
        Y_flux_forcing = []
        Z_flux_forcing = []
        interval_of_analysis = len(BSD[:,0])*2//3
        print('The last ' + str(interval_of_analysis) + ' steps are included in RainCloud plots')
        for x in range(interval_of_analysis):
            x = x + (len(BSD[:,0]) - interval_of_analysis)
            X_flux_forcing.append(BSD[x,0]-Xequil)
            Y_flux_forcing.append(BSD[x,1]-Yequil)
            Z_flux_forcing.append(BSD[x,2]-Zequil)
        index = list(range(1, len(X_flux_forcing)+1))
        AllData1 = np.array([index, X_flux_forcing, Y_flux_forcing, Z_flux_forcing])
        AllData1 = np.transpose(AllData1)
        AllData1 = pd.DataFrame(AllData1, columns = ['Index', 'X', 'Y', 'Z'])
        DDF1 = pd.melt(AllData1, id_vars = ['Index'], value_vars = ['X', 'Y', 'Z'], var_name = 'state values', value_name = 'Dist From Equil')

        f, ax = plt.subplots(figsize=(12, 11))

        dy = 'Dist From Equil'; dx = 'state values'; ort = "v"
        # Draw a violinplot with a narrower bandwidth than the default
        ax=pt.half_violinplot(data = DDF1, palette = "Set2", bw=.2,  linewidth=1,cut=0.,\
                        scale="area", width=.8, inner=None,orient=ort,x=dx,y=dy)
        ax=sns.stripplot(data=DDF1, palette="Set2", edgecolor="white",size=2,orient=ort,\
                        x=dx,y=dy,jitter=1,zorder=0)
        ax=sns.boxplot(data=DDF1, color="black",orient=ort,width=.15,x=dx,y=dy,zorder=10,\
                    showcaps=True,boxprops={'facecolor':'none', "zorder":10},\
                    showfliers=True,whiskerprops={'linewidth':2, "zorder":10},saturation=1)
        #ax = sns.pointplot(x=dx, y=dy, data=ddf,color='red')
        # Finalize the figure
        f.suptitle( ' '.join(plot_title) + ' Forced Lorenz State Fluctuations - Learning ' + str(total_learnings + 1), fontsize=16)
        ax.set(ylim=(20, -20))
        sns.despine(left=True)



        #No Forcing      
        X_flux_noforcing = []
        Y_flux_noforcing = []
        Z_flux_noforcing = []
        for x in range(interval_of_analysis):
            X_flux_noforcing.append(UFSD[x,0]-Xequil)
            Y_flux_noforcing.append(UFSD[x,1]-Yequil)
            Z_flux_noforcing.append(UFSD[x,2]-Zequil)
        index = list(range(1, len(X_flux_noforcing)+1))
        AllData2 = np.array([index, X_flux_noforcing, Y_flux_noforcing, Z_flux_noforcing])
        AllData2 = np.transpose(AllData2)
        AllData2 = pd.DataFrame(AllData2, columns = ['Index', 'X', 'Y', 'Z'])
        DDF2 = pd.melt(AllData2, id_vars = ['Index'], value_vars = ['X', 'Y', 'Z'], var_name = 'state values', value_name = 'Dist From Equil')

        f, ax = plt.subplots(figsize=(12, 11))

        dy = 'Dist From Equil'; dx = 'state values'; ort = "v"
        # Draw a violinplot with a narrower bandwidth than the default
        ax=pt.half_violinplot(data = DDF2, palette = "Set2", bw=.2,  linewidth=1,cut=0.,\
                        scale="area", width=.8, inner=None,orient=ort,x=dx,y=dy)
        ax=sns.stripplot(data=DDF2, palette="Set2", edgecolor="white",size=2,orient=ort,\
                        x=dx,y=dy,jitter=1,zorder=0)
        ax=sns.boxplot(data=DDF2, color="black",orient=ort,width=.15,x=dx,y=dy,zorder=10,\
                    showcaps=True,boxprops={'facecolor':'none', "zorder":10},\
                    showfliers=True,whiskerprops={'linewidth':2, "zorder":10},saturation=1)
        #ax = sns.pointplot(x=dx, y=dy, data=ddf,color='red')
        # Finalize the figure
        f.suptitle('Unforced Lorenz State Fluctuations - Learning ' + str(total_learnings + 1), fontsize=16)
        ax.set(ylim=(20, -20))
        sns.despine(left=True)

    #Show plots only after all Learnings are Complete
    plt.show()


#Testing the Definition
#DDPGcontrol(Episodes, random_steps, max_episode_steps, update_freq, Learnings)