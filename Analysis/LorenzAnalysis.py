## Metrics of Algorithm Performance for the Lorenz System are Calculated Below

#Add to python path
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)

#Imports for DDPG
import numpy as np
import matplotlib.pyplot as plt
import torch
#Imports for RainClouds plots
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid",font_scale=2)
import ptitprince as pt
import torch.nn as nn

def Analysis(env, max_episode_steps, run):
    #INITIALIZE NETWORK
    from Control_Method import DDPG

    state_dim = env.observation_space.shape[0] #Dimension of the state space (in this case three continous values for x, y, and z of the Lorenz System)
    action_dim = env.action_space.shape[0] #Dimension of the action space (in this case continuous values between 0 and 50)
    max_action = float(env.action_space.high[0]) #The max forcing that can be applied

    agent = DDPG.ddpg(state_dim, action_dim, max_action) #Initializes the DDPG algorithm (Class containing the Actor-Critic NN that chooses x forcing action)

    #ITERATIVELY ACCESS THE BEST PERFORMING PARAMETERS    
    #Variable to Labeling Analysis Output
    LearningNumber = 0
    #Access Current Run
    AnalysisDir = os.path.dirname(os.path.abspath(__file__))
    FluidMLDir = os.path.abspath(os.path.join(AnalysisDir, '..'))
    BestParamDirectory = os.path.join(FluidMLDir, run, 'Best_Parameters')

    for directories in os.listdir(BestParamDirectory):
        directory_path = os.path.join(BestParamDirectory, directories)
        print("directory path: ", directory_path)
        #Access folders containing best parameters of each learning
        if os.path.isdir(directory_path) and directories.startswith("BestParams"):
            for files in os.listdir(directory_path):
                file_path = os.path.join(directory_path, files)
                #Access the parameters of each folder
                print("file path: ", file_path)

                if os.path.isfile(file_path) and files.startswith('BestParamsActor'):
                    #Increment labeling variable
                    LearningNumber += 1
                    #Load actor parameters
                    agent.actor.load_state_dict(torch.load(file_path))
                    agent.actor.eval()

                    print("Learning's Actor params after init")
                    for mod2 in agent.actor.modules():
                        if isinstance(mod2, nn.Linear):
                            print(mod2.weight)
                    
                    #Create path to analysis file to contain plots  
                    Analysis_dir = os.path.join(FluidMLDir, run, 'Analysis', f'Analysis_Learning{LearningNumber}')
                    print('Analysis_dir Path: ', Analysis_dir)

                    if not os.path.exists(Analysis_dir):
                        os.makedirs(Analysis_dir)

                    for file_name in os.listdir(Analysis_dir):
                        file_path = os.path.join(Analysis_dir, file_name)
                        if os.path.isfile(file_path):
                            os.remove(file_path)

                    #INITIALIZE ENVIRONMENT FOR SIMULATION
                    #Generate initial state value (same for forced and unforced system)
                    s = env.reset()
                    state_data = torch.tensor(s)

                    s_noforcing = torch.clone(s)
                    state_data_noforcing = s_noforcing.view(1,3)



                    #TESTING THE ACTOR PARAMETERS
                    a = agent.actor(s)
                    test1 = agent.actor(torch.tensor([0.0, 0.0, 0.0]))
                    test2 = agent.actor(torch.tensor([-0.5, -0.5, 18]))
                    test3 = agent.actor(torch.tensor([0.8949, 0.4012, 12.1023]))
                    print('testing_outcomes:')
                    print('This should be 0: ', test1)
                    print('test2: ', test2)
                    print('test3: ', test3)


                    #RUN SIMULATION
                    #Begin analysis run using actor network params for each learning
                    print('Analysis execution for learning ' + str(LearningNumber) + ' has begun')
                    for episode_steps in range(max_episode_steps):

                        #Generate the unforced lorenz values
                        s_noforcing += env.UnforcedSystem(s_noforcing)
                        state_data_noforcing = torch.cat((state_data_noforcing, s_noforcing.view(1,3)), dim = 0)

                        #Generate the forced lorenz falues
                        s_, r, terminated, truncated = env.step(a, s)
                        state_data = torch.cat([state_data.view(-1,3), s_.view(1,3)])
                        s = s_

                        #Calculate the average MSE for each episode with forcing (EpAveMSE) and w/o forcing (EpAveMSE_NF)
                        if episode_steps == 0:
                            print('Start of Analysis: BestParams_Learning' + str(LearningNumber))
                            EpAveMSE = (((abs(s[0] - env.Ftarget[0])+abs(s[1] - env.Ftarget[1])+abs(s[2] - env.Ftarget[2]))**2))
                            EpAveMSE_NF = (((abs(s_noforcing[0] - env.Ftarget[0])+abs(s[1] - env.Ftarget[1])+abs(s[2] - env.Ftarget[2]))**2))
                        else:
                            EpAveMSE = ((EpAveMSE*(episode_steps-1)) + (abs(s[0] - env.Ftarget[0])**2+abs(s[1] - env.Ftarget[1])**2+abs(s[2] - env.Ftarget[2])**2))/episode_steps
                            EpAveMSE_NF = ((EpAveMSE_NF*(episode_steps-1)) + (abs(s_noforcing[0] - env.Ftarget[0])**2+abs(s_noforcing[1] - env.Ftarget[1])**2+abs(s_noforcing[2] - env.Ftarget[2])**2))/episode_steps
                       
                        #Executes after the final step in each episode    
                        if episode_steps == max_episode_steps-1:

                            #Store the State Data for the Forcing and Non-Forcing Cases as np arrays for plotting
                            BSD = np.array(state_data)
                            UFSD = np.array(state_data_noforcing)
                            print('Finished Analysis: BestParams_Learning' + str(LearningNumber))

                    #MSE AND PLOT GENERATION
                    #Print the average Mean Squared Error for the Episode
                    print('Episode Ave MSE:', EpAveMSE)
                    print('Episode Ave MSE No Forcing:', EpAveMSE_NF)
                    print()

                    #if terminated:
                    #break

                    #Plot the forced state data
                    plot_title = []
                    if env.Force_X == True:
                        plot_title.append('X')
                    if env.Force_Y == True:
                        plot_title.append('Y')
                    if env.Force_Z == True:
                        plot_title.append('Z')
                    fig1 = plt.figure()
                    forcing = fig1.add_subplot(111, projection = '3d')
                    forcing.plot(BSD[:,0], BSD[:,1], BSD[:,2], label = ('Best' + ' '.join(plot_title) + 'Forcing Policy'))
                    forcing.set_title(env.SystemName + ': Best ' + ' '.join(plot_title) + ' Forcing Policy - Learning ' + str(LearningNumber))
                    forcing.set_xlabel('X', labelpad = 10)
                    forcing.set_ylabel('Y', labelpad = 10)
                    forcing.set_zlabel('Z', labelpad = 10)

                    Analysis_Forced_State_Data = 'Analysis_Forced_Learning' + str(LearningNumber) + '.png'
                    Analysis_state_plot_path1 = os.path.join(Analysis_dir, Analysis_Forced_State_Data)
                    plt.savefig(Analysis_state_plot_path1)
                    
                    #Plot the unforced state data
                    fig2 = plt.figure()
                    noforcing = fig2.add_subplot(111, projection = '3d')
                    noforcing.plot(UFSD[:,0], UFSD[:,1], UFSD[:,2], label = 'Corresponding Unforced Lorenz')              
                    noforcing.set_title(env.SystemName + ': Corresponding Unforced Dynamics - Learning ' + str(LearningNumber))
                    noforcing.set_xlabel('X', labelpad = 10)
                    noforcing.set_ylabel('Y', labelpad = 10)
                    noforcing.set_zlabel('Z', labelpad = 10)

                    Analysis_Unforced_State_Data = 'Analysis_Unforced_Learning' + str(LearningNumber) + '.png'
                    Analysis_state_plot_path2 = os.path.join(Analysis_dir, Analysis_Unforced_State_Data)
                    plt.savefig(Analysis_state_plot_path2)                        

                else: break
        else:
            print('No best performing parameter folders found')
            break
    return