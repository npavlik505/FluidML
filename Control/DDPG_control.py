### DDPG APPLIED TO THE LORENZ SYSTEM

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


def DDPGcontrol(env, Episodes, random_steps, max_episode_steps, update_freq, Learnings):
    from ControlMethod import DDPG

    testt1 = 0
    testt2 = 0

    state_dim = env.observation_space.shape[0] #Dimension of the state space (in this case three continous values for x, y, and z of the Lorenz System)
    action_dim = env.action_space.shape[0] #Dimension of the action space (in this case continuous values between 0 and 50)
    max_action = float(env.action_space.high[0]) #The max forcing that can be applied

    agent = DDPG.ddpg(state_dim, action_dim, max_action) #Initializes the DDPG algorithm (Class containing the Actor-Critic NN that chooses x forcing action)
    replay_buffer = DDPG.ReplayBuffer(state_dim, action_dim) #Initializes the ReplayBuffer (stores s,a,s_,r sequences for batch learning)

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration


    for total_learnings in range(Learnings):
        print("This is learning ", str(total_learnings + 1))
        #Test1 and Test2 Used is counting/iteration
        testt1 = 0
        testt2 = 0

        #Initialize lists to store the average MSE for forcing and non-forcing cases
        LearningAveMSE = []
        LearningAveMSE_NF = []

        #Code below resets network parameters to original set between Learnings, making learnings independent of each other but helping to eliminate random interLearning variation
        if total_learnings > 0:
            # #Place before and after desired network below to print weights before and after reinitialization            
            # print("Learning's Actor params before init")
            # for mod1 in agent.actor.modules():
            #     if isinstance(mod1, nn.Linear):
            #         print(mod1.weight)
            # print("Learning's Actor params after init")
            # for mod2 in agent.actor.modules():
            #     if isinstance(mod2, nn.Linear):
            #         print(mod2.weight)           
            OriginalActorParams = torch.load('InitialActorParameters.pt')
            agent.actor.load_state_dict(OriginalActorParams)
            OriginalActorTargetParams = torch.load('InitialActorTargetParameters.pt')
            agent.actor_target.load_state_dict(OriginalActorTargetParams)
            OriginalCriticParams = torch.load('InitialCriticParameters.pt')
            agent.critic.load_state_dict(OriginalCriticParams)    
            OriginalCriticTargetParams = torch.load('InitialCriticTargetParameters.pt')
            agent.critic_target.load_state_dict(OriginalCriticTargetParams)      

        for Episode in range(Episodes):
            #Generate initial value for episode (note: forced ICs same as Unforced ICs)
            s = env.reset()
            state_data = torch.tensor(s)
 
            s_noforcing = torch.clone(s)
            state_data_noforcing = s_noforcing.view(1,3)

            #Decriments the random_steps value so fewer are taken each episode (Consider if random steps are actually needed)
            if random_steps >= 0:
                random_steps += -25

            for episode_steps in range(max_episode_steps):

                #Generate the unforced lorenz values
                s_noforcing += env.UnforcedSystem(s_noforcing)
                state_data_noforcing = torch.cat((state_data_noforcing, s_noforcing.view(1,3)), dim = 0)

                #Randomly select, or have the NN choose, an action for the current step (i.e. forcing value on x, y, or z)
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

                #Calculate the average MSE for each episode with forcing 
                if episode_steps == 0:
                    print('Start of Episode ' + str(Episode+1))
                    EpAveMSE = (((abs(s[0] - env.Ftarget[0])+abs(s[1] - env.Ftarget[1])+abs(s[2] - env.Ftarget[2]))**2))
                else:
                    EpAveMSE = ((EpAveMSE*(episode_steps-1)) + (abs(s[0] - env.Ftarget[0])**2+abs(s[1] - env.Ftarget[1])**2+abs(s[2] - env.Ftarget[2])**2))/episode_steps


                #Calculate the average MSE for each episode without forcing 
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
        if env.Force_X == True:
            plot_title.append('X')
        if env.Force_Y == True:
            plot_title.append('Y')
        if env.Force_Z == True:
            plot_title.append('Z')
        fig1 = plt.figure()
        forcing = fig1.add_subplot(111, projection = '3d')
        forcing.plot(BSD[:,0], BSD[:,1], BSD[:,2], label = ('Best' + ' '.join(plot_title) + 'Forcing Policy'))
        forcing.set_title(env.SystemName + ': Best ' + ' '.join(plot_title) + ' Forcing Policy - Learning ' + str(total_learnings + 1))
        forcing.set_xlabel('X', labelpad = 10)
        forcing.set_ylabel('Y', labelpad = 10)
        forcing.set_zlabel('Z', labelpad = 10)

        #Plot the unforced state data
        fig2 = plt.figure()
        noforcing = fig2.add_subplot(111, projection = '3d')
        noforcing.plot(UFSD[:,0], UFSD[:,1], UFSD[:,2], label = 'Corresponding Unforced Lorenz')              
        noforcing.set_title(env.SystemName + ': Corresponding Unforced Dynamics - Learning ' + str(total_learnings + 1))
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
        plt.title(env.SystemName + ': MSE for Learning ' + str(total_learnings + 1))
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
        # Finalize the figure
        f.suptitle( env.SystemName + ': ' + ' '.join(plot_title) + ' Forced State Fluctuations - Learning ' + str(total_learnings + 1), fontsize=16)
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
        # Finalize the figure
        f.suptitle(env.SystemName + ': Unforced State Fluctuations - Learning ' + str(total_learnings + 1), fontsize=16)
        ax.set(ylim=(20, -20))
        sns.despine(left=True)

    #Show plots only after all Learnings are Complete
    plt.show()