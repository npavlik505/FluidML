sigma = 10
rho = 28
beta = 8/3

X = True
Y = True
Z = True

time = 10
dt = .001
datasets = 1

Episodes = 40
random_steps = 500 #Ususally at 500
max_episode_steps = 5000 #Usually at 5000
update_freq = 5
Learnings = 2 #Was at 5
Force_X = True
Force_Y = False
Force_Z = False
Max_Force = 50

run = '20240628.133325'

from System.LorenzEnvironment import LorenzEnv
env = LorenzEnv(sigma, rho, beta, time, dt, X, Y, Z, Force_X, Force_Y, Force_Z, Max_Force)

# Modelling Loop with or without masking
def ModelLoop(env, datasets):
    from Feature_Selection.LorenzFeatureSelection import SelectFeatures
    SelectFeatures(env, datasets) #Produces datasets with same name scheme as "CurrentFile" below
    from Model import SINDy
    for i in range(datasets):
        CurrentFile = 'DataSet' + str(i+1) + '.h5py'
        print('Model for dataset ' + str(i+1) )
        SINDy.StandardSindy(CurrentFile, env.dt)

# #Testing Modelling Loop with or without masking below
# ModelLoop(env, datasets)



#Control Loop with or without masking
def ControlLoop(env, Episodes, random_steps, max_episode_steps, update_freq, Learnings):
    from Control_Implementation.DDPG_control import DDPGcontrol
    DDPGcontrol(env, Episodes, random_steps, max_episode_steps, update_freq, Learnings)

# #Testing Control Loop w/out masking below
# ControlLoop(env, Episodes, random_steps, max_episode_steps, update_freq, Learnings)

# #Testing Control Loop w/out masking below
# for forcing_sweep in range(5,51,5):
#     Max_Force = forcing_sweep
#     from System.LorenzEnvironment import LorenzEnv
#     env = LorenzEnv(sigma, rho, beta, time, dt, X, Y, Z, Force_X, Force_Y, Force_Z, Max_Force)
#     ControlLoop(env, Episodes, random_steps, max_episode_steps, update_freq, Learnings)



#Testing Analysis
from Analysis.LorenzAnalysis import Analysis
Analysis(env, max_episode_steps, run) 