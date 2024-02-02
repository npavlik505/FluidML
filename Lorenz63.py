sigma = 10
rho = 28
beta = 8/3

X = True
Y = True
Z = True

time = 20
dt = .001
datasets = 1

Episodes = 5
random_steps = 500 #Ususally at 500
max_episode_steps = 5000 #Usually at 5000
update_freq = 5
Learnings = 2 #Was at 5
Force_X = True
Force_Y = False
Force_Z = False

from Source.LorenzEnvironment import LorenzEnv
env = LorenzEnv(sigma, rho, beta, dt, X, Y, Z, Force_X, Force_Y, Force_Z)

# Modelling Loop with or without masking
def LorenzModelLoop(env, time, dt, datasets):
    from FeatureSelection.LorenzFeatureSelection import GenerateData
    GenerateData(env, time, dt, datasets) #Produces datasets with same name scheme as "CurrentFile" below
    from Model import SINDy
    for i in range(datasets):
        CurrentFile = 'LorenzDataSet_20s_Set' + str(i+1) + '.h5py'
        print('SINDy model for set ' + str(i+1) )
        SINDy.StandardSindy(CurrentFile, dt)

# #Testing Modelling Loop with or without masking below
# LorenzModelLoop(env, time, dt, datasets)


#Control Loop with or without masking
def LorenzControlLoop(env, dt, Episodes, random_steps, max_episode_steps, update_freq, Learnings):
    from ControlApplication.DDPG_lorenz_control import DDPGcontrol
    DDPGcontrol(env, dt, Episodes, random_steps, max_episode_steps, update_freq, Learnings)

#Testing Control Loop w/out masking below
LorenzControlLoop(env, dt, Episodes, random_steps, max_episode_steps, update_freq, Learnings)