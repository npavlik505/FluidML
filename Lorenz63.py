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
Learnings = 1
Force_X = True
Force_Y = True
Force_Z = False


# Modelling Loop with or without masking
def LorenzModelLoop(sigma, rho, beta, time, dt, datasets, X, Y, Z):
    from Source.TimeSeriesData import LorenzDataGenerator
    data = LorenzDataGenerator(sigma, rho, beta, time, dt, datasets, X, Y, Z)
    data.GenerateData() #Produces datasets with same name scheme as "CurrentFile" below
    from Model import SINDy
    for i in range(datasets):
        CurrentFile = 'LorenzDataSet_20s_Set' + str(i+1) + '.h5py'
        print('SINDy model for set ' + str(i+1) )
        SINDy.StandardSindy(CurrentFile, dt)

#Testing Modelling Loop with or without masking below
#LorenzModelLoop(sigma, rho, beta, time, dt, datasets, X, Y, Z)


#Control Loop with or without masking
def LorenzControlLoop(sigma, rho, beta, dt, Episodes, random_steps, max_episode_steps, update_freq, Learnings, Force_X, Force_Y, Force_Z):
    from Control.DDPG_lorenz_control import DDPGcontrol
    DDPGcontrol(sigma, rho, beta, dt, Episodes, random_steps, max_episode_steps, update_freq, Learnings, Force_X, Force_Y, Force_Z)

#Testing Control Loop w/out masking below
LorenzControlLoop(sigma, rho, beta, dt, Episodes, random_steps, max_episode_steps, update_freq, Learnings, Force_X, Force_Y, Force_Z)