###Package Structure
#LorenzAPI
    #TestImports.py
    #Source
        #TimeSeriesData.py
        #xForcingLorenz.py
    #Select Data
    #Model
        #PySINDy (a Package)
    #Control
        #DDPG.py

# class LorenzAPI:
#     def __init__(self, sigma, rho, beta, dt, time, datasets):
#         import Source, SelectData, Model, Control 
    #Case1: If only source is being modified: Lorenz System Parameters (rho, sigma, beta), dt, and time can be modified
        #Case1.A: Control Loop (->Source->DataSelection->Control->)
        #Case1.B: Modeling Loop (Source-DataSelection<->Model, Model Modified)
        #Case1.C: Control from Extracted Model Loop (Model<->Control)
        #Case1.D: Data Selection Loop (Source-DataSelection<->Model, DataSeletion Modified)

print('Variable Pool')
sigma = 10
rho = 28
beta = 8/3

time = 20
dt = .001
datasets = 1

Episodes = 5
random_steps = 500 #Ususally at 500
max_episode_steps = 5000 #Usually at 5000
update_freq = 5
Learnings = 2


#Modelling Loop Completed w/out masking
def LorenzModelLoop1(sigma, rho, beta, time, dt, datasets):
    from Source.TimeSeriesData import LorenzDataGenerator
    data = LorenzDataGenerator(sigma, rho, beta, time, dt, datasets)
    data.GenerateData() #Produces datasets with same name scheme as "CurrentFile" below
    from Model import SINDy
    for i in range(datasets):
        CurrentFile = 'LorenzDataSet_20s_Set' + str(i+1) + '.h5py'
        print('SINDy model for set ' + str(i+1) )
        SINDy.StandardSindy(CurrentFile, dt)

##Testing out Model loop below
#LorenzModelLoop1(sigma, rho, beta, time, dt, datasets)

#Control Loop w/out masking
def LorenzControlLoop1(sigma, rho, beta, dt, Episodes, random_steps, max_episode_steps, update_freq, Learnings):
    from Control.DDPG_lorenz_control import DDPGcontrol
    DDPGcontrol(sigma, rho, beta, dt, Episodes, random_steps, max_episode_steps, update_freq, Learnings)

#Testing out Control loop below
LorenzControlLoop1(sigma, rho, beta, dt, Episodes, random_steps, max_episode_steps, update_freq, Learnings)

#%%
# import h5py
# import numpy as np
# view = h5py.File('LorenzDataSet_20s_Set1.h5py', 'r')
# view.keys()
# view1 = view.get("X_Data")
# view1 = np.array(view1)
# print(view1.shape)
#%%