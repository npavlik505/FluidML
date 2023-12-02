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

print('Hello')
sigma = 10
rho = 28
beta = 8/3
dt = .001
time = 20
datasets = 3

def ModelLoop1(sigma, rho, beta, time, dt, datasets):
    from Source.TimeSeriesData import LorenzDataGenerator
    data = LorenzDataGenerator(sigma, rho, beta, time, dt, datasets)
    data.GenerateData()
    from Model import SINDy
    for i in range(datasets):
        CurrentFile = 'LorenzDataSet_20s_Set' + str(i+1) + '.h5py'
        print('SINDy model for set ' + str(i+1) )
        SINDy.StandardSindy(CurrentFile, dt)

ModelLoop1(sigma, rho, beta, time, dt, datasets)



#%%
# import h5py
# import numpy as np
# view = h5py.File('LorenzDataSet_20s_Set1.h5py', 'r')
# view.keys()
# view1 = view.get("X_Data")
# view1 = np.array(view1)
# print(view1.shape)
#%%