## SELECTION OF X, Y, AND/OR Z for MODELLING

#Add to python path
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)

import h5py


# Selects the chosen variables (X, Y, or Z) and stores them in an hdf5 file
def SelectFeatures(env, time, datasets):
    for i in range(datasets):
        file = h5py.File('DataSet' + str(i+1) + '.h5py', 'w')
        s = env.reset()
        DataSetX = []
        DataSetY = []
        DataSetZ = []
        for x in range(int(time/env.dt)):
            DataSetX.append(s[0].item())
            DataSetY.append(s[1].item())
            DataSetZ.append(s[2].item())          
            s += env.UnforcedSystem(s)
            if x == ((time/env.dt)-1):
                if env.X == True:
                    file.create_dataset("X", data = DataSetX)
                if env.Y == True:
                    file.create_dataset("Y", data = DataSetY)
                if env.Z == True:
                    file.create_dataset("Z", data = DataSetZ)
                print(env.SystemName + 'DataSet' + str(i+1) +'_'+ str(time) + 's')
                file.close()


