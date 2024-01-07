import h5py
import numpy as np
import pysindy as ps

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)

def StandardSindy(CurrentFile, dt):
    DataSet0 = h5py.File(CurrentFile, 'r')
    Data = DataSet0.get('XYZ_Data')
    Data = np.array(Data)
    DataSet0.close()
    feature_names = ['x', 'y', 'z']
    opt = ps.STLSQ(threshold = 0.5) # Choose an optimizer: Sequentially thresholded least squares algorithm, and pick threshold (lambda value, i.e. reularizer strength)
    model = ps.SINDy(feature_names = feature_names, optimizer = opt)
    model.fit(Data, t = dt) #x_train is the training data and the timestep btwn data points must be specified
    model.print()



# import h5py
# import numpy as np
# import pysindy as ps    
# DataSet0 = h5py.File('LorenzDataSet_20s_Set1.h5py', 'r')
# print(DataSet0.keys())
# Data = DataSet0.get('XYZ_Data')
# Data = np.array(Data)
# print(Data.shape)
# DataSet0.close()

# # Using https://www.youtube.com/watch?v=HvOdfwgTPnM (i.e. PySINDy tutorial 2: Choosing algorithm hyperparameters)
# feature_names = ['x', 'y', 'z']
# opt = ps.STLSQ(threshold = 0.5) # Choose an optimizer: Sequentially thresholded least squares algorithm, and pick threshold (lambda value, i.e. reularizer strength)
# model = ps.SINDy(feature_names = feature_names, optimizer = opt)
# model.fit(Data, t = .001) #x_train is the training data and the timestep btwn data points must be specified
# model.print()