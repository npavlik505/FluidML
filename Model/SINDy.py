import h5py
import numpy as np
import pysindy as ps

DataSet0 = h5py.File('LorenzDataSet20s_0.h5py', 'r')
print(DataSet0.keys())
Data = DataSet0.get('XYZ_Data')
Data = np.array(Data)
print(Data.shape)
DataSet0.close()

# Using https://www.youtube.com/watch?v=HvOdfwgTPnM (i.e. PySINDy tutorial 2: Choosing algorithm hyperparameters)
feature_names = ['x', 'y', 'z']
opt = ps.STLSQ(threshold = 0.5) # Choose an optimizer: Sequentially thresholded least squares algorithm, and pick threshold (lambda value, i.e. reularizer strength)
model = ps.SINDy(feature_names = feature_names, optimizer = opt)
model.fit(Data, t = .001) #x_train is the training data and the timestep btwn data points must be specified
model.print()