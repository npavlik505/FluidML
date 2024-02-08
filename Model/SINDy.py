## APPLICATION OF SINDy METHOD FOR MODELLING SYSTEM DYNAMICS

#Add to python path
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)

import h5py
import numpy as np
import pysindy as ps

def StandardSindy(CurrentFile, dt):
    DataSet = h5py.File(CurrentFile, 'r')
    feature_names = list(DataSet.keys())
    DataList = []
    for names in feature_names:
        vector_list = DataSet[names][:]
        DataList.append(vector_list)
    Data = np.stack(DataList)
    Data = np.transpose(Data)
    DataSet.close()
    opt = ps.STLSQ(threshold = 0.5) # Choose an optimizer: Sequentially thresholded least squares algorithm, and pick threshold (lambda value, i.e. reularizer strength)
    model = ps.SINDy(feature_names = feature_names, optimizer = opt)
    model.fit(Data, t = dt) #Data is the training data and the timestep btwn data points must be specified
    model.print()

#REFERENCED FOR CREATING SINDy.py FILE
#https://www.youtube.com/watch?v=SfIJiuJ38W0