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
    feature_names = list(DataSet0.keys())
    DataList = []
    for names in feature_names:
        print(names)
        #Data1 = [DataSet0.get(names)]
        vector_list = DataSet0[names][:]
        DataList.append(vector_list)
        print(np.shape(DataList)) 
    Data = np.stack(DataList)
    Data = np.transpose(Data)
    #Data = np.array(Data)
    print('This is inputed data file', np.shape(Data))
    DataSet0.close()
    opt = ps.STLSQ(threshold = 0.5) # Choose an optimizer: Sequentially thresholded least squares algorithm, and pick threshold (lambda value, i.e. reularizer strength)
    model = ps.SINDy(feature_names = feature_names, optimizer = opt)
    model.fit(Data, t = dt) #D is the training data and the timestep btwn data points must be specified
    model.print()


## For reference after changes. Copied at 6:34pm 1.23.24
# def StandardSindy(CurrentFile, dt):
#     DataSet0 = h5py.File(CurrentFile, 'r')
#     Data = DataSet0.get('XYZ_Data')
#     Data = np.array(Data)
#     DataSet0.close()
#     feature_names = ['x', 'y', 'z']
#     opt = ps.STLSQ(threshold = 0.5) # Choose an optimizer: Sequentially thresholded least squares algorithm, and pick threshold (lambda value, i.e. reularizer strength)
#     model = ps.SINDy(feature_names = feature_names, optimizer = opt)
#     model.fit(Data, t = dt) #x_train is the training data and the timestep btwn data points must be specified
#     model.print()