#Create .hdf5 files of lorenz X,Y,Z
import h5py
import numpy as np
import torch

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)


class LorenzDataGenerator:

    def __init__(self, sigma, rho, beta, time, dt, datasets, X, Y, Z):
        #Parameters
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.time = time
        self.dt = dt
        self.steps = self.time/self.dt
        self.datasets = datasets
        self.X = X
        self.Y = Y
        self.Z = Z

    def LorenzData(self, s):
        s = s.to(torch.float32)
        dFL = [(self.sigma*(s[1] - s[0])),
            (s[0]*(self.rho - s[2]) - s[1]),
            (s[0]*s[1] - self.beta*s[2])]
        #list comprehension to multiply a float by a list
        dFL = [x*self.dt for x in dFL]
        dFL = torch.tensor(dFL)
        return dFL

    #Random Initial Conditions
    def reset(self):
        self.F0 = np.array([np.random.uniform(-21, 21), np.random.uniform(-16, 16), np.random.uniform(-1, 46)])
        self.F0 = torch.tensor(self.F0)
        return self.F0

    #Take one step forward in the Lorenz System
    def step(self, s):
        s = torch.tensor(s)
        s += torch.tensor(self.LorenzData(s))
        return s


    #Generate the data sets
    def GenerateData(self):
        for i in range(self.datasets):
            file = h5py.File('LorenzDataSet_20s_Set' + str(i+1) + '.h5py', 'w')
            s = self.reset()
            DataSet = []
            DataSetX = []
            DataSetY = []
            DataSetZ = []
            for x in range(int(self.steps)):
                DataSet.append(s)
                DataSetX.append(s[0])
                DataSetY.append(s[1])
                DataSetZ.append(s[2])
                s = self.step(s)
                if x == (self.steps-1):
                    # file.create_dataset("XYZ_Data", data = DataSet)
                    if self.X == True:
                        file.create_dataset("X_Data", data = DataSetX)
                    if self.Y == True:
                        file.create_dataset("Y_Data", data = DataSetY)
                    if self.Z == True:
                        file.create_dataset("Z_Data", data = DataSetZ)
                    print('LorenzDataSet_' + str(self.time) + 's_Set' + str(i+1) + '_complete')
                    file.close()



##### This is the original code that has been copied for reference at 6:19pm 1.23 before edits for data masking
# class LorenzDataGenerator:

#     def __init__(self, sigma, rho, beta, time, dt, datasets):
#         #Parameters
#         self.sigma = sigma
#         self.rho = rho
#         self.beta = beta
#         self.time = time
#         self.dt = dt
#         self.steps = self.time/self.dt
#         self.datasets = datasets

#     def LorenzData(self, s):
#         s = s.to(torch.float32)
#         dFL = [(self.sigma*(s[1] - s[0])),
#             (s[0]*(self.rho - s[2]) - s[1]),
#             (s[0]*s[1] - self.beta*s[2])]
#         #list comprehension to multiply a float by a list
#         dFL = [x*self.dt for x in dFL]
#         dFL = torch.tensor(dFL)
#         return dFL

#     #Random Initial Conditions
#     def reset(self):
#         self.F0 = np.array([np.random.uniform(-21, 21), np.random.uniform(-16, 16), np.random.uniform(-1, 46)])
#         self.F0 = torch.tensor(self.F0)
#         return self.F0

#     #Take one step forward in the Lorenz System
#     def step(self, s):
#         s = torch.tensor(s)
#         s += torch.tensor(self.LorenzData(s))
#         return s


#     #Generate the data sets
#     def GenerateData(self):
#         for i in range(self.datasets):
#             file = h5py.File('LorenzDataSet_20s_Set' + str(i+1) + '.h5py', 'w')
#             s = self.reset()
#             DataSet = []
#             DataSetX = []
#             DataSetY = []
#             DataSetZ = []
#             for x in range(int(self.steps)):
#                 DataSet.append(s)
#                 DataSetX.append(s[0])
#                 DataSetY.append(s[1])
#                 DataSetZ.append(s[2])
#                 s = self.step(s)
#                 if x == (self.steps-1):
#                     file.create_dataset("XYZ_Data", data = DataSet)
#                     file.create_dataset("X_Data", data = DataSetX)
#                     file.create_dataset("Y_Data", data = DataSetY)
#                     file.create_dataset("Z_Data", data = DataSetZ)
#                     print('LorenzDataSet_' + str(self.time) + 's_Set' + str(i+1) + '_complete')
#                     file.close()