## Test of TimeSeriesData below
from Source.TimeSeriesData import LorenzDataGenerator

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)

sigma = 10
rho = 28
beta = 8/3
dt = .001
time = 20
datasets = 3

Test = LorenzDataGenerator(sigma, rho, beta, time, dt, datasets)
Test.GenerateData()

import h5py
import numpy as np
view = h5py.File('LorenzDataSet_20s_Set1.h5py', 'r')
view.keys()