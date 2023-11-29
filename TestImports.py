sigma = 10
rho = 28
beta = 8/3
dt = .001
time = 20
datasets = 3

import sys
#print('TEST_', sys.path)
sys.path.append('/Users/natha/OneDrive/Desktop/Lorenz-API/Source/TimeSeriesData')
from TimeSeriesData import LorenzDataGenerator

EnvTest = LorenzDataGenerator(sigma, rho, beta, time, dt, datasets)
EnvTest.GenerateData()