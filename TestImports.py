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

sigma = 10
rho = 28
beta = 8/3
dt = .001
time = 20
datasets = 3

class LorenzAPI:
    def __init__(self, sigma, rho, beta, dt, time, datasets)
    import Source, SelectData, Model, Control 
    #Case1: If only source is being modified: Lorenz System Parameters (rho, sigma, beta), dt, and time can be modified
        #Case1.A: Control Loop (->Source->DataSelection->Control->)
        #Case1.B: Modeling Loop (Source-DataSelection<->Model, Model Modified)
        #Case1.C: Control from Extracted Model Loop (Model<->Control)
        #Case1.D: Data Selection Loop (Source-DataSelection<->Model, DataSeletion Modified)

    #Case1.B
    from Source import TimeSeriesData
    from Model import SINDy
    #Primative PySINDy class (Use until full functionality of PySINDy known)
    def Case1B():
            from Source import TimeSeriesData
            TimeSeriesData.LorenzDataGenerator(sigma, rho, beta, time, dt, datasets)

