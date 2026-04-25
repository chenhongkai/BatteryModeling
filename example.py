#%%
import numpy as np
import matplotlib.pyplot as plt

from ParameterIdentification import Identification

Qnom = 20              # Nominal capacity [Ah]
TVT_ = (.7, .15, .15)  # Ratio of Training, validation, test data
T = 10#00              # Iterations
N = 20#0               # population size
n_jobs = -1            # joblib parallel
algorithm = 'STA'      # State transition algorithm
objective = 'RMSE'
Nsample = 5_00#00      # Samples for objective weight estimation
IC = 0.9               # Current C-rate
tC = 25                # Temperature [°C]
onset = 0              # Charging onset time [s]
duration = 1000        # Charging duration [s]
ΔtEIS = 50             # Sampling interval of EIS
ΔtUDC = 10             # Sampling interval of UDC
Δt = 10                # Time step
f_ = np.logspace(np.log10(400), np.log10(4), 17)  # EIS frequencies [Hz]
targets_ = ('UDC', 'Zreal', 'Zimag')
pnormfixed_ = {}        # Normalized values of parameters that are fixed

task = Identification(
    Qnom=Qnom,
    IC=IC,
    tC=tC,
    TVT_=TVT_,
    onset=onset, duration=duration,
    Δt=Δt, ΔtUDC=ΔtUDC, ΔtEIS=ΔtEIS,
    f_=f_,
    T=T, N=N,
    n_jobs=n_jobs,
    verbose=True,
    algorithm=algorithm, objective=objective,)


chargingData = np.load('chargingData.npz',
    allow_pickle=True)['chargingData'].item()

task.receive_measured_data(
    tUDCmea_=chargingData['tUDCmea_'],
    UDCmea_=chargingData['UDCmea_'],
    tZmea_=chargingData['tZmea_'],
    fZmea_=chargingData['fZmea_'],
    Zmea_=chargingData['Zmea_'],
    )

record = task.identify(
    pnormfixed_,
    targets_=targets_,
    Nsample=Nsample,
    )
