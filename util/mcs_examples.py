# Simulate data and predicts

from AI.util import MCS
import numpy as np
import pandas as pd

def DataGeneratingProcess(a0, a1, a2, b1, N):
    y = np.zeros(N)
    np.random.seed(123)
    y[0] = 5
    epsilon = np.random.normal(0,1,N)
    for t in range(1, N-1):
        y[t+1] = a0 + a1*y[t-1]+a2*y[t-2]+epsilon[t]+b1*epsilon[t-1]
    return y

N = 100
y  = DataGeneratingProcess(0.1,0.8,-0.2,0.3,N)
x1 = DataGeneratingProcess(0.1,0.75,-0.2,0.3,N)
x2 = DataGeneratingProcess(0.1,0.9,-0.2,0.3,N)
x3 = DataGeneratingProcess(0.1,0.0,-0.0,0.1,N)
x4 = DataGeneratingProcess(0.1,0.9,-0.0,0.0,N)
x5 = DataGeneratingProcess(0.1,0.4,-0.5,0.0,N)

# Wrap data and compute the Mean Absolute Error
data = pd.DataFrame(np.c_[x1,x2,x3,x4,x5], columns=['M1','M2','M3','M4','M5'])
for model in ['M1','M2','M3','M4','M5']:
    data[model] = np.abs(data[model] - y)

mcs = ModelConfidenceSet(data, 0.1,3, 1000).run()
#%%
