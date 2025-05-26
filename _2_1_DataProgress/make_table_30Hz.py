import pandas as pd
import scipy.io
from sympy.codegen import Print

filepath_B30 = f"./datas/30Hz/B1_30.MAT"
filepath_M30 = f"./datas/30Hz/M1_30.MAT"
filepath_N30 = f"./datas/30Hz/N1_30.MAT"
filepath_R30 = f"./datas/30Hz/R1_30.MAT"

data_B30 = scipy.io.loadmat(filepath_B30)
data_M30 = scipy.io.loadmat(filepath_M30)
data_N30 = scipy.io.loadmat(filepath_N30)
data_R30 = scipy.io.loadmat(filepath_R30)

print(data_B30.keys())
print(data_M30.keys())
print(data_N30.keys())
print(data_R30.keys())

print(data_B30['Data'].shape)
print(data_M30['Data'].shape)
print(data_N30['Data'].shape)
print(data_R30['Data'].shape)

data_B30_x = data_B30['Data'][0]

table_30Hz = pd.DataFrame()
table_30Hz["B30"] = data_B30['Data'][:, 0][0:300000]
table_30Hz["M30"] = data_M30['Data'][:, 0][0:300000]
table_30Hz["N30"] = data_N30['Data'][:, 0][0:300000]
table_30Hz["R30"] = data_R30['Data'][:, 0][0:300000]

print(table_30Hz.shape)

table_30Hz.set_index('B30', inplace=True)
table_30Hz.to_csv('table_30Hz.csv')
