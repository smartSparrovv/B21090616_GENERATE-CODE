import pandas as pd
import scipy.io
from sympy.codegen import Print

filepath_B20 = f"./datas/20Hz/B1_20.MAT"
filepath_M20 = f"./datas/20Hz/M1_20.MAT"
filepath_N20 = f"./datas/20Hz/N1_20.MAT"
filepath_R20 = f"./datas/20Hz/R1_20.MAT"

data_B20 = scipy.io.loadmat(filepath_B20)
data_M20 = scipy.io.loadmat(filepath_M20)
data_N20 = scipy.io.loadmat(filepath_N20)
data_R20 = scipy.io.loadmat(filepath_R20)

print(data_B20.keys())
print(data_M20.keys())
print(data_N20.keys())
print(data_R20.keys())

print(data_B20['Data'].shape)
print(data_M20['Data'].shape)
print(data_N20['Data'].shape)
print(data_R20['Data'].shape)

data_B20_x = data_B20['Data'][0]

table_20Hz = pd.DataFrame()
table_20Hz["B20"] = data_B20['Data'][:, 0][0:300000]
table_20Hz["M20"] = data_M20['Data'][:, 0][0:300000]
table_20Hz["N20"] = data_N20['Data'][:, 0][0:300000]
table_20Hz["R20"] = data_R20['Data'][:, 0][0:300000]

print(table_20Hz.shape)

table_20Hz.set_index('B20', inplace=True)
table_20Hz.to_csv('table_20Hz.csv')
