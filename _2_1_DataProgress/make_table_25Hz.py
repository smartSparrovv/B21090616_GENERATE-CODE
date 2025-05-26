import pandas as pd
import scipy.io
from sympy.codegen import Print

filepath_B25 = f"./datas/25Hz/B1_25.MAT"
filepath_M25 = f"./datas/25Hz/M1_25.MAT"
filepath_N25 = f"./datas/25Hz/N1_25.MAT"
filepath_R25 = f"./datas/25Hz/R1_25.MAT"

data_B25 = scipy.io.loadmat(filepath_B25)
data_M25 = scipy.io.loadmat(filepath_M25)
data_N25 = scipy.io.loadmat(filepath_N25)
data_R25 = scipy.io.loadmat(filepath_R25)

print(data_B25.keys())
print(data_M25.keys())
print(data_N25.keys())
print(data_R25.keys())

print(data_B25['Data'].shape)
print(data_M25['Data'].shape)
print(data_N25['Data'].shape)
print(data_R25['Data'].shape)

data_B25_x = data_B25['Data'][0]

table_25Hz = pd.DataFrame()
table_25Hz["B25"] = data_B25['Data'][:, 0][0:300000]
table_25Hz["M25"] = data_M25['Data'][:, 0][0:300000]
table_25Hz["N25"] = data_N25['Data'][:, 0][0:300000]
table_25Hz["R25"] = data_R25['Data'][:, 0][0:300000]

print(table_25Hz.shape)

table_25Hz.set_index('B25', inplace=True)
table_25Hz.to_csv('table_25Hz.csv')
