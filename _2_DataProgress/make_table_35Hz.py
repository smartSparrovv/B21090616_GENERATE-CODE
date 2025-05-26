import pandas as pd
import scipy.io
from sympy.codegen import Print

filepath_B35 = f"./datas/35Hz/B1_35.MAT"
filepath_M35 = f"./datas/35Hz/M1_35.MAT"
filepath_N35 = f"./datas/35Hz/N1_35.MAT"
filepath_R35 = f"./datas/35Hz/R1_35.MAT"

data_B35 = scipy.io.loadmat(filepath_B35)
data_M35 = scipy.io.loadmat(filepath_M35)
data_N35 = scipy.io.loadmat(filepath_N35)
data_R35 = scipy.io.loadmat(filepath_R35)

print(data_B35.keys())
print(data_M35.keys())
print(data_N35.keys())
print(data_R35.keys())

print(data_B35['Data'].shape)
print(data_M35['Data'].shape)
print(data_N35['Data'].shape)
print(data_R35['Data'].shape)

data_B35_x = data_B35['Data'][0]

table_35Hz = pd.DataFrame()
table_35Hz["B35"] = data_B35['Data'][:, 0][0:350000]
table_35Hz["M35"] = data_M35['Data'][:, 0][0:350000]
table_35Hz["N35"] = data_N35['Data'][:, 0][0:350000]
table_35Hz["R35"] = data_R35['Data'][:, 0][0:350000]

print(table_35Hz.shape)

table_35Hz.set_index('B35', inplace=True)
table_35Hz.to_csv('table_35Hz.csv')
