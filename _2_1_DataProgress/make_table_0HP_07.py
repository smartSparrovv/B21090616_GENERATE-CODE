import pandas as pd
import scipy.io
from sympy.codegen import Print

filepath_normal_0 = f"./datas/0HP_07/normal_0.mat"
filepath_B007_0 = f"./datas/0HP_07/B007_0.mat"
filepath_IR007_0 = f"./datas/0HP_07/IR007_0.mat"
filepath_OR007_0 = f"./datas/0HP_07/OR007at6_0.mat"

data_normal = scipy.io.loadmat(filepath_normal_0)
data_B007_0 = scipy.io.loadmat(filepath_B007_0)
data_IR007_0 = scipy.io.loadmat(filepath_IR007_0)
data_OR007_0 = scipy.io.loadmat(filepath_OR007_0)

print(data_normal.keys())
print(data_B007_0.keys())
print(data_IR007_0.keys())
print(data_OR007_0.keys())

print(data_normal['X097_DE_time'].shape)
print(data_B007_0['X122_DE_time'].shape)
print(data_IR007_0['X109_DE_time'].shape)
print(data_OR007_0['X135_DE_time'].shape)

data_42k_0HP_07 = pd.DataFrame()
data_42k_0HP_07["de_normal_0"] = data_normal['X097_DE_time'].reshape(-1)[0:243538]
data_42k_0HP_07["de_B007_0"] = data_B007_0['X122_DE_time'].reshape(-1)[0:243538]
data_42k_0HP_07["de_IR007_0"] = data_IR007_0['X109_DE_time'].reshape(-1)[0:243538]
data_42k_0HP_07["de_OR007_0"] = data_OR007_0['X135_DE_time'].reshape(-1)[0:243538]
print(data_42k_0HP_07.shape)

data_42k_0HP_07.set_index('de_normal_0', inplace=True)
data_42k_0HP_07.to_csv('DE_0HP_07.csv')
