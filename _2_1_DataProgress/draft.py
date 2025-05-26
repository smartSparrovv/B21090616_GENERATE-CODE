import scipy.io
import pandas as pd

# 文件路径
filepath_B20 = f"./datas/20Hz/B1_20.MAT"
filepath_M20 = f"./datas/20Hz/M1_20.MAT"
filepath_N20 = f"./datas/20Hz/N1_20.MAT"
filepath_R20 = f"./datas/20Hz/R1_20.MAT"

# 加载数据
data_B20 = scipy.io.loadmat(filepath_B20)
data_M20 = scipy.io.loadmat(filepath_M20)
data_N20 = scipy.io.loadmat(filepath_N20)
data_R20 = scipy.io.loadmat(filepath_R20)

# 取出Data字段的第一列
col_B20 = data_B20['Data'][:, 0]
col_M20 = data_M20['Data'][:, 0]
col_N20 = data_N20['Data'][:, 0]
col_R20 = data_R20['Data'][:, 0]

# 构造表格（如果你只想要前100个数据点，也可以加 [0:100]）
table_20Hz = pd.DataFrame({
    "B20": col_B20[0:100],
    "M20": col_M20[0:100],
    "N20": col_N20[0:100],
    "R20": col_R20[0:100]
})

# 可选：设置索引（不强制）
# table_20Hz.set_index('B20', inplace=True)  # 不推荐这样设置索引，除非你确实想用B20的值作为索引

# 保存为CSV
table_20Hz.to_csv('table_20Hz.csv', index=False)

print(table_20Hz.shape)
