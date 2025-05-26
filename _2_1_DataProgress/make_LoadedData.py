import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

data_table = pd.read_csv(f"./datas/tables/DE_48K_ball_0-1HP.csv")
length = len(data_table.index)
print(f"table length is: {length}")
de_normal_48k = data_table.iloc[0:500*200, 0:1]
signal_ = de_normal_48k.values.flatten()

# 假设信号数据已经加载为ndarray类型的 signal
# 例如：signal = np.load('path_to_your_signal.npy')

signal = signal_
# 将信号数据按每500个点切分成块
chunk_size = 100
num_chunks = len(signal) // chunk_size  # 计算可以分割成多少个块

# 按块切分数据
signal_chunks = signal[:num_chunks * chunk_size].reshape(-1, chunk_size)


# 自定义数据集类
class SignalDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)  # 转换为PyTorch张量

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # 返回索引位置的数据块


# 创建自定义数据集实例
dataset = SignalDataset(signal_chunks)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=100, shuffle=False)  # 批量大小为32

# 测试 DataLoader
for batch in dataloader:
    print(batch.shape)  # 打印每个批次的形状

# 将切分后的数据存储为 Tensor 文件
torch.save(torch.tensor(signal_chunks, dtype=torch.float32), './datas/loaded_100/de_B007_42k_0HP_0-100000.pt')

