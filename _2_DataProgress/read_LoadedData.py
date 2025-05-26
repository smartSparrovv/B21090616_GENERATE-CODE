import torch
from torch.utils.data import Dataset, DataLoader
from make_LoadedData import SignalDataset

# 自定义数据集类
# class SignalDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]


# 重新加载存储的数据
normal_data = torch.load(f'../_1_DataProgress/datas/loaded/de_normal_42k_0-100000.pt')
b07_data = torch.load(f"../_1_DataProgress/datas/loaded/de_B007_42k_0-100000.pt")
# 创建自定义数据集
dataset = SignalDataset(loaded_data)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 测试 DataLoader
for batch in dataloader:
    print(batch.shape)  # 打印每个批次的形状
