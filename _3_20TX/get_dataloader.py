import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GetDataset(Dataset):
    def __init__(self, folder1, index,  transform=None):
        # 1. 初始化文件夹路径
        self.folder1 = folder1

        # 2. 获取文件夹中的所有图片文件路径
        self.images = []
        self.labels = []

        # 读取第一类文件夹中的图片
        for img_name in os.listdir(folder1):
            if img_name.endswith(".jpg",) or img_name.endswith(".jpeg",) or img_name.endswith(".png",):
                self.images.append(os.path.join(folder1, img_name))
                self.labels.append(index)  # 第一类标签为0

        # 3. 可以选择添加图像预处理方法
        self.transform = transform

    def __len__(self):
        # 返回数据集的大小
        return len(self.images)

    def __getitem__(self, idx):
        # 获取指定索引的图片和标签
        img_path = self.images[idx]
        label = self.labels[idx]
        # 打开图片
        image = Image.open(img_path).convert('RGB')
        # 如果有预处理方法，则进行转换
        if self.transform:
            image = self.transform(image)

        return image, label


# 数据预处理：可以根据需要调整
transform_toTensor = transforms.Compose([
    # transforms.Resize((32, 32)),  # 调整图像大小到32x32（与CIFAR10类似）
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# folder = "../_2_TS2I01/TS2I02_de_ball_07_48k_0HP"
#
# test_dataset = GetDataset(folder, index=0, transform=transform_toTensor)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


