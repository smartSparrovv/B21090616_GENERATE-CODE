import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


class CustomImageDataset(Dataset):
    def __init__(self, folder1, folder2, folder3, folder4, transform=None):
        # 1. 初始化文件夹路径
        self.folder1 = folder1
        self.folder2 = folder2

        # 2. 获取文件夹中的所有图片文件路径
        self.images = []
        self.labels = []

        # 读取第一类文件夹中的图片
        for img_name in os.listdir(folder1):
            if img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png"):
                self.images.append(os.path.join(folder1, img_name))
                self.labels.append(0)  # 第一类标签为0

        # 读取第二类文件夹中的图片
        for img_name in os.listdir(folder2):
            if img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png"):
                self.images.append(os.path.join(folder2, img_name))
                self.labels.append(1)  # 第二类标签为1

        # 读取第三类文件夹中的图片
        for img_name in os.listdir(folder3):
            if img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png"):
                self.images.append(os.path.join(folder3, img_name))
                self.labels.append(2)  # 第三类标签为2

        for img_name in os.listdir(folder4):
            if img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png"):
                self.images.append(os.path.join(folder4, img_name))
                self.labels.append(3)

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


def get_dataloader(folder0, folder1, folder2, folder3, batch_size=32):
    # 数据预处理：可以根据需要调整
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),  # 调整图像大小到32x32（与CIFAR10类似）
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 创建自定义数据集
    dataset = CustomImageDataset(folder0, folder1, folder2, folder3, transform=transform)

    # 将数据集分为训练集和测试集
    # 按照类别分别进行划分
    class_0_images = [img for img, label in zip(dataset.images, dataset.labels) if label == 0]
    class_1_images = [img for img, label in zip(dataset.images, dataset.labels) if label == 1]
    class_2_images = [img for img, label in zip(dataset.images, dataset.labels) if label == 2]
    class_3_images = [img for img, label in zip(dataset.images, dataset.labels) if label == 3]

    # 使用 train_test_split 划分训练集和测试集
    train_class_0, test_class_0 = train_test_split(class_0_images, test_size=0.2, random_state=42)
    train_class_1, test_class_1 = train_test_split(class_1_images, test_size=0.2, random_state=42)
    train_class_2, test_class_2 = train_test_split(class_2_images, test_size=0.2, random_state=42)
    train_class_3, test_class_3 = train_test_split(class_3_images, test_size=0.2, random_state=42)

    # 结合训练集和测试集
    train_images = train_class_0 + train_class_1 + train_class_2 + train_class_3
    train_labels = [0] * len(train_class_0) + [1] * len(train_class_1) + [2] * len(train_class_2) + [3] * len(
        train_class_3)

    # 训练集划分出1/8作为验证集
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=1 / 8, random_state=42, stratify=train_labels
    )

    test_images = test_class_0 + test_class_1 + test_class_2 + test_class_3
    test_labels = [0] * len(test_class_0) + [1] * len(test_class_1) + [2] * len(test_class_2) + [3] * len(test_class_3)

    # 构造训练集、验证集和测试集的数据集对象
    train_dataset = CustomImageDataset(folder0, folder1, folder2, folder3, transform=transform)
    train_dataset.images = train_images
    train_dataset.labels = train_labels

    val_dataset = CustomImageDataset(folder0, folder1, folder2, folder3, transform=transform)
    val_dataset.images = val_images
    val_dataset.labels = val_labels

    test_dataset = CustomImageDataset(folder0, folder1, folder2, folder3, transform=transform)
    test_dataset.images = test_images
    test_dataset.labels = test_labels

    # 查看数据集长度
    print("训练数据集长度：{}".format(len(train_dataset)))
    print("验证数据集长度：{}".format(len(val_dataset)))
    print("测试数据集长度：{}".format(len(test_dataset)))

    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, len(train_dataset), len(val_dataset), len(test_dataset)
