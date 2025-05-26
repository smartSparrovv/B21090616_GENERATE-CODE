import time

import numpy as np
import torch
from torch import optim
from data_loader import get_dataloader
from AlexNet import AlexNet
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = 0
epochs = 6
batch_size = 64
lr = 0.001
save_threshold = epochs - 5
source_folder0 = "./LEN_WT_COLOR/20Hz/B20"
source_folder1 = "./LEN_WT_COLOR/20Hz/M20"
source_folder2 = "./LEN_WT_COLOR/20Hz/N20"
source_folder3 = "./LEN_WT_COLOR/20Hz/R20"


source_val_accuracy_list = []

transform_toTensor = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])


source_train_dataloader, source_val_dataloader, source_test_dataloader, source_train_length, source_val_length, source_test_length = get_dataloader(
    folder0=source_folder0, folder1=source_folder1,
    folder2=source_folder2, folder3=source_folder3,
    batch_size=batch_size)

print(len(source_train_dataloader))

model = AlexNet()
model = model.to(device)
for param in model.parameters():
    param.requires_grad = True
loss_classification = torch.nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)


len_dataloader = len(source_train_dataloader)  # 确定每个epoch的数据加载次数

for epoch in range(epochs):
    print(f"epoch {epoch} is starting:")
    total_train_loss = 0
    step = 0
    for source_datas in source_train_dataloader:
        source_images, source_labels = source_datas
        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        source_features, source_classify_labels = model(source_images, mode='source')

        # loss += coral
        source_classification_loss = loss_classification(source_classify_labels, source_labels)  # 计算对源域样本分类的损失（错误率）
        total_train_loss += source_classification_loss
        optimizer.zero_grad()
        source_classification_loss.backward()
        optimizer.step()

    print(f"epoch {epoch} loss = {total_train_loss}")

    # 测试
    total_source_val_accuracy = 0
    with torch.no_grad():
        for source_val_datas in source_val_dataloader:
            source_val_images, source_val_labels = source_val_datas
            source_val_images = source_val_images.to(device)
            source_val_labels = source_val_labels.to(device)
            source_val_features, source_val_classify_labels = model(source_val_images, mode='source')
            accuracy = (source_val_classify_labels.argmax(1) == source_val_labels).sum()
            total_source_val_accuracy += accuracy

        print(f"epoch_{epoch}_source_accuracy: {total_source_val_accuracy / source_val_length:.4f}")
        source_val_accuracy_list.append((total_source_val_accuracy / source_val_length).cpu().numpy())

    # 保存模型
     if epoch >= save_threshold:
         torch.save(model.state_dict(), f"./train_model_20TX/model_only_20/model_only_20_{epoch}.pth")
         print(f"model_{epoch}.pth saved" + time.strftime("%Y%m%d-%H%M%S"))

index = 0
with open('accuracy/accuracy_only_20.txt', 'w') as file:
    for acc in source_val_accuracy_list:
        file.write('epoch' + str(index) + ' ' + str(acc) + '\n') 
        index += 1

