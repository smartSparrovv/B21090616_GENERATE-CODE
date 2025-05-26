import time
import numpy as np
import torch
from torch import optim
from data_loader import get_dataloader
from AlexNet import AlexNet
import MMD_loss
from torchvision import transforms

SIGMA = [1, 2, 4]


def mmd_loss(x_src, x_tar):
    return MMD_loss.mix_rbf_mmd2(x_src, x_tar, SIGMA)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gamma = 2000000
# gamma = 1
loss = 0
epochs = 3
batch_size = 64
momentum = 0.9
lr = 0.001
weight_decay = 5e-4
save_threshold = epochs - 6
source_folder0 = "./LEN_WT_COLOR/20Hz/B20"
source_folder1 = "./LEN_WT_COLOR/20Hz/M20"
source_folder2 = "./LEN_WT_COLOR/20Hz/N20"
source_folder3 = "./LEN_WT_COLOR/20Hz/R20"
#
target_folder0 = "../MIX_WT_COLOR/25Hz/B25"
target_folder1 = "../MIX_WT_COLOR/25Hz/M25"
target_folder2 = "../MIX_WT_COLOR/25Hz/N25"
target_folder3 = "../MIX_WT_COLOR/25Hz/R25"

source_test_accuracy_list = []
target_test_accuracy_list = []
target_val_accuracy_list = []

source_train_dataloader, source_val_dataloader, source_test_dataloader, source_train_length, source_val_length, source_test_length = get_dataloader(
    folder0=source_folder0, folder1=source_folder1,
    folder2=source_folder2, folder3=source_folder3,
    batch_size=batch_size)

target_train_dataloader, target_val_dataloader, target_test_dataloader, target_train_length, target_val_length, target_test_length = get_dataloader(
    folder0=target_folder0, folder1=target_folder1,
    folder2=target_folder2, folder3=target_folder3,
    batch_size=batch_size)
print(len(source_train_dataloader), len(target_train_dataloader))

model = AlexNet()
model = model.to(device)
model.load_state_dict(torch.load(f'./train_model_20TX/model_only_20/model_only_20_196_as0.pth'))
for param in model.parameters():
    param.requires_grad = True
loss_classification = torch.nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)

len_dataloader = min(len(source_train_dataloader), len(target_train_dataloader))  # 确定每个epoch的数据加载次数

for epoch in range(epochs):
    print(f"epoch {epoch} is starting:")

    total_train_loss = 0
    step = 0
    for source_datas, target_datas in zip(source_train_dataloader, target_train_dataloader):
        step += 1

        target_images, target_labels = target_datas
        target_images = target_images.to(device)
        target_labels = target_labels.to(device)

        target_features, target_classify_labels = model(target_images, mode='source')

        source_images, source_labels = source_datas
        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        source_features, source_classify_labels = model(source_images, mode='source')

        mmd = 0  # 清零前次MMD
        mmd = mmd_loss(source_features, target_features)
        source_classification_loss = loss_classification(source_classify_labels, source_labels)  # 计算对源域样本分类的损失（错误率）
        target_classification_loss = loss_classification(target_classify_labels, target_labels)
        # train_loss = 0.1 * mmd + 0.8 * source_classification_loss + 1.5 * target_classification_loss
        train_loss = 1.5 * mmd + 0.8 * target_classification_loss + 0.6 * source_classification_loss
        total_train_loss += train_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    print(f"epoch {epoch} loss = {total_train_loss}")

    # ===== 源域验证集验证 =====
    total_source_val_accuracy = 0
    with torch.no_grad():
        for source_val_datas in source_val_dataloader:
            source_val_images, source_val_labels = source_val_datas
            source_val_images = source_val_images.to(device)
            source_val_labels = source_val_labels.to(device)
            _, source_val_classify_labels = model(source_val_images, mode='source')
            accuracy = (source_val_classify_labels.argmax(1) == source_val_labels).sum()
            total_source_val_accuracy += accuracy

    val_source_acc = total_source_val_accuracy / source_val_length
    print(f"epoch_{epoch}_val_source_accuracy: {val_source_acc:.4f}")

    # ===== 目标域验证集验证 =====
    total_target_val_accuracy = 0
    with torch.no_grad():
        for target_val_datas in target_val_dataloader:
            target_val_images, target_val_labels = target_val_datas
            target_val_images = target_val_images.to(device)
            target_val_labels = target_val_labels.to(device)
            _, target_val_classify_labels = model(target_val_images, mode='source')
            accuracy = (target_val_classify_labels.argmax(1) == target_val_labels).sum()
            total_target_val_accuracy += accuracy

    val_target_acc = total_target_val_accuracy / target_val_length
    print(f"epoch_{epoch}_val_target_accuracy: {val_target_acc:.4f}")
    print(f"epoch_{epoch}_target_accuracy: {total_target_val_accuracy / target_val_length:.4f}")
    target_val_accuracy_list.append((total_target_val_accuracy / target_val_length).cpu().numpy())


    # 保存模型
    if epoch >= save_threshold:
        torch.save(model.state_dict(), f"./train_model_20TX/model_20T25/model_20T25_MIX_{epoch}.pth")
        print(f"model_{epoch}.pth saved" + time.strftime("%Y%m%d-%H%M%S"))

# 保存每轮训练后对验证数据诊断准确率
index = 0
with open('accuracy/accuracy_20T25/accuracy_20T25_on_target_epoch.txt', 'w') as file:
    for acc in target_val_accuracy_list:
        file.write('epoch' + str(index) + ' ' + str(acc) + '\n')
        index += 1

index = 0
with open('accuracy/accuracy_20T25/accuracy_20T25_on_target.txt', 'w') as file:
    for acc in target_val_accuracy_list:
        file.write(str(acc) + '\n')
        index += 1

