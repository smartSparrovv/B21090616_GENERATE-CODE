import argparse
import os
import numpy as np
from torchvision import transforms

from torchvision.utils import save_image
import torch.nn as nn
import torch

from AlexNet import AlexNet

# 需要修改的有 type，label， generator.load_state_dict，img_path
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=60, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=3, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
# print(opt)
img_shape = (opt.channels, opt.img_size, opt.img_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置参数
type = 'fB20'
output_dir = "./FAKE_LEN_WT_COLOR/20Hz/" + type
# os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

num_images = 1000
batch_size = 20  # 一次生成的图片数量，可调整
latent_dim = opt.latent_dim  # 潜在空间维度
label = 0

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


generator = Generator()
generator.load_state_dict(torch.load('./model_weight/model_weight_20Hz/MAKE_B20/gen_B20_599.pth'))
generator.to(device)
generator.eval()
# 开始生成图片
saved_count = 0
gen_count = 0
while saved_count < num_images:
    # 随机噪声
    z = torch.randn(batch_size, latent_dim).to(device)

    # 生成图片
    with torch.no_grad():
        gen_images = generator(z)
    gen_images = (gen_images + 1) / 2  # generator生成的图像是标准化图像，要反标准化
    for i in range(batch_size):
        if saved_count >= num_images:
            break
        
        img_path = os.path.join(output_dir, type + f"_{saved_count}.jpg")
        save_image(gen_images[i], img_path)
        saved_count += 1

    gen_count += batch_size

