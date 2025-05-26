import os
import cv2

# 源文件夹（灰度图像所在位置）
folder = 'R35'
source_folder = "./LEN_WT/35Hz/"+folder
# 目标文件夹（存放伪彩色图像）
target_folder = "./LEN_WT_COLOR/35Hz/"+folder
# 确保目标目录存在
os.makedirs(target_folder, exist_ok=True)

# 读取源文件夹内所有图片
for filename in os.listdir(source_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):  # 只处理图片文件
        img_path = os.path.join(source_folder, filename)

        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        color_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_TWILIGHT)  # 你可以换成其他伪彩色映射

        target_path = os.path.join(target_folder, filename)

        cv2.imwrite(target_path, color_img)

print("伪彩色转换完成！")
