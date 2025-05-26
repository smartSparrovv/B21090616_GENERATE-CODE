import pandas as pd
from PIL import Image

from _1_GAF02 import *


data_table = pd.read_csv(f"../_2_DataProgress/datas/tables/table_25Hz.csv")
data_type = ['B25', 'M25', 'N25', 'R25']
type_index = 0

for type_name in data_type:
    print(f"now slice {data_type[type_index]}")
    start_index = 0
    window_size = 224 * 4 * 8
    step_size = 16
    for image_num in range(2400):  # 对每一个状态制作样本的个数
        # print(index*100, index*100+100)
        data_slice = data_table.iloc[start_index:start_index + window_size, type_index:type_index + 1]
        print(f"{start_index}-{start_index + window_size}")
        print(image_num)
        start_index += step_size
        signal_ = data_slice.values.flatten()
        signal = signal_.reshape(1, window_size)
        # 参数
        image_size = 224  # GAF 图像的大小
        method = "summation"  # GASF 方法
        sample_range = (-1, 1)  # 标准化范围

        # 生成 GAF 图像
        gaf_images = gramian_angular_field(signal, image_size, method, sample_range)
        # 去掉第一个维度，使其变为 (32, 32)
        gaf_images = gaf_images.squeeze(0)
        # 将 ndarray 转换为 8-bit 图像（需要将数据标准化到 0-255 范围内）
        gaf_images = (gaf_images - gaf_images.min()) / (gaf_images.max() - gaf_images.min()) * 255
        gaf_images = gaf_images.astype(np.uint8)

        # 使用 PIL 将 ndarray 转换为图像
        image = Image.fromarray(gaf_images)
        # 保存为 .jpg 格式
        jpg_filename = f"./LEN1792_WT/25Hz/{type_name}/{type_name}_{image_num + 1}.jpg"
        image.save(jpg_filename)
    type_index += 1

