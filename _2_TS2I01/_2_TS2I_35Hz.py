import pandas as pd
from PIL import Image

from _1_GAF02 import *


data_table = pd.read_csv(f"../_2_DataProgress/datas/tables/table_35Hz.csv")
data_type = ['B35', 'M35', 'N35', 'R35']
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


        gaf_images = gramian_angular_field(signal, image_size, method, sample_range)

        gaf_images = gaf_images.squeeze(0)

        gaf_images = (gaf_images - gaf_images.min()) / (gaf_images.max() - gaf_images.min()) * 255
        gaf_images = gaf_images.astype(np.uint8)


        image = Image.fromarray(gaf_images)

        jpg_filename = f"./LEN1792_WT/35Hz/{type_name}/{type_name}_{image_num + 1}.jpg"
        image.save(jpg_filename)
    type_index += 1

