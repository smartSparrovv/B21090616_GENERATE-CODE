import numpy as np
import matplotlib.pyplot as plt


def gramian_angular_field(X, image_size, method, sample_range):

    n_samples, n_timestamps = X.shape  # 这里n_samples=1，因为一次只输入1个序列样本
    if image_size < 1 or image_size > n_timestamps:  # image_size对应于论文里设定的组数m
        raise ValueError("'image_size' must be between 1 and n_timestamps.")

    # Step 1: 对序列分段聚合近似，image_size=224
    X_paa = piecewise_aggregate_approximation(X, image_size)

    # Step 2: Min-Max scaling
    # 对分段聚合近似后的数值归一化至[-1,1]得X~，也就等于cos(φ_i)，因为原理里是先对归一化后的序列中的数值，先求φ_i=arccos(x~i)，所以cos(φ_i)=(x~i)
    # 再求cos(φ_i+φ_j)，=cos(φ_i)*cos(φ_j)-sin(φ_i)*sin(φ_j)。
    X_cos = min_max_scaler(X_paa, sample_range)  # cos(φ_i)序列，i=0,...,m-1
    X_sin = np.sqrt(np.clip(1 - X_cos ** 2, 0, 1))  # sin(φ_i)序列，i=0,...,m-1。arccos范围是[0,pi]，所以sin(φ_i)的范围是[0,1]

    # Step 3: Generate GASF or GADF
    if method == "summation":
        return gasf(X_cos, X_sin, n_samples, image_size)
    elif method == "difference":
        return gadf(X_cos, X_sin, n_samples, image_size)
    else:
        raise ValueError("Method must be 'summation' or 'difference'.")


def gasf(X_cos, X_sin, n_samples, image_size):
    """
    Generates the Gramian Angular Summation Field (GASF).
    """
    X_gasf = np.zeros((n_samples, image_size, image_size))
    for i in range(n_samples):
        cosi = X_cos[i, :]  # 对应的余弦列向量
        sini = X_sin[i, :]  # 对应的正弦列向量
        # Generate a square matrix
        X_gasf[i] = np.outer(cosi, cosi) - np.outer(sini, sini)  # cosi*cosi的转置-sini*sini的转置，即矩阵cos(φ_i)*cos(φ_j)-矩阵sin(φ_i)*cos(φ_j)
    return X_gasf


def gadf(X_cos, X_sin, n_samples, image_size):
    """
    Generates the Gramian Angular Difference Field (GADF).
    """
    X_gadf = np.zeros((n_samples, image_size, image_size))
    for i in range(n_samples):
        cosi = X_cos[i, :]
        sini = X_sin[i, :]  # 对应的正弦列向量
        # Generate a square matrix
        X_gadf[i] = np.outer(sini, cosi) - np.outer(cosi, sini)  # 矩阵cos(φ_i)*cos(φ_j)-矩阵sin(φ_i)*cos(φ_j)
    return X_gadf

# 对分段聚合近似后的序列归一化
def min_max_scaler(X, sample_range):
    """
    Scales the input time series to the specified range.
    """
    min_val, max_val = sample_range
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    X_std = (X - X_min) / (X_max - X_min)
    return X_std * (max_val - min_val) + min_val  # 归一化过程，对应公式3-3

# 分段聚合近似
def piecewise_aggregate_approximation(X, out_size):
    """
    Applies Piecewise Aggregate Approximation (PAA) to compress the time series.
    """
    n_samples, n_timestamps = X.shape
    # out_size=image_size=224

    segment_size = n_timestamps // out_size  # 确定每组数值个数，即论文里的w，对应公式3-1
    X_paa = np.zeros((n_samples, out_size))

    for i in range(out_size):
        start_idx = i * segment_size
        end_idx = min(start_idx + segment_size, n_timestamps)
        X_paa[:, i] = X[:, start_idx:end_idx].mean(axis=1)  # 对每组数值求均值，对应公式3-2
    return X_paa


if __name__ == '__main__':
    # 示例数据
    X = np.array([[100, 202, 563, 334, 895],
                  [5, 4, 3, 2, 1]])
    # 参数
    image_size = 5  # GAF 图像的大小
    method = "summation"  # GASF 方法
    sample_range = (-1, 1)  # 标准化范围

    # 生成 GAF 图像
    gaf_images = gramian_angular_field(X, image_size, method, sample_range)

    print("GAF 图像形状：", gaf_images.shape)
    print("GAF 图像内容：\n", gaf_images)
    n_samples = gaf_images.shape[0]
    for i in range(n_samples):
        plt.figure(figsize=(5, 5))
        plt.imshow(gaf_images[i], cmap='viridis')  # 使用 'viridis' 色彩映射，也可尝试 'hot' 或 'cool'

        plt.axis('off')  # 关闭坐标轴
        # 保存为 PDF，文件名可以根据需求修改
        pdf_filename = f"./images/gaf_image_{i + 1}.jpg"
        plt.savefig(pdf_filename, format='jpg', bbox_inches='tight', pad_inches=0) # 保存时只保存有效的图片
        # plt.close()  # 关闭当前图，避免占用内存
        print(f"GAF 图像 {i + 1} 已保存为 {pdf_filename}")

        plt.axis('on')  # 开启坐标轴

        plt.colorbar(label="Intensity")  # 添加颜色条
        plt.title(f"GAF Image {i + 1}")
        plt.xlabel("Time")
        plt.ylabel("Time")
        plt.show()


