import torch



def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # 默认源域、目标域一批次样本数相同
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # 每行总和 - 对角线元素
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # 每行总和 - 对角线元素
    K_XY_sums_0 = K_XY.sum(dim=0)           # 每列总和

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2

def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))  # 输入是(batch_size, d)形状。里面是batch_size行，每行是d维的特征向量，是行向量
    m = X.size(0)  # m是样本数量，即batch_size

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^Ty
    # ||x||^2 和 ||y||^2 是向量的平方范数，可以从 Gram 矩阵的对角线提取；
    # x^Ty 就是 ZZT[i, j] 中的值。
    Z = torch.cat((X, Y), 0)  # 按行拼接，拼接成(2*batch_size, d)形状
    ZZT = torch.mm(Z, Z.t())  # 对张量Z和它的转置Z.t()做矩阵乘法，结果是一个Gram矩阵，即样本之间的线性内积相似度矩阵。形状是(2*batch_size, 2*batch_size)
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)  # 提取Gram矩阵ZZT的对角线元素（每个样本的平方范数），并添加一个维度，形状是(2*batch_size,1)的列向量
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)  # K会自动转换为与矩阵同形状的张量

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)  # 返回K_XX,K_XY,K_YY

def mix_rbf_mmd2(X, Y, sigma_list, biased=False):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)
