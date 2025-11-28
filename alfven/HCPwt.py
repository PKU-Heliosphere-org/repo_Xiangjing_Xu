import scipy.fft
import numpy as np


def morlet1DT(new_k, new_w, k_0=6, w_0=6):
    # Computing the wavelet in frequency domain
    out = np.exp(- ((new_k - k_0) ** 2 + (new_w - w_0) ** 2) / 2.0)
    return out


def obtain_k_or_w_arr(data_len, freq_sample):
    data_len_2 = int(np.floor((data_len - 1) / 2))
    array_tmp = np.concatenate((np.arange(0, data_len_2 + 1, 1), np.arange(data_len_2 - data_len + 1, 0, 1)))
    k_or_w_arr = 2 * np.pi / data_len * array_tmp * freq_sample
    return k_or_w_arr


def obtain_new_k_w(k, w, a_s, a_t, c_vel):
    ncvel = np.abs(c_vel) ** 0.5
    new_k = a_s * k * ncvel
    new_w = a_t * w / ncvel * np.sign(c_vel)
    return new_k, new_w


def cwt1DT(fsig,
           freq_sample_spatial,
           freq_sample_time,
           lambda_arr, period_arr,
           c_vel_arr, normalization='L2'):
    """
    Purpose:
    Input:
        fsig: the 2D fft result of the 1D+1T signal by invoking the function of scipy.fft.fftn;
        freq_sample_spatial: sampling frequency in spatial domain, which is defined as 1/dx;
        freq_sample_time: sampling frequency in time domain, which is defined as 1/dt;
        lambda_arr: the 1d array of wavelength to be analyzed with wavelet transform;
        period_arr: the 1d array of period to be analyzed with wavelet transform;
        c_vel_arr: the 1d array of velocity scaling coefficients to be analyzed with wavelet transform;
        normalization: 'L1' and 'L2'. default: 'L2'
    Output:
        out_data: the 5d array of complexed wavelet coefficient, which involves the dimensions of '1D + 1T + lambda + period + c_vel'
    Record:
        first written by Chuanpeng Hou in 2023-03-14;
    """
    # fsig: fft of signal # 傅立叶变换结果：前一半系数对应正频靠近坐标原点; 后一半对应负频，顺序序频率逐渐降低，如[1,2,3,-3,-2,-1]Hz.

    # 检查傅立叶变换后行列关系：是否和原数据保持一致？ Done! 保持一致
    data_len_x = np.shape(fsig)[0]
    data_len_t = np.shape(fsig)[1]

    # 检查傅立叶变换后系数和频率的对应关系：是否需要平移频率中心？ Done! 不需要
    k_arr = obtain_k_or_w_arr(data_len_x, freq_sample_spatial)  ##是否需要指定采样频率？？需要
    w_arr = obtain_k_or_w_arr(data_len_t, freq_sample_time)
    k_2D_grid, w_2D_grid = np.meshgrid(k_arr, w_arr, indexing='ij')

    # 创建一个空的复数矩阵
    out_data = np.zeros(shape=(data_len_x, data_len_t,
                               len(lambda_arr), len(period_arr),
                               len(c_vel_arr)),
                        dtype=complex)
    # 循环每个参数
    k_0 = 6
    w_0 = 6
    epsilon = 1

    for lambda_index in range(len(lambda_arr)):
        for period_index in range(len(period_arr)):
            for c_vel_index in range(len(c_vel_arr)):
                a_s = lambda_arr[lambda_index] / 2.0 / np.pi * k_0
                a_t = period_arr[period_index] / 2.0 / np.pi * w_0
                c_vel = c_vel_arr[c_vel_index]
                # k and w come from meshgrid, which should be 2D.
                # other four parameters are a single scalar. we need four loops.
                new_k, new_w = obtain_new_k_w(k_2D_grid, w_2D_grid, a_s, a_t, c_vel)
                # Call of the wavelet function.
                mask = morlet1DT(new_k, new_w, k_0=k_0, w_0=w_0)
                if normalization == 'L1':
                    out_data[:, :, lambda_index, period_index, c_vel_index] = scipy.fft.ifftn(fsig * np.conj(mask),
                                                                                              axes=(0, 1))  # 检查是否是矩阵元素相乘
                if normalization == 'L2':
                    out_data[:, :, lambda_index, period_index, c_vel_index] = np.abs(a_s) ** (0.5) * np.abs(a_t) ** (
                        0.5) * scipy.fft.ifftn(fsig * np.conj(mask), axes=(0, 1))  # 检查是否是矩阵元素相乘
    return out_data