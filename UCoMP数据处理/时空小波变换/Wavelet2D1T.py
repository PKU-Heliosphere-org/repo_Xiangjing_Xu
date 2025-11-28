import scipy.fft
import numpy as np
import cupy as cp
import cupyx.scipy.fft
from scipy.io import savemat


def morlet2DT(new_kx, new_ky, new_w, k_0=-6, w_0=6, epsilon=1):
    # Computing the wavelet in frequency domain
    new_ky_square = new_ky ** 2
    first_item = cp.exp(-1.0 / 2 * ((epsilon ** (-0.5) * new_kx - k_0) ** 2 + new_ky_square))
    second_item = cp.exp(-1.0 / 2 * ((epsilon ** (-0.5) * new_kx) ** 2 + new_ky_square + k_0 ** 2))
    third_item = cp.exp(-1.0 / 2 * (new_w - w_0) ** 2)
    fourth_item = cp.exp(-1.0 / 2 * (new_w ** 2 + w_0 ** 2))
    out = (first_item) * (third_item)
    return out


def mexcian(kx, ky, w, k_0=-6, w_0=6, epsilon=1):
    out = - (2 * np.pi) * (kx ** 2 + ky ** 2) ** (2 / 2) * np.exp(- ((kx) ** 2 + (ky) ** 2) / 2)
    return out


def obtain_k_or_w_arr(data_len, freq_sample):
    data_len_2 = int(np.floor((data_len - 1) / 2))
    array_tmp = np.concatenate((np.arange(0, data_len_2 + 1, 1), np.arange(data_len_2 - data_len + 1, 0, 1)))
    k_or_w_arr = 2 * np.pi / data_len * array_tmp * freq_sample
    return k_or_w_arr


def expand_dims_and_tile(init_array, len_new_axis):
    n_dim = init_array.ndim
    # axis=cp.array([],dtype=int)
    for i in range(len(len_new_axis)):
        axis = n_dim + i
        init_array = cp.expand_dims(init_array, axis=axis)
    # len_new_axis_append_one = cp.append(cp.ones(n_dim, dtype=int),len_new_axis)

    #     high_dim_array = cp.tile(init_array_expand_dims, tuple(len_new_axis_append_one))
    return init_array


def obtain_new_k_w(kx, ky, w, a_s, a_t, c_vel, rot_angle):
    rot_angle_rad = rot_angle * cp.pi / 180
    temp = a_s * c_vel ** (1.0 / 3)
    cos_rot_angle_rad = cp.cos(-rot_angle_rad)
    sin_rot_angle_rad = cp.sin(-rot_angle_rad)
    new_kx = temp * (cos_rot_angle_rad * kx - sin_rot_angle_rad * ky)  # 检查是否是矩阵元素相乘
    new_ky = temp * (sin_rot_angle_rad * kx + cos_rot_angle_rad * ky)
    new_w = a_t * abs(c_vel) ** (-2.0 / 3) * cp.sign(c_vel) * w
    return new_kx, new_ky, new_w


def cwt2DT(fsig,
           freq_sample_spatial_x,
           freq_sample_spatial_y,
           freq_sample_time,
           lambda_arr, period_arr,
           c_vel_arr, rot_angle_arr):
    """
    Purpose:
    Input:
        fsig: the 3D fft result of the 2D+1T signal by invoking the function of scipy.fft.fftn;
        freq_sample_spatial_x/y: sampling frequency in spatial domain, which is defined as 1/dx and 1/dy;
        freq_sample_time: sampling frequency in time domain, which is defined as 1/dt;
        #a_s_arr: the 1d array of wavelength to be analyzed with wavelet transform, a_s_arr may be defined as 'np.linspace(?,?,?)*dx';
        #a_t_arr: the 1d array of period to be analyzed with wavelet transform, a_t_arr may be defined as 'np.linspace(?,?,?)*dt';
        lambda_arr: the 1d array of wavelength to be analyzed with wavelet transform;
        period_arr: the 1d array of period to be analyzed with wavelet transform;
        c_vel_arr: the 1d array of velocity scaling coefficients to be analyzed with wavelet transform;
        rot_angle_arr: the 1d array of rotational angle to be analyzed with wavelet transform;
    Output:
        out_data: the 7d array of complexed wavelet coefficient, which involves the dimensions of '2D + T, a_s + a_t + c_vel + rot_angle'
    Record:
        first written by Chuanpeng Hou in 2023-01;
        modified by Jiansen He on 2023-02-15
    """
    # fsig: fft of signal # 傅立叶变换结果：前一半系数对应正频靠近坐标原点; 后一半对应负频，顺序序频率逐渐降低，如[1,2,3,-3,-2,-1]Hz.
    # 将fsig拷贝到GPU
    fsig_gpu = cp.asarray(fsig)
    # 检查傅立叶变换后行列关系：是否和原数据保持一致？ Done! 保持一致
    data_len_x = np.shape(fsig)[0]
    data_len_y = np.shape(fsig)[1]
    data_len_time = np.shape(fsig)[2]
    # 检查傅立叶变换后系数和频率的对应关系：是否需要平移频率中心？ Done! 不需要

    kx_arr_cpu = obtain_k_or_w_arr(data_len_x, freq_sample_spatial_x)  ##是否需要指定采样频率？？需要
    ky_arr_cpu = obtain_k_or_w_arr(data_len_y, freq_sample_spatial_y)
    w_arr_cpu = obtain_k_or_w_arr(data_len_time, freq_sample_time)
    # 拷贝到GPU
    kx_arr_gpu = cp.asarray(kx_arr_cpu)
    ky_arr_gpu = cp.asarray(ky_arr_cpu)
    w_arr_gpu = cp.asarray(w_arr_cpu)

    kx_3D_grid_gpu, ky_3D_grid_gpu, w_3D_grid_gpu = cp.meshgrid(kx_arr_gpu, ky_arr_gpu, w_arr_gpu, indexing='ij')

    # 循环每个参数
    k_0 = -6
    w_0 = 6
    epsilon = 1

    a_s_arr = lambda_arr / 2.0 / np.pi * abs(k_0)
    a_t_arr = period_arr / 2.0 / np.pi * abs(w_0)
    a_s_arr_gpu = cp.asarray(a_s_arr)
    a_t_arr_gpu = cp.asarray(a_t_arr)
    c_vel_arr_gpu = cp.asarray(c_vel_arr)
    rot_angle_arr_gpu = cp.asarray(rot_angle_arr)
    a_s_grid_gpu, a_t_grid_gpu, c_vel_grid_gpu, rot_angle_grid_gpu = cp.meshgrid(a_s_arr_gpu, a_t_arr_gpu, \
                                                                                 c_vel_arr_gpu, rot_angle_arr_gpu, \
                                                                                 indexing='ij')

    len_new_axis = [len(kx_arr_gpu), len(ky_arr_gpu), len(w_arr_gpu)]
    a_s_grid_gpu_expand_dims = expand_dims_and_tile(a_s_grid_gpu, len_new_axis)
    a_t_grid_gpu_expand_dims = expand_dims_and_tile(a_t_grid_gpu, len_new_axis)
    c_vel_grid_gpu_expand_dims = expand_dims_and_tile(c_vel_grid_gpu, len_new_axis)
    rot_angle_grid_gpu_expand_dims = expand_dims_and_tile(rot_angle_grid_gpu, len_new_axis)

    new_kx_gpu, new_ky_gpu, new_w_gpu = obtain_new_k_w(kx_3D_grid_gpu, ky_3D_grid_gpu, w_3D_grid_gpu,
                                                       a_s_grid_gpu_expand_dims,
                                                       a_t_grid_gpu_expand_dims,
                                                       c_vel_grid_gpu_expand_dims,
                                                       rot_angle_grid_gpu_expand_dims)
    # Call of the wavelet function.
    mask_gpu = morlet2DT(new_kx_gpu, new_ky_gpu, new_w_gpu, k_0=k_0, w_0=w_0, epsilon=epsilon)
    out_data_gpu = cp.abs(a_s_grid_gpu_expand_dims) * cp.abs(a_t_grid_gpu_expand_dims) ** (0.5) * cupyx.scipy.fft.ifftn(
        fsig_gpu * cp.conj(mask_gpu), axes=(4, 5, 6))
    out_data_gpu = cp.transpose(out_data_gpu, (4, 5, 6, 0, 1, 2, 3))
    # 将结果拷贝回主机cpu
    out_data = cp.asnumpy(out_data_gpu)

    del out_data_gpu, mask_gpu, new_kx_gpu, new_ky_gpu, new_w_gpu, a_s_grid_gpu_expand_dims, a_t_grid_gpu_expand_dims, c_vel_grid_gpu_expand_dims, rot_angle_grid_gpu_expand_dims

    return out_data