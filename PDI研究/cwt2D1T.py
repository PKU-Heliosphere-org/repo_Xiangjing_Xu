import cupy as cp
import numpy as np
from scipy.integrate import dblquad, quad
from itertools import product
from tqdm.notebook import tqdm

def c_psi(epsilon, k0, omega0, epsabs=1e-8, epsrel=1e-8):

    def integrand_space(ky, kx):
        term1 = np.exp(-0.5 * ((k0 - kx)**2 + epsilon * ky**2))
        term2 = np.exp(-0.5 * (k0**2 + kx**2 + epsilon * ky**2))
        numerator = (term1 - term2)**2
        denominator = kx**2 + ky**2
        if denominator < 1e-15:
            return 0.0
        return numerator / denominator

    I_s, _ = dblquad(integrand_space,
                         -np.inf, np.inf,
                         -np.inf, np.inf,
                         epsabs=epsabs, epsrel=epsrel)

    def integrand_time(omega):
        term1 = np.exp(-0.5 * (omega - omega0)**2)
        term2 = np.exp(-0.5 * (omega0**2 + omega**2))
        numerator = (term1 - term2)**2
        if abs(omega) < 1e-12:
            return 0.0
        return numerator / np.abs(omega)

    I_t, _ = quad(integrand_time,
                  -np.inf, np.inf,
                  epsabs=epsabs, epsrel=epsrel)

    factor = (2 * np.pi) ** 3 * epsilon # 需要进一步确认
    C_psi = factor * I_t * I_s

    return C_psi

def f_morlet_2d1t(kx, ky, omega, a_s, a_t, theta, epsilon=1.0, k0=6.0, omega0=6.0):
    cos_theta = cp.cos(theta)
    sin_theta = cp.sin(theta)
    omega_ = a_t * omega
    kx_ = a_s * (kx * cos_theta + ky * sin_theta)
    ky_ = a_s * (ky * cos_theta - kx * sin_theta)

    result = a_s * cp.sqrt(a_t * epsilon) * cp.sqrt(8*cp.pi**3)
    result = result * (cp.exp(-0.5*(omega_-omega0)**2) - cp.exp(-0.5*(omega0**2 + omega_**2)))
    result = result * (cp.exp(-0.5*((k0-kx_)**2 + epsilon*ky_**2)) - cp.exp(-0.5*(k0**2 + kx_**2 + epsilon*ky_**2)))
    return result

def cwt_2d1t(s, kx, ky, omega, dx, dy, dt, epsilon=1.0, k0=6.0, omega0=6.0):
    # 转换kx、ky为像素坐标并转换为cupy数组
    kx_cp = dx*cp.asarray(kx[None, None, None, :, None, None], dtype=cp.float32)
    ky_cp = dy*cp.asarray(ky[None, None, None, None, :, None], dtype=cp.float32)
    # 计算a_s和theta
    a_s_cp = cp.abs(k0)/cp.sqrt(kx_cp**2 + ky_cp**2)
    theta_cp = cp.arctan2(ky_cp, kx_cp)
    del kx_cp, ky_cp
    # 计算a_t
    a_t_cp = cp.abs(omega0)/dt/cp.asarray(omega[None, None, None, None, None, :], dtype=cp.float32)

    # 获取傅里叶变换后的所有波数和频率
    kx_s_cp = 2 * cp.pi * cp.fft.fftfreq(s.shape[0], d=1)[:, None, None, None, None, None]
    ky_s_cp = 2 * cp.pi * cp.fft.fftfreq(s.shape[1], d=1)[None, :, None, None, None, None]
    omega_s_cp = 2 * cp.pi * cp.fft.fftfreq(s.shape[2], d=1)[None, None, :, None, None, None]

    # 计算频域的小波函数
    f_psi_cp = f_morlet_2d1t(kx_s_cp, ky_s_cp, omega_s_cp, a_s_cp, a_t_cp, theta_cp, epsilon=epsilon, k0=k0, omega0=omega0)
    del kx_s_cp, ky_s_cp, omega_s_cp, a_s_cp, a_t_cp, theta_cp

    # 计算信号的傅里叶变换
    f_s_cp = cp.fft.fftn(cp.asarray(s, dtype=cp.float32))[:, :, :, None, None, None]
    # 计算频域乘积
    f_s_cp = f_s_cp * f_psi_cp
    # 计算傅里叶逆变换
    f_s_cp = cp.fft.ifftn(f_s_cp, axes=(0, 1, 2))
    result = cp.asnumpy(f_s_cp).astype(np.complex64)
    del f_s_cp
    return result*np.sqrt(dx*dy*dt)

def icwt_2d1t(w, kx, ky, omega, dx, dy, dt, epsilon=1.0, k0=6.0, omega0=6.0):
    sign = -1*np.sign(omega[1]-omega[0])*np.sign(kx[1]-kx[0])*np.sign(ky[1]-ky[0])
    # 转换kx、ky为像素坐标并转换为cupy数组
    kx_cp = dx*cp.asarray(kx[None, None, None, :, None, None], dtype=cp.float32)
    ky_cp = dy*cp.asarray(ky[None, None, None, None, :, None], dtype=cp.float32)
    # 计算a_s和theta
    a_s_cp = cp.abs(k0)/cp.sqrt(kx_cp**2 + ky_cp**2)
    theta_cp = cp.arctan2(ky_cp, kx_cp)
    del kx_cp, ky_cp
    # 计算a_t
    a_t_cp = cp.abs(omega0)/dt/cp.asarray(omega[None, None, None, None, None, :], dtype=cp.float32)

    # 获取傅里叶变换后的所有波数和频率
    kx_s_cp = 2 * cp.pi * cp.fft.fftfreq(w.shape[0], d=1)[:, None, None, None, None, None]
    ky_s_cp = 2 * cp.pi * cp.fft.fftfreq(w.shape[1], d=1)[None, :, None, None, None, None]
    omega_s_cp = 2 * cp.pi * cp.fft.fftfreq(w.shape[2], d=1)[None, None, :, None, None, None]

    # 计算频域的小波函数
    f_psi_cp = f_morlet_2d1t(kx_s_cp, ky_s_cp, omega_s_cp, a_s_cp, a_t_cp, theta_cp, epsilon=epsilon, k0=k0, omega0=omega0)
    del kx_s_cp, ky_s_cp, omega_s_cp, a_s_cp, theta_cp

    # 计算小波系数的傅里叶变换
    f_w_cp = cp.fft.fftn(cp.asarray(w, dtype=cp.complex64), axes=(0, 1, 2))

    # 计算被积函数
    f_w_cp *= f_psi_cp
    del f_psi_cp
    f_w_cp /= (a_t_cp**2 * k0**2)
    del a_t_cp
    # 进行积分
    a_t_i = cp.abs(omega0)/dt/cp.asarray(omega, dtype=cp.float32)
    f_w_cp = cp.trapz(f_w_cp, x=a_t_i, axis=-1)
    del a_t_i

    ky_i = dy*cp.asarray(ky, dtype=cp.float32)
    f_w_cp = cp.trapz(f_w_cp, x=ky_i, axis=-1)
    del ky_i

    kx_i = dx*cp.asarray(kx, dtype=cp.float32)
    f_w_cp = cp.trapz(f_w_cp, x=kx_i, axis=-1)
    del kx_i

    # 逆傅里叶变换
    f_w_cp = cp.fft.ifftn(f_w_cp)
    result = cp.asnumpy(f_w_cp.real).astype(np.float32)
    del f_w_cp

    return sign*result / np.sqrt(dx*dy*dt) / c_psi(k0, omega0, epsilon)

def split_array(m, n):
    if n < 1:
        raise ValueError("块数n必须≥1")
    if m < 0:
        raise ValueError("数组长度m不能为负数")
    if n > m:
        raise ValueError("块数n不能大于数组长度")
    q, r = divmod(m, n)
    index = []
    start = 0

    for i in range(n):
        length = q + 1 if i < r else q
        index.append((start, start + length))
        start += length

    return index

def split_array_int(m, n):
    # 边界校验：块数不能小于1，数组长度非负
    if n < 1:
        raise ValueError("块数n必须≥1")
    if m < 0:
        raise ValueError("数组长度m不能为负数")
    if n > m-1:
        raise ValueError("块数n不能大于数组长度-1")

    # 只有1块时，直接返回整个数组的索引区间
    if n == 1:
        return [(0, m)]

    # 核心计算：总元素位（含共享元素）= 原长度 + 块数 - 1
    total_pos = m + n - 1
    q, r = divmod(total_pos, n)  # q=基础长度，r=需要多1个元素的块数

    index = []
    start = 0  # 每块的起始索引

    for i in range(n):
        # 确定当前块的长度（前r块多1个元素，保证长度差异≤1）
        length = q + 1 if i < r else q
        end = start + length  # 左闭右开的结束索引
        index.append((start, end))
        # 关键：下一块的起始 = 当前块的最后一个元素索引（end-1），实现首尾共享
        start = end - 1
        # 兜底：避免start超过数组最大索引（当块数远大于数组长度时）
        if start >= m:
            start = m - 1 if m > 0 else 0

    return index

def full_3d_psd(data, kx_array, ky_array, omega_array, dx, dy, dt, epsilon=1, k0=6, omega0=6, blocks=(4, 4, 4)):
    psd = np.zeros((data.shape[0], data.shape[1], data.shape[2], len(kx_array), len(ky_array), len(omega_array)))
    # fft_data = scipy.fft.fftn(data)

    # 划分参数数组的索引范围（每个元素为(start, end)
    sub_kx_index = split_array(len(kx_array), blocks[0]) # kx的块索引
    sub_ky_index = split_array(len(ky_array), blocks[1]) # ky的块索引
    sub_omega_index = split_array(len(omega_array), blocks[2]) # omega的块索引

    # 生成三个维度块索引的所有组合（笛卡尔积），用单重循环遍历
    total = len(sub_kx_index) * len(sub_ky_index) * len(sub_omega_index)
    for kx_idx, ky_idx, omega_idx in tqdm(product(sub_kx_index, sub_ky_index, sub_omega_index),
                                          total=total, desc='processing parameter blocks'):
        sub_kx = kx_array[kx_idx[0]:kx_idx[1]]
        sub_ky = ky_array[ky_idx[0]:ky_idx[1]]
        sub_omega = omega_array[omega_idx[0]:omega_idx[1]]

        # 计算小波变换
        # w = wt.cwt2DT(fft_data, 1/dx, 1/dy, 1/dt, 2*np.pi/sub_kx, 2*np.pi/sub_ky, 2*np.pi/sub_omega)
        w = cwt_2d1t(data, sub_kx, sub_ky, sub_omega, dx, dy, dt, epsilon=epsilon, k0=k0, omega0=omega0)
        psd[:, :, :, kx_idx[0]:kx_idx[1], ky_idx[0]:ky_idx[1], omega_idx[0]:omega_idx[1]] = np.abs(w)**2

    return psd

def full_3d_cwt(data, kx_array, ky_array, omega_array, dx, dy, dt, epsilon=1, k0=6, omega0=6, blocks=(4, 4, 4)):
    w = np.zeros((data.shape[0], data.shape[1], data.shape[2], len(kx_array), len(ky_array), len(omega_array)), dtype=np.complex64)

    # 划分参数数组的索引范围（每个元素为(start, end)
    sub_kx_index = split_array(len(kx_array), blocks[0]) # kx的块索引
    sub_ky_index = split_array(len(ky_array), blocks[1]) # ky的块索引
    sub_omega_index = split_array(len(omega_array), blocks[2]) # omega的块索引

    # 生成三个维度块索引的所有组合（笛卡尔积），用单重循环遍历
    total = len(sub_kx_index) * len(sub_ky_index) * len(sub_omega_index)
    for kx_idx, ky_idx, omega_idx in tqdm(product(sub_kx_index, sub_ky_index, sub_omega_index),
                                          total=total, desc='processing parameter blocks'):
        sub_kx = kx_array[kx_idx[0]:kx_idx[1]]
        sub_ky = ky_array[ky_idx[0]:ky_idx[1]]
        sub_omega = omega_array[omega_idx[0]:omega_idx[1]]

        # 计算小波变换
        w[:, :, :, kx_idx[0]:kx_idx[1], ky_idx[0]:ky_idx[1], omega_idx[0]:omega_idx[1]] = cwt_2d1t(data, sub_kx, sub_ky, sub_omega, dx, dy, dt, epsilon=epsilon, k0=k0, omega0=omega0)

    return w

def full_3d_icwt(w, kx_array, ky_array, omega_array, dx, dy, dt, epsilon=1, k0=6, omega0=6, blocks=(4, 4, 4)):
    signal = np.zeros((w.shape[0], w.shape[1], w.shape[2]))

    # 划分参数数组的索引范围（每个元素为(start, end)）
    sub_kx_index = split_array_int(len(kx_array), blocks[0]) # kx的块索引
    sub_ky_index = split_array_int(len(ky_array), blocks[1]) # ky的块索引
    sub_omega_index = split_array_int(len(omega_array), blocks[2]) # omega的块索引

    # 生成三个维度块索引的所有组合（笛卡尔积），用单重循环遍历
    total = len(sub_kx_index) * len(sub_ky_index) * len(sub_omega_index)
    for kx_idx, ky_idx, omega_idx in tqdm(product(sub_kx_index, sub_ky_index, sub_omega_index),
                                          total=total, desc='processing parameter blocks'):
        sub_kx = kx_array[kx_idx[0]:kx_idx[1]]
        sub_ky = ky_array[ky_idx[0]:ky_idx[1]]
        sub_omega = omega_array[omega_idx[0]:omega_idx[1]]
        sub_w = w[:, :, :, kx_idx[0]:kx_idx[1], ky_idx[0]:ky_idx[1], omega_idx[0]:omega_idx[1]]

        # 计算小波逆变换
        signal += icwt_2d1t(sub_w, sub_kx, sub_ky, sub_omega, dx, dy, dt, epsilon=epsilon, k0=k0, omega0=omega0)
    return signal