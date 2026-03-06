import cupy as cp
import numpy as np

def f_morlet_2d1t(kx, ky, omega, a_s, a_t, theta, epsilon=0.5, k0=-6, omega0=6):
    cos_theta = cp.cos(theta)
    sin_theta = cp.sin(theta)
    omega_ = a_t * omega
    kx_ = a_s * (kx * cos_theta + ky * sin_theta)
    ky_ = a_s * (ky * cos_theta - kx * sin_theta)

    result = a_s*cp.sqrt(a_t/epsilon)
    result = result * cp.exp(-0.5*((omega_-omega0)**2 + epsilon*(kx_/epsilon-k0)**2 + ky_**2))
    return result * cp.sqrt(8*cp.pi**3)

def cwt_2d1t(s, kx, ky, omega, dx, dy, dt, epsilon=0.5, k0=-6, omega0=6):
    # 转换kx、ky为像素坐标并转换为cupy数组
    kx_cp = dx*cp.asarray(kx[None, None, None, :, None, None], dtype=cp.float32)
    ky_cp = dy*cp.asarray(ky[None, None, None, None, :, None], dtype=cp.float32)
    # 计算a_s和theta
    a_s_cp = epsilon*cp.abs(k0)/cp.sqrt(kx_cp**2 + ky_cp**2)
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
    return result

def icwt_2d1t(w, kx, ky, omega, dx, dy, dt, epsilon=0.5, k0=-6, omega0=6):
    # 转换kx、ky为像素坐标并转换为cupy数组
    kx_cp = dx*cp.asarray(kx[None, None, None, :, None, None], dtype=cp.float32)
    ky_cp = dy*cp.asarray(ky[None, None, None, None, :, None], dtype=cp.float32)
    # 计算a_s和theta
    a_s_cp = epsilon*cp.abs(k0)/cp.sqrt(kx_cp**2 + ky_cp**2)
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
    f_w_cp /= (a_t_cp**2 * k0**2 * epsilon**2)
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

    return result / np.sqrt(epsilon**5 * k0**4 * omega0**2 / np.pi**3)