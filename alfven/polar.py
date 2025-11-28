import numpy as np
import concurrent.futures
from tqdm import tqdm
from numba import jit


@jit(nopython=True)
def count_value(pix, pix0, data0, delta, rsun_arc, crota, cdelt1, cdelt2, max_x, max_y, euv_map_data):
    """
    计算极坐标图像像素点处对应的值
    :param pix: 元组，行列像素
    :param pix0: 元组，太阳中心点对应的笛卡尔坐标行列像素
    :param data0: 元组，行列起始处的像素点对应的值
    :param delta: 元组，行列每像素对应的
    :param rsun_arc: 太阳半径的角秒数
    :param crota: 图像的旋转角度
    :param cdelt1: 每像素的横向分辨率
    :param cdelt2: 每像素的纵向分辨率
    :param max_x: 横轴最大像素值
    :param max_y: 纵轴最大像素值
    :param euv_map_data: sunpy.Map.map.data
    :return: 对应像素点的值
    """
    r = (pix[0] - 0.5) * delta[0] + data0[0] * rsun_arc
    theta = ((pix[1] - 0.5) * delta[1] + data0[1] + crota) / 180 * np.pi

    pix_x = r * np.sin(theta) / cdelt1 + pix0[1]
    pix_y = pix0[0] - r * np.cos(theta) / cdelt2

    if 1 <= pix_x < max_x and 1 <= pix_y < max_y:
        x0 = int(np.floor(pix_x))
        x1 = x0 + 1
        y0 = int(np.floor(pix_y))
        y1 = y0 + 1

        if x1 >= max_x:
            x1 = x0
        if y1 >= max_y:
            y1 = y0

        Ia = euv_map_data[y0, x0]
        Ib = euv_map_data[y1, x0]
        Ic = euv_map_data[y0, x1]
        Id = euv_map_data[y1, x1]

        wa = (x1 - pix_x) * (y1 - pix_y)
        wb = (x1 - pix_x) * (pix_y - y0)
        wc = (pix_x - x0) * (y1 - pix_y)
        wd = (pix_x - x0) * (pix_y - y0)

        interpolated_value = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return pix[0], pix[1], interpolated_value
    else:
        return pix[0], pix[1], np.nan


def to_polar(euv_map, r_range, deg_range, **kwargs):
    """
    坐标转换
    :param euv_map: 原始sunpy.Map.map对象
    :param r_range: 径向坐标范围
    :param deg_range: 角度范围，北极点算0度，顺时针增加
    :param kwargs: deg_pix, r_pix, 分别代表每像素的角度值、角秒值
    :return:
    """
    rsun_arc = euv_map.fits_header['RSUN_ARC']
    crota = euv_map.fits_header['CROTA']
    cdelt1 = euv_map.fits_header['CDELT1']
    cdelt2 = euv_map.fits_header['CDELT2']
    max_x = euv_map.fits_header["NAXIS1"]
    max_y = euv_map.fits_header["NAXIS2"]
    euv_map_data = euv_map.data

    deg_pix = kwargs.get('deg_pix', cdelt1 / rsun_arc * 180 / np.pi)
    r_pix = kwargs.get('r_pix', cdelt1)

    # dpix1 = euv_map.fits_header['CRVAL1'] / cdelt1
    # dpix2 = euv_map.fits_header['CRVAL2'] / cdelt1
    # rota = crota / 180 * np.pi
    # pix_x0 = euv_map.fits_header['CRPIX1'] - dpix1 * np.cos(rota) - dpix2 * np.sin(rota)
    # pix_y0 = euv_map.fits_header['CRPIX2'] - dpix1 * np.sin(rota) + dpix2 * np.cos(rota)
    pix_x0 = euv_map.fits_header['EUXCEN']
    pix_y0 = max_y - euv_map.fits_header['EUYCEN'] + 1

    shape = (int((r_range[1] - r_range[0]) * rsun_arc / r_pix), int((deg_range[1] - deg_range[0]) / deg_pix))
    deg_arr = np.arange(deg_range[0] + 0.5 * deg_pix, deg_range[0] + shape[1] * deg_pix, deg_pix)
    r_arr = np.arange(r_range[0] + 0.5 * r_pix / rsun_arc, r_range[0] + shape[0] * r_pix / rsun_arc, r_pix / rsun_arc)

    tasks = [((i, j), (pix_y0, pix_x0), (r_range[0], deg_range[0]), (r_pix, deg_pix), rsun_arc, crota, cdelt1, cdelt2,
              max_x, max_y, euv_map_data)
             for i in range(1, shape[0] + 1) for j in range(1, shape[1] + 1)]

    polar_data = np.empty(shape=shape)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda p: count_value(*p), tasks), total=len(tasks), desc="Processing"))

    for x, y, value in results:
        polar_data[x - 1, y - 1] = value

    return np.flipud(polar_data), r_arr, deg_arr
