import sunpy.map
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import itertools
import numpy as np


# 绘制图像
def draw_hri_fig(hri_map, rotate=False, **kwargs):
    """
    绘制sunpy.map.Map对象图像
    :param hri_map: 文件名或sunpy.map.Map对象
    :param rotate: bool，True则按照CROTA对图像进行旋转
    :param kwargs: 传递给sunpy.map.Map.plot()方法的其他参数
    :return: none
    """
    if type(hri_map) is str:
        hri_map = sunpy.map.Map(hri_map)
    if rotate:
        hri_map = hri_map.rotate()
    fig = plt.figure()
    ax = fig.add_subplot(projection=hri_map)
    hri_map.plot(axes=ax, **kwargs)


# 绘制submap
def get_submap(hri_map, c_x, c_y, l, draw=True, **kwargs):
    """
    得到sunpy.map.Map对象图像的正方形submap
    :param hri_map: 文件名或sunpy.map.Map对象
    :param c_x: submap中心处的角秒横坐标
    :param c_y: submap中心处的角秒纵坐标
    :param l: 正方形图像的像素
    :param draw: bool，若为True则会绘制submap
    :param kwargs: 传递给sunpy.map.Map.plot()方法的其他参数
    :return: 原map的submap，sunpy.map.Map对象
    """
    if type(hri_map) is str:
        hri_map = sunpy.map.Map(hri_map)

    origin = SkyCoord(c_x * u.arcsec, c_y * u.arcsec, frame=hri_map.coordinate_frame)
    out_shape = (l, l)
    out_header = sunpy.map.make_fitswcs_header(
        out_shape,
        origin,
        scale=[0.492, 0.492] * u.arcsec / u.pix,
        projection_code="TAN"
    )
    supplement_list = [
        'DSUN_OBS',
        'RSUN_OBS',
        'RSUN_ARC',
        'FILENAME'
    ]
    unit_trasnform = {
        'CUNIT1': 'arcsec',
        'CUNIT2': 'arcsec',
        'CDELT1': 0.492,
        'CDELT2': 0.492,
        'CRVAL1': c_x,
        'CRVAL2': c_y
    }

    out_map = hri_map.reproject_to(out_header)
    for key in supplement_list:
        out_map.meta[key] = hri_map.meta[key]
    for key, value in unit_trasnform.items():
        out_map.meta[key] = value

    if draw:
        draw_hri_fig(out_map, **kwargs)

    return out_map


# 移动图像
def offset_map(hri_map, offset, draw=False, unit='pix', **kwargs):
    """
    对图像进行平移
    :param hri_map: 文件名或sunpy.map.Map对象
    :param offset: 横纵坐标的平移量
    :param draw: bool，True则绘制平移后的图像
    :param unit: pix或arcsec
    :param kwargs: 传递给sunpy.map.Map.plot()方法的其他参数
    :return: 平移后的map，sunpy.map.Map对象
    """
    if type(hri_map) is str:
        hdu = fits.open(hri_map)
        header = hdu[-1].header
        data = hdu[-1].data
    else:
        header = hri_map.fits_header
        data = hri_map.data

    if unit == 'pix':
        cdelt = header['CDELT1']
        offset = [o * cdelt for o in offset]

    header['CRVAL1'] += offset[0]
    header['CRVAL2'] += offset[1]

    offset_hri_map = sunpy.map.Map(data, header)

    if draw:
        draw_hri_fig(offset_hri_map, **kwargs)

    return offset_hri_map


def get_diff(hri, fsi, c_x, c_y, l, offset):
    hri_offset = offset_map(hri, offset, unit='arcsec')
    hri_data = get_submap(hri_offset, c_x=c_x, c_y=c_y, l=l, draw=False).data
    fsi_data = get_submap(fsi, c_x=c_x, c_y=c_y, l=l, draw=False).data

    def normalize(image):
        return (image - np.mean(image)) / np.std(image)

    return np.nansum(np.abs(normalize(hri_data) - normalize(fsi_data)))


def iterate_alignment(hri, fsi, c_x, c_y, l, accuracy, offset_range):
    accuracy = np.array(accuracy)
    range_x, range_y = offset_range[0], offset_range[1]
    min_diff = np.inf
    best_offset = (0, 0)
    for a in accuracy:
        offset_vect_x = np.arange(range_x[0], range_x[1] + a, a)
        offset_vect_y = np.arange(range_y[0], range_y[1] + a, a)
        offsets = list(itertools.product(offset_vect_x, offset_vect_y))

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(get_diff, hri, fsi, c_x, c_y, l, offset): offset for offset in offsets}
            for future in tqdm(as_completed(futures), total=len(offsets), desc=f"computing best offset, accracy: {a}",
                               leave=False):
                diff = future.result()
                if diff < min_diff:
                    min_diff = diff
                    best_offset = futures[future]

        range_x = (best_offset[0] - a, best_offset[0] + a)
        range_y = (best_offset[1] - a, best_offset[1] + a)

    return best_offset


file_path = "./data/"
files = [os.path.abspath(os.path.join(file_path, file)) for file in os.listdir(file_path) if file.endswith('.fits')]


def extract_datetime(file_name):
    match = re.search(r'\d{8}T\d{6}', file_name)
    if match:
        return datetime.strptime(match.group(), '%Y%m%dT%H%M%S')
    else:
        return datetime.min


files = sorted(files, key=extract_datetime)
file0 = files[295]
needed_files = files[295:504]
offset_vect = []
for file in tqdm(needed_files):
    best_offset = iterate_alignment(file, file0, c_x=-100, c_y=-2700, l=1200,
                                    offset_range=((-3, 3), (-3, 3)), accuracy=[1, 0.5, 0.2, 0.05])
    offset_vect.append(best_offset)

offset_array = np.array(offset_vect)
np.save('./offset_array.npy', offset_array)
