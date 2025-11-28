from astropy.io import fits
import sunpy.map
import image_registration
from tqdm import tqdm
import os
import re
from datetime import datetime
import numpy as np

file_path = "./data/"
files = [os.path.abspath(os.path.join(file_path, file)) for file in os.listdir(file_path) if file.endswith('.fits')]
def extract_datetime(file_name):
    match = re.search(r'\d{8}T\d{6}', file_name)
    if match:
        return datetime.strptime(match.group(), '%Y%m%dT%H%M%S')
    else:
        return datetime.min

files = sorted(files, key=extract_datetime)

def sub_data(euv, x_range, y_range):
    """
    选取一定像素范围图像进行比对
    :param euv: fits文件或sunpy.map.Map对象
    :param x_range: x轴像素范围
    :param y_range: x轴像素范围
    :return: 裁剪后的数据
    """
    if type(euv) is str:
        euv_data = fits.getdata(euv)
    else:
        euv_data = euv.data

    return euv_data[y_range[0]:y_range[1], x_range[0]:x_range[1]]

def get_offset(euv1, euv2, x_range, y_range):
    """
    得到hri图像之间的偏移量
    :param euv1: 标定图像，将会被投影到euv2的观察者系，fits文件或sunpy.map.Map对象
    :param euv2: 需要被移动的图像，fits文件或sunpy.map.Map对象
    :param x_range: x轴像素范围
    :param y_range: x轴像素范围
    :return: 平移量
    """
    if type(euv1) is str:
        euv1 = sunpy.map.Map(euv1)
    if type(euv2) is str:
        euv2 = sunpy.map.Map(euv2)

    euv1_reproject = euv1.reproject_to(euv2.wcs)
    data1 = sub_data(euv1_reproject, x_range, y_range)
    data2 = sub_data(euv2, x_range, y_range)
    dx, dy = image_registration.chi2_shift(data2, data1, return_error=False)

    return dx, dy

x_range = (640, 1660)
y_range = (1250, 2270)

offset_vect = []
for i in tqdm(range(600), total=600, desc='computing offsets'):
    dx, dy = get_offset(files[0], files[i], x_range, y_range)
    offset_vect.append([dx, dy])

np.save('./offset_array.npy', offset_vect)