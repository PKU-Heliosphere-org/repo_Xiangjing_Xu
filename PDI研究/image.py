import cv2
import numpy as np
import os
from watroo import AtrousTransform, B3spline, Triangle
import re
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image


def box_car_smoothing(image, kernel_size):
    # 创建盒式滤波器的内核
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    # 对图像应用滤波器
    smoothed_image = cv2.filter2D(image, -1, kernel)
    return smoothed_image


def unsharp_mask(image, kernel_size, amount=1.0, high_freq_only=True):
    # 对图像进行平滑处理
    smoothed_image = box_car_smoothing(image, kernel_size)
    # 计算高频信息
    high_freq_image = image - smoothed_image
    if high_freq_only:
        return high_freq_image
    # 根据系数调整高频信息，并加回原图
    sharpened_image = image + amount * high_freq_image
    return sharpened_image


def a_trous(image, level, method='B3spline'):
    if method == 'B3spline':
        transform = AtrousTransform(B3spline)
    elif method == 'Triangle':
        transform = AtrousTransform(Triangle)
    else:
        raise ValueError('method must be B3spline or Triangle')

    return transform(image, level)


def images_to_video(input_path, output_movie, fps=30):
    images = [img for img in os.listdir(input_path) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()

    if not images:
        print("No images found in the folder.")
        return

    first_image_path = os.path.join(input_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_movie, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(input_path, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()


def images_to_gif(input_path, output_gif, width=None, fps=30):
    image_files = [img for img in os.listdir(input_path) if img.endswith(".jpg") or img.endswith(".png")]
    image_files.sort()
    images = [Image.open(os.path.join(input_path, img)) for img in image_files]

    if width is not None:
        height = int(width * images[0].height / images[0].width)
        target_size = (width, height)
        images = [img.resize(target_size, Image.LANCZOS) for img in images]

    # 将调整后的图片保存为GIF
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=int(1000/fps), loop=0)