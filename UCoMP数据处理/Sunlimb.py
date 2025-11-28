import sunpy.map
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from matplotlib.ticker import FuncFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize

def to_polar(sun_map, deg_range, r_range=None, r=None, **kwargs):
    """
    将原始数据投影到极坐标
    :param sun_map: 需要转换的Sunlimb.Sun对象或sunpy.Map.map对象
    :param deg_range: 角度范围 [unit: degree]
    :param r_range: 径向范围 [unit: Rs]
    :param r: 距离太阳边界距离 [unit: Mm]
    :param kwargs: d_r[arcsec/pix], d_deg[degree/pix]
    :return:
    """
    if (r_range is None) == (r is None):
        raise ValueError("You must provide exactly one of 'r_range' or 'r', not both and not neither.")

    fits_header = sun_map.fits_header

    d_r = kwargs.get('d_r', fits_header['CDELT1'])  # unit: arcsec/pix
    if r_range is None:
        r_vect = np.array((r * 1e6 / fits_header['RSUN_REF'] + 1) * fits_header['RSUN_OBS'])
    else:
        r_vect = np.arange(r_range[0] * fits_header['RSUN_OBS'], r_range[1] * fits_header['RSUN_OBS'] + d_r, d_r)

    d_deg = kwargs.get('d_deg', 360 * fits_header['CDELT1'] / (2 * np.pi * fits_header['RSUN_OBS']))  # unit: deg/pix
    deg_vect = np.arange(deg_range[0], deg_range[1] + d_deg / 2, d_deg)

    # Generate the grid
    r_vert_grid, deg_vert_grid = np.meshgrid(r_vect, deg_vect, indexing='ij')
    cos_theta = np.cos(np.deg2rad(deg_vert_grid))
    sin_theta = np.sin(np.deg2rad(deg_vert_grid))
    x_grid = r_vert_grid * sin_theta
    y_grid = r_vert_grid * cos_theta

    # Convert to pixel coordinates
    delta_x_pixel_grid = (x_grid - fits_header['CRVAL1']) / fits_header['CDELT1']
    delta_y_pixel_grid = (y_grid - fits_header['CRVAL2']) / fits_header['CDELT2']
    interp_grid_x = fits_header['CRPIX1'] + delta_x_pixel_grid - 1
    interp_grid_y = fits_header['CRPIX2'] + delta_y_pixel_grid - 1

    # Prepare coordinates for map_coordinates
    coords = np.array([interp_grid_y, interp_grid_x])

    # Interpolate using map_coordinates
    polar_map = map_coordinates(sun_map.data, coords, order=1, mode='nearest')

    return polar_map, r_vect / fits_header['RSUN_OBS'], deg_vect


def plot_plt_imshow(ax, plot_x, plot_y, data, norm, cmap='gray', colorbar=False):
    dx = (plot_x[1] - plot_x[0]) / 2.
    dy = (plot_y[1] - plot_y[0]) / 2.
    extent = [plot_x[0] - dx, plot_x[-1] + dx, plot_y[0] - dy, plot_y[-1] + dy]
    im = ax.imshow(data, extent=extent, cmap=cmap, norm=norm, origin='lower')
    ax.set_aspect('auto')
    return im


def plot_data(ax, data, x_arr, y_arr, colorbar=False, **kwargs):
    vmin = kwargs.get('vmin', np.nanpercentile(data, 1))
    vmax = kwargs.get('vmax', np.nanpercentile(data, 99))
    norm = kwargs.get('norm', Normalize(vmin=vmin, vmax=vmax))
    cmap = kwargs.get('cmap', 'sdoaia171')
    im = plot_plt_imshow(ax, x_arr, y_arr, data, cmap=cmap, norm=norm)
    if "title" in kwargs:
        ax.set_title(kwargs["title"])
    if "xlabel" in kwargs:
        ax.set_xlabel(kwargs['xlabel'])
    if "ylabel" in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # 调整 size 和 pad 以控制 colorbar 大小和位置
        plt.colorbar(im, cax=cax)
    plt.show()


class Sun:
    def __init__(self, file, preview=False):
        if type(file) is str:
            self.map = sunpy.map.Map(file)
        else:
            self.map = file
        self.rotate_map = self.map.rotate()
        self.fits_header = self.rotate_map.fits_header
        self.data = self.rotate_map.data
        if preview:
            data, r_vect, theta_vect = to_polar(self.rotate_map, r_range=(0, 1.3), deg_range=(0, 360), d_r=5, d_angle=1)
            plt.pcolor(theta_vect, r_vect, data, cmap='sdoaia171')
            plt.show()

    def raw_image(self, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(projection=self.rotate_map)
        self.rotate_map.plot(axes=ax, **kwargs)

    def cut(self, deg_range, r_range=None, r=None, plot=False, process=None, **kwargs):
        data, y, x = to_polar(self, deg_range, r_range, r, **kwargs)
        if process is not None:
            data = process(data)
        if plot:
            fig, ax = plt.subplots()
            plot_data(ax, data, x, y, **kwargs)

            xlabel = kwargs.get('xlabel', 'deg')
            ylabel = kwargs.get('ylabel', 'Rs')
            title = kwargs.get('title', self.fits_header['date'])

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        return data

    def draw_limb(self, deg_range, r_range, limbs, process=None, **kwargs):
        data, y, x = to_polar(self.rotate_map, deg_range, r_range=r_range, **kwargs)
        if process is not None:
            data = process(data)

        def convert(y_tick, pos):  # Rs转换为Mm
            Rs = y_tick * (r_range[1] - r_range[0]) + r_range[0]
            Mm = (Rs - 1) * self.fits_header['RSUN_REF'] / 1e6
            return f'{Mm:.2f}'

        figsize = kwargs.get('figsize', (8, 6))
        fig, ax = plt.subplots(figsize=figsize)
        plot_data(ax, data, x, y, **kwargs)

        color = kwargs.get('color', 'k')
        linestyle = kwargs.get('linestyle', '--')
        title = kwargs.get('title', self.fits_header['date-obs'])

        if not isinstance(limbs, (list, tuple, np.ndarray)):
            limbs = [limbs]

        for limb in limbs:
            y_line = limb * 1e6 / self.fits_header['RSUN_REF'] + 1
            ax.hlines(y_line, deg_range[0], deg_range[-1], color=color, linestyle=linestyle)

        ax1 = ax.twinx()
        ax1.set_yticks(ax1.get_yticks())
        ax1.yaxis.set_major_formatter(FuncFormatter(convert))
        ax1.set_ylabel('Mm')

        ax.set_title(title)
        ax.set_xlabel('deg')
        ax.set_ylabel('Rs')
        return data


def process_file(file, process, deg_range, r, d_deg, **kwargs):
    euv_map = Sun(file)
    if process is not None:
        euv_map.data = process(euv_map.data)
    return euv_map.cut(deg_range, r=r, **{'d_deg': d_deg, **kwargs})


def space_time_plot(ax, files, deg_range, r, d_time, d_deg=0.01, process=None, **kwargs):
    space_time_data = []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file, process, deg_range, r, d_deg, **kwargs): file for file in files}

        for future in tqdm(as_completed(futures), total=len(files), desc="Processing files"):
            space_time_data.append(future.result())

    space_time_data = np.vstack(space_time_data)
    deg_vect = np.arange(deg_range[0], deg_range[1] + d_deg / 2, d_deg)
    time_vect = np.arange(0, len(files) * d_time + 0.5 * d_time, d_time) / 60  # unit: minute

    plot_data(ax, space_time_data, deg_vect, time_vect, **kwargs)

    title = kwargs.get('title', "space_time_plot")
    x_label = kwargs.get('x_label', 'degree')
    y_label = kwargs.get('y_label', 'time(mim)')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_aspect('auto')

    return space_time_data
