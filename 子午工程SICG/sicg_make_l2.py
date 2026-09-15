#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sicg_make_l2.py

将 IDL 版本 sicg_make_l2.pro 转换而来的 Python 脚本。

功能：
    从 L1B/SPEI 五波长 FITS 文件生成 SICG L2 产品：
        1. 峰值强度图 Peak Intensity Map
        2. 多普勒速度图 Doppler Map
        3. 谱线线宽图 Line-width Map

依赖：
    numpy
    astropy
    scipy  可选，仅在使用 --despike 平滑时需要；如果没有 scipy，会自动使用纯 numpy 慢速卷积。

示例：
    python sicg_make_l2.py \
        --l1b-dir "D:/L1B/2024-11-13" \
        --outroot "D:/L2/2024-11-13" \
        --wave 637.4 \
        --write-fits \
        --despike

说明：
    原 IDL 脚本假定 FITS 数据维度为 [x, y, wave]，且包含 5 个波长采样。
    Python/astropy 读取 FITS 后的数组轴顺序可能与 IDL 不同，
    因此本脚本提供 --fits-order 参数：
        auto       默认，自动判断五波长轴并转成 [x, y, wave]
        xyw        数据已经是 [x, y, wave]
        wyx        常见 FITS/NumPy 顺序 [wave, y, x]，会转为 [x, y, wave]
        yxw        [y, x, wave]，会转为 [x, y, wave]

作者注：
    本脚本尽量保持原 IDL 版本的算法公式、默认参数、目录结构和输出文件名。
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits


C_LIGHT_KM_S = 3.0e5  # 光速，单位 km/s；与原 IDL 脚本保持一致
SQRT_2LN2 = math.sqrt(2.0 * math.log(2.0))


@dataclass
class Geometry:
    """SICG L2 处理所需的几何和仪器参数。"""

    x0: float = 1022.0       # 掩星盘中心 x 坐标，单位 pixel
    y0: float = 1022.0       # 掩星盘中心 y 坐标，单位 pixel
    rlimb: float = 487.0     # 掩星盘半径 / 拟合内边界，单位 pixel
    rmax: float = 200.0      # 外延拟合范围，实际外边界为 rlimb + rmax
    bandpass: float = 1.0    # 仪器 FWHM，单位 Angstrom


def datestr_from_header(header: fits.Header, filename: str | Path) -> str:
    """
    从 FITS 头的 DATE-OBS 读取观测时间，并转换为 YYYYMMDDhhmmss。

    如果 DATE-OBS 不存在，则从文件名中查找连续 14 位数字作为 fallback。
    如果仍然找不到，则返回 '00000000000000'。
    """
    date_obs = str(header.get("DATE-OBS", "")).strip()

    # 常见格式：2024-11-13T01:01:00.000
    if len(date_obs) >= 19:
        return (
            date_obs[0:4]
            + date_obs[5:7]
            + date_obs[8:10]
            + date_obs[11:13]
            + date_obs[14:16]
            + date_obs[17:19]
        )

    # fallback：从文件名里找 14 位日期时间
    match = re.search(r"\d{14}", Path(filename).name)
    if match:
        return match.group(0)

    return "00000000000000"


def obs_time_from_header(header: fits.Header) -> str:
    """读取 FITS 头里的 DATE-OBS；不存在时返回提示文本。"""
    date_obs = str(header.get("DATE-OBS", "")).strip()
    return date_obs if date_obs else "DATE-OBS missing"


def filetime() -> str:
    """生成写入 L2 FITS 头的当前文件创建时间，格式 YYYY-MM-DDTHH:MM:SS。"""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def set_default_geometry(
    wave: float,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
    rlimb: Optional[float] = None,
    rmax: Optional[float] = None,
    bandpass: Optional[float] = None,
) -> Geometry:
    """
    设置默认几何参数和仪器带宽。

    wave 接近 530.3 nm 时默认 bandpass=0.67 Å；否则默认 bandpass=1.00 Å。
    """
    return Geometry(
        x0=1022.0 if x0 is None else float(x0),
        y0=1022.0 if y0 is None else float(y0),
        rlimb=487.0 if rlimb is None else float(rlimb),
        rmax=200.0 if rmax is None else float(rmax),
        bandpass=(0.67 if abs(wave - 530.3) <= 0.2 else 1.00)
        if bandpass is None
        else float(bandpass),
    )


def wave_offsets(wave: float) -> Tuple[np.ndarray, float]:
    """
    返回 5 个采样波长相对静止中心波长的偏移，单位 Angstrom。

    返回：
        psam: shape=(5,), wavelength_offsets = sampling_wavelengths - wvl0_ang
        wvl0_ang: 采用的静止中心波长，单位 Angstrom
    """
    if abs(wave - 530.3) <= 0.2:
        wvl0_ang = 5302.65
        sampling = np.array([5301.00, 5302.00, 5302.65, 5303.30, 5306.00], dtype=np.float64)
    else:
        wvl0_ang = 6374.35
        sampling = np.array([6372.00, 6373.70, 6374.35, 6375.00, 6377.00], dtype=np.float64)
    return sampling - wvl0_ang, wvl0_ang


def make_annulus(nx: int, ny: int, x0: float, y0: float, rlimb: float, rmax: float) -> np.ndarray:
    """
    生成日冕环形区域 mask。

    mask=True 的像素满足：
        rlimb <= r <= rlimb + rmax
    """
    # indexing='ij' 得到 x.shape=(nx, ny), y.shape=(nx, ny)，匹配 IDL 的 [x,y]
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    rr = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return (rr >= rlimb) & (rr <= (rlimb + rmax))


def _convolve2d_numpy_edge_zero(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    纯 numpy 的二维卷积，边界按 0 处理。

    只有在没有 scipy 且启用 --despike 时使用；速度比 scipy.signal.convolve2d 慢。
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    out = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i : i + kh, j : j + kw]
            out[i, j] = np.sum(window * kernel)
    return out


def smooth_cube(cube: np.ndarray, mask: np.ndarray, smooth_size: int = 3) -> np.ndarray:
    """
    对五波长数据立方体做简单空间平滑，对应 IDL 中的 sicg_l2_smooth。

    平滑方式：
        对每个波长层分别做邻域平均；中心像素权重为 0。
        只有 mask 内、原数据和卷积结果都有限的像素会被替换。

    参数：
        cube: shape=(nx, ny, nw)，会返回平滑后的副本，不直接修改原数组。
        mask: shape=(nx, ny)，环形区域 mask。
        smooth_size: 平滑核大小，自动修正为 >=3 的奇数。
    """
    if cube.ndim != 3:
        return cube

    smooth_size = int(smooth_size)
    if smooth_size < 3:
        smooth_size = 3
    if smooth_size % 2 == 0:
        smooth_size += 1

    kernel = np.ones((smooth_size, smooth_size), dtype=np.float64)
    kernel[smooth_size // 2, smooth_size // 2] = 0.0
    kernel /= np.sum(kernel)

    try:
        from scipy.signal import convolve2d  # type: ignore

        def conv2(d: np.ndarray) -> np.ndarray:
            return convolve2d(d, kernel, mode="same", boundary="fill", fillvalue=0.0)

    except Exception:
        conv2 = lambda d: _convolve2d_numpy_edge_zero(d, kernel)

    out = cube.astype(np.float32, copy=True)
    nw = out.shape[2]
    for iw in range(nw):
        d = out[:, :, iw]
        s = conv2(d)
        ind = mask & np.isfinite(d) & np.isfinite(s) & (np.abs(s - d) > 0.0)
        d[ind] = s[ind]
        out[:, :, iw] = d
    return out


def three_point(
    cube: np.ndarray,
    wave: float,
    bandpass: float,
    x0: float,
    y0: float,
    rlimb: float,
    rmax: float,
    quiet: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    三点 Gaussian 解析反演，对应 IDL 中的 sicg_l2_three_point。

    输入：
        cube: shape=(nx, ny, nw)，需要至少 5 个波长采样。
        wave: 线中心，单位 nm。常用 530.3 或 637.4。
        bandpass: 仪器 FWHM，单位 Angstrom。

    输出：
        inten: 峰值强度图
        dop: 多普勒速度图，单位 km/s
        wid: 去除仪器展宽后的谱线 FWHM，单位 Angstrom
    """
    if cube.ndim != 3:
        raise ValueError("Input cube must be a 3-D array with shape [x, y, wave].")

    nx, ny, nw = cube.shape
    if nw < 5:
        raise ValueError("Input cube must contain five wavelength samples.")

    inten = np.full((nx, ny), np.nan, dtype=np.float32)
    dop = np.full((nx, ny), np.nan, dtype=np.float32)
    wid = np.full((nx, ny), np.nan, dtype=np.float32)

    psam, wvl0_ang = wave_offsets(wave)
    vsam = np.array([1, 2, 3], dtype=int)  # 中间三个波长采样点
    dw = abs(psam[vsam[2]] - psam[vsam[1]])
    gf = bandpass / (2.0 * SQRT_2LN2)

    mask = make_annulus(nx, ny, x0, y0, rlimb, rmax)

    good_width_count = 0
    bad_count = 0

    # 为了最大程度保持 IDL 逻辑，这里使用显式循环。
    # 如果数据很大，可以进一步改为向量化以提升速度。
    for ix in range(nx):
        for iy in range(ny):
            if not mask[ix, iy]:
                continue

            isam = cube[ix, iy, :]
            if np.nanmin(isam[vsam]) <= 0.0:
                bad_count += 1
                continue

            a = math.log(float(isam[vsam[2]] / isam[vsam[1]]))
            b = math.log(float(isam[vsam[0]] / isam[vsam[1]]))

            if not (math.isfinite(a) and math.isfinite(b) and ((a + b) < 0.0)):
                bad_count += 1
                continue

            wo = math.sqrt(-2.0 / (a + b)) * dw
            wfwhm = wo * (2.0 * SQRT_2LN2) / math.sqrt(2.0)
            lam0 = (wo**2) / 4.0 / dw * (a - b) + psam[vsam[1]]
            vel = lam0 / wvl0_ang * C_LIGHT_KM_S

            if wfwhm <= bandpass:
                bad_count += 1
                continue

            gsfwhm_c = math.sqrt(wfwhm**2 - bandpass**2)
            gs_c = gsfwhm_c / (2.0 * SQRT_2LN2)
            gso = wo / math.sqrt(2.0)
            denom = (
                math.sqrt(2.0 * math.pi)
                * gs_c
                * gf
                / gso
                * math.exp(-0.5 * (((lam0 - psam[2]) ** 2) / (gso**2)))
            )

            if denom > 0.0:
                inten[ix, iy] = float(isam[2] / denom)
                dop[ix, iy] = float(vel)
                wid[ix, iy] = float(gsfwhm_c)
                good_width_count += 1
            else:
                bad_count += 1

    if not quiet:
        print(f"Valid fitted pixels: {good_width_count}")
        print(f"Rejected fitted pixels: {bad_count}")

    return inten, dop, wid


def correct_doppler(
    dop: np.ndarray,
    x0: float,
    y0: float,
    rlimb: float,
    rmax: float,
    dop_limit: float = 200.0,
    poly_order: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    多普勒图列方向中值校正，对应 IDL 中的 sicg_l2_correct_doppler。

    当前版本逻辑：
        对每一个 x 列，在环形区域内选择 abs(dop)<dop_limit 的有效像素，
        计算该列的中值速度，并从整列减去该中值。

    poly_order 参数保留只是为了兼容原脚本，目前不参与计算。
    """
    _ = poly_order  # 保留参数但不使用

    nx, ny = dop.shape
    mask = make_annulus(nx, ny, x0, y0, rlimb, rmax)

    dop_col_median = np.full(nx, np.nan, dtype=np.float32)
    for ix in range(nx):
        col = dop[ix, :]
        ind = mask[ix, :] & np.isfinite(col) & (np.abs(col) < dop_limit)
        if np.count_nonzero(ind) > 5:
            dop_col_median[ix] = float(np.nanmedian(col[ind]))

    dop_corr = dop.astype(np.float32, copy=True)
    for ix in range(nx):
        if np.isfinite(dop_col_median[ix]):
            dop_corr[ix, :] = dop[ix, :] - dop_col_median[ix]
        else:
            dop_corr[ix, :] = dop[ix, :]

    return dop_corr, dop_col_median


def update_header(header: fits.Header, content: str, quantity: str, current_filetime: str) -> fits.Header:
    """复制并更新 L2 产品 FITS 头。"""
    hdr = header.copy()
    hdr["DATA_LEV"] = "L2"
    hdr["CONTENT"] = content
    hdr["QUANTITY"] = quantity
    hdr["FILETIME"] = current_filetime
    hdr["DATE"] = (current_filetime, "time when this file was initially created")
    return hdr


def write_one(outfile: str | Path, data_map: np.ndarray, header0: fits.Header, content: str, quantity: str) -> None:
    """写出单个 L2 FITS 产品。"""
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    hdr = update_header(header0, content, quantity, filetime())
    fits.writeto(
        outfile,
        np.asarray(data_map, dtype=np.float32),
        hdr,
        overwrite=True,
        output_verify="ignore",
    )


def normalize_cube_order(cube: np.ndarray, fits_order: str = "auto") -> np.ndarray:
    """
    将 FITS 读取出的数据统一转换为 [x, y, wave]。

    fits_order:
        auto: 自动识别长度为 5 的波长轴；若最后一轴为 5，则认为已经是 [x,y,wave]；
              若第一轴为 5，则认为是 [wave,y,x] 并转为 [x,y,wave]。
        xyw:  不转置。
        wyx:  [wave,y,x] -> [x,y,wave]
        yxw:  [y,x,wave] -> [x,y,wave]
    """
    if cube.ndim != 3:
        raise ValueError(f"FITS data are not 3-D. Actual shape: {cube.shape}")

    order = fits_order.lower()
    if order == "xyw":
        out = cube
    elif order == "wyx":
        out = np.transpose(cube, (2, 1, 0))
    elif order == "yxw":
        out = np.transpose(cube, (1, 0, 2))
    elif order == "auto":
        if cube.shape[-1] == 5:
            out = cube
        elif cube.shape[0] == 5:
            out = np.transpose(cube, (2, 1, 0))
        else:
            # 如果没有明确的 5 波长轴，按原样返回，让后续 nw<5 报错或由用户指定参数。
            out = cube
    else:
        raise ValueError("fits_order must be one of: auto, xyw, wyx, yxw")

    return np.asarray(out, dtype=np.float32)


def find_l1b_files(l1b_dir: str | Path, wave: float) -> Tuple[list[Path], str]:
    """按原 IDL 逻辑在目录中搜索 L1B FITS 文件，并返回文件列表和 expo 字符串。"""
    l1b_dir = Path(l1b_dir)
    expo = "5303" if abs(wave - 530.3) <= 0.2 else "6374"

    files = sorted(l1b_dir.glob(f"*SPEI_L1B*_{expo}.FITS"))
    if not files:
        files = sorted(l1b_dir.glob(f"*L1B*_{expo}.FITS"))

    # 兼容小写扩展名
    if not files:
        files = sorted(l1b_dir.glob(f"*SPEI_L1B*_{expo}.fits"))
    if not files:
        files = sorted(l1b_dir.glob(f"*L1B*_{expo}.fits"))

    return files, expo


def make_l2(
    l1b_dir: Optional[str | Path] = None,
    l1b_files: Optional[Sequence[str | Path]] = None,
    outroot: str | Path = "./L2",
    wave: float = 637.4,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
    rlimb: Optional[float] = None,
    rmax: Optional[float] = None,
    bandpass: Optional[float] = None,
    dop_limit: float = 200.0,
    poly_order: int = 5,
    smooth_size: int = 3,
    write_fits: bool = False,
    despike: bool = False,
    quiet: bool = False,
    fits_order: str = "auto",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    主处理函数，对应 IDL 中的 sicg_make_l2。

    返回：
        int_out, dop_out, wid_out：最后一个处理文件的三个 L2 map。
        如果没有成功处理文件，则返回 (None, None, None)。
    """
    geom = set_default_geometry(wave, x0=x0, y0=y0, rlimb=rlimb, rmax=rmax, bandpass=bandpass)
    outroot = Path(outroot)

    if l1b_files is None:
        if l1b_dir is None:
            raise ValueError("Please set l1b_dir or l1b_files.")
        files, expo = find_l1b_files(l1b_dir, wave)
    else:
        files = sorted(Path(p) for p in l1b_files)
        expo = "5303" if abs(wave - 530.3) <= 0.2 else "6374"

    if not files:
        raise FileNotFoundError("No L1B FITS files found.")

    if not quiet:
        print(f"Number of L1B files: {len(files)}")

    int_dir = outroot / "INT"
    dop_dir = outroot / "DOP"
    wid_dir = outroot / "WID"
    if write_fits:
        int_dir.mkdir(parents=True, exist_ok=True)
        dop_dir.mkdir(parents=True, exist_ok=True)
        wid_dir.mkdir(parents=True, exist_ok=True)

    int_out: Optional[np.ndarray] = None
    dop_out: Optional[np.ndarray] = None
    wid_out: Optional[np.ndarray] = None

    for file in files:
        if not quiet:
            print(f"Processing: {file}")

        with fits.open(file, memmap=False) as hdul:
            # 默认读取主 HDU；如你的数据在其他 HDU，可按需要修改这里。
            raw_cube = hdul[0].data
            header = hdul[0].header.copy()

        if raw_cube is None:
            if not quiet:
                print(f"Skip file because primary HDU has no data: {file}")
            continue

        cube = normalize_cube_order(raw_cube, fits_order=fits_order)
        if cube.ndim != 3:
            if not quiet:
                print(f"Skip file because data are not [x,y,wave]: {file}")
            continue

        nx, ny, _ = cube.shape
        annmask = make_annulus(nx, ny, geom.x0, geom.y0, geom.rlimb, geom.rmax)
        if despike:
            cube = smooth_cube(cube, annmask, smooth_size=smooth_size)

        inten, dop, wid = three_point(
            cube,
            wave,
            geom.bandpass,
            geom.x0,
            geom.y0,
            geom.rlimb,
            geom.rmax,
            quiet=quiet,
        )
        dop_corr, _dop_col_median = correct_doppler(
            dop,
            geom.x0,
            geom.y0,
            geom.rlimb,
            geom.rmax,
            dop_limit=dop_limit,
            poly_order=poly_order,
        )

        int_out = inten
        dop_out = dop_corr
        wid_out = wid

        if write_fits:
            datestr = datestr_from_header(header, file)
            int_file = int_dir / f"OYULO_SICG01_PKIM_L2_STP_{datestr}_V01.00_{expo}.FITS"
            dop_file = dop_dir / f"OYULO_SICG01_ECDG_L2_STP_{datestr}_V01.00_{expo}.FITS"
            wid_file = wid_dir / f"OYULO_SICG01_CLWM_L2_STP_{datestr}_V01.00_{expo}.FITS"

            write_one(int_file, inten, header, "Coronal Peak-Intensity Maps", "1.E-06 B/Bsun")
            write_one(dop_file, dop_corr, header, "Coronal Dopplergrams", "km/s")
            write_one(wid_file, wid, header, "Coronal Line-Width Maps", "Angstrom")

            if not quiet:
                print(f"Written: {int_file}")
                print(f"Written: {dop_file}")
                print(f"Written: {wid_file}")

    return int_out, dop_out, wid_out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """命令行参数解析。"""
    parser = argparse.ArgumentParser(
        description="Generate SICG L2 products from L1B/SPEI five-wavelength FITS files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--l1b-dir", type=str, help="包含 L1B/SPEI FITS 文件的目录。")
    input_group.add_argument(
        "--l1b-files",
        nargs="+",
        help="指定一个或多个输入 L1B FITS 文件；指定后会忽略 --l1b-dir。",
    )

    parser.add_argument("--outroot", type=str, default="./L2", help="L2 输出根目录，会在下面创建 INT/DOP/WID。")
    parser.add_argument("--wave", type=float, default=637.4, help="线中心，单位 nm；常用 637.4 或 530.3。")
    parser.add_argument("--x0", type=float, default=None, help="掩星盘中心 x 坐标，单位 pixel。")
    parser.add_argument("--y0", type=float, default=None, help="掩星盘中心 y 坐标，单位 pixel。")
    parser.add_argument("--rlimb", type=float, default=None, help="掩星盘半径 / 拟合内边界，单位 pixel。")
    parser.add_argument("--rmax", type=float, default=None, help="外拟合范围，外边界为 rlimb + rmax。")
    parser.add_argument("--bandpass", type=float, default=None, help="仪器 FWHM，单位 Angstrom；不填则按波长自动设置。")
    parser.add_argument("--dop-limit", type=float, default=200.0, help="列中值校正时允许的最大绝对速度，单位 km/s。")
    parser.add_argument("--poly-order", type=int, default=5, help="兼容原脚本参数；当前中值校正不使用。")
    parser.add_argument("--smooth-size", type=int, default=3, help="--despike 使用的平滑核大小，自动修正为奇数。")
    parser.add_argument("--write-fits", action="store_true", help="写出 L2 FITS 产品。")
    parser.add_argument("--despike", action="store_true", help="处理前对输入 cube 做简单邻域平均平滑。")
    parser.add_argument("--quiet", action="store_true", help="减少命令行输出。")
    parser.add_argument(
        "--fits-order",
        choices=["auto", "xyw", "wyx", "yxw"],
        default="auto",
        help="FITS 数据轴顺序。auto 会尝试自动识别 5 波长轴并转为 [x,y,wave]。",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。"""
    args = parse_args(argv)
    make_l2(
        l1b_dir=args.l1b_dir,
        l1b_files=args.l1b_files,
        outroot=args.outroot,
        wave=args.wave,
        x0=args.x0,
        y0=args.y0,
        rlimb=args.rlimb,
        rmax=args.rmax,
        bandpass=args.bandpass,
        dop_limit=args.dop_limit,
        poly_order=args.poly_order,
        smooth_size=args.smooth_size,
        write_fits=args.write_fits,
        despike=args.despike,
        quiet=args.quiet,
        fits_order=args.fits_order,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
