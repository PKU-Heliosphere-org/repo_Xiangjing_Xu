from pathlib import Path
import argparse
from astropy.io import fits
import numpy as np
import re
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
from sunpy.coordinates import frames
from typing import Dict, Any

def _mean_header_number(value):
    """
    对关键字CRPIX/CRVAL进行处理，修整为单一数值形式
    :param value: 头信息中关键字对应的值
    :return: 修正后关键字对应的值
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if not numbers:
        raise ValueError(f"No numeric value found in header value: {value!r}")
    return float(np.mean([float(number) for number in numbers]))

def _fix_l2_header(hdr: Dict[str, Any]) -> Dict[str, Any]:
    """
    修正L2fits文件投信息，包括修正CRPIX/CRVAL/CDELT形式、添加CUNIT和观察者信息以符合astropy的fits头文件标准
    :param hdr: 待修复的头信息
    :return: 修复后的头信息
    """
    # 修复CRPIX/CRVAL/CDELT，采用列表值平均
    wcs_scalar_keys = ["CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "CRVAL1", "CRVAL2"]
    for key in wcs_scalar_keys:
        hdr[key] = _mean_header_number(hdr[key])

    # 补充CUNIT信息，单位为arcsec
    hdr["CUNIT1"] = ("arcsec", "Image coordinate unit for axis 1")
    hdr["CUNIT2"] = ("arcsec", "Image coordinate unit for axis 2")

    # 修正CTYPE信息为日心投影坐标系
    hdr["CTYPE1"] = ("HPLN-TAN", "Helioprojective longitude")
    hdr["CTYPE2"] = ("HPLT-TAN", "Helioprojective latitude")

    # 补充观察者信息，包括HGLN_OBS/HGLT_OBS/DSUN_OBS/CRLN_OBS/CRLT_OBS，从台站的地球坐标自动计算
    obstime = Time(hdr["DATE-OBS"])
    location = EarthLocation(
        lon=float(hdr["DEV_LON"]) * u.deg,
        lat=float(hdr["DEV_LAT"]) * u.deg,
        height=float(hdr.get("DEV_ALT", 0.0)) * u.m,
    )

    observer_itrs = location.get_itrs(obstime=obstime)
    observer_hgs = SkyCoord(observer_itrs).transform_to(
        frames.HeliographicStonyhurst(obstime=obstime)
    )
    observer_hgc = observer_hgs.transform_to(
        frames.HeliographicCarrington(obstime=obstime, observer="self")
    )

    hdr["HGLN_OBS"] = (observer_hgs.lon.to_value(u.deg), "observer Stonyhurst longitude [deg]")
    hdr["HGLT_OBS"] = (observer_hgs.lat.to_value(u.deg), "observer Stonyhurst latitude [deg]")
    hdr["DSUN_OBS"] = (observer_hgs.radius.to_value(u.m), "observer distance from Sun center [m]")
    hdr["CRLN_OBS"] = (observer_hgc.lon.to_value(u.deg), "observer Carrington longitude [deg]")
    hdr["CRLT_OBS"] = (observer_hgc.lat.to_value(u.deg), "observer Carrington latitude [deg]")

    # 修正RSUN为R_SUN_OBS
    hdr["RSUN_OBS"] = (float(hdr["RSUN"]), "apparent solar radius [arcsec]")

    # 添加RSUN_REF
    hdr["RSUN_REF"] = (695700000.0, "reference solar radius [m]")

    # 修正TEMP关键字的非ASCII注释
    hdr["TEMP"] = (hdr["TEMP"], "[Celsius] ambient temperature")

    return hdr

def sicg_fix_l2(input_root: str, output_root: str='./L2_header_fixed/') ->None:
    """
    修正L2fits文件投信息，包括修正CRPIX/CRVAL/CDELT形式、添加CUNIT和观察者信息以符合astropy的fits头文件标准
    :param input_root: 输入的L2fits文件根目录
    :param output_root: 输出的fits文件根目录
    :return: None
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    fits_paths = sorted(path for path in input_root.rglob("*") if path.suffix.lower() == ".fits")
    if not fits_paths:
        raise FileNotFoundError(f"No FITS files found under {input_root}")

    for src in fits_paths:
        dst = output_root / src.relative_to(input_root)
        dst.parent.mkdir(parents=True, exist_ok=True)

        with fits.open(src, memmap=False) as hdul:
            hdul[0].header = _fix_l2_header(hdul[0].header)
            hdul.writeto(dst, overwrite=True, output_verify="silentfix")

    print(f"Fixed {len(fits_paths)} FITS files")
    print(f"Output directory: {output_root}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fix SICG L2 FITS headers for SunPy compatibility."
    )
    parser.add_argument(
        "input_root",
        help="Input root directory containing SICG L2 FITS files.",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        default="./L2_header_fixed/",
        help="Output root directory. Default: ./L2_header_fixed/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sicg_fix_l2(args.input_root, args.output_root)
