import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import os
from tqdm import tqdm


# 找到最近时间点
def find_nearest_time(time, time_list):
    closest_time = min(time_list, key=lambda x: abs(x - time))
    return closest_time


# 得到下载链接
def get_url(start, end, resolution, wavelength, instrument):
    base_url = "https://www.sidc.be/EUI/data/L2/"
    date_format = "%Y%m%dT%H%M%S"
    pattern = r"(\d{8})T(\d{6})"

    download_all = False
    if resolution == 0:
        download_all = True
        resolution = 86400

    # 得到时间点
    print('getting timepoints...')
    start = datetime.strptime(start, date_format)
    end = datetime.strptime(end, date_format)
    interval = timedelta(seconds=resolution)
    timepoints = []
    current = start
    while current <= end:
        timepoints.append(current)
        current += interval

    # 按照年月日分组
    time_dict = {}
    for time_point in timepoints:
        year_month_day = (time_point.year, time_point.month, time_point.day)
        if year_month_day not in time_dict:
            time_dict[year_month_day] = []
        time_dict[year_month_day].append(time_point)

    # 计算下载链接
    download_links = []

    for ymd, times in time_dict.items():
        url = base_url + "{:04d}/{:02d}/{:02d}/".format(ymd[0], ymd[1], ymd[2])
        response = requests.get(url)
        print('visit website successful!')
        soup = BeautifulSoup(response.text, 'html.parser')
        # 所有链接
        all_links = soup.find_all('a')
        # 初步筛选, 选择174或304波段, 排除包含"short"的链接
        links = []
        for href_link in all_links:
            link = href_link.get("href")
            if str(wavelength) in link and instrument in link and not ("short" in link):
                links.append(link)

        # 选择所需链接
        link_times = []
        link_dict = {}
        for link in links:
            match = re.search(pattern, link)
            if match:
                link_time = datetime.strptime(match.group(), date_format)
                link_times.append(link_time)
                if link_time not in link_dict:
                    link_dict[link_time] = link

        if download_all:
            begin_time = find_nearest_time(times[0], link_times)
            begin_index = link_times.index(begin_time)
            for i in range(begin_index, len(link_times)):
                if url + link_dict[link_times[i]] not in download_links and end > link_times[i]:
                    download_links.append(url + link_dict[link_times[i]])

        else:
            for time in times:
                closest_time = find_nearest_time(time, link_times)
                if url + link_dict[closest_time] not in download_links:
                    download_links.append(url + link_dict[closest_time])

    return download_links


# 下载所有数据
def download_eui_data(start, end, resolution, wavelength, instrument, filepath, overwrite=False, links_only=False):
    download_links = get_url(start, end, resolution, wavelength, instrument)
    if links_only:
        return download_links

    number = len(download_links)

    if number == 0:
        print("No files to download!")
        return -1

    print(f"{number} files needed to be downloaded")

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    files = []

    for i in range(0, len(download_links)):
        url = download_links[i]
        response = requests.get(url, stream=True)
        # 检查请求是否成功
        if response.status_code == 200:
            # 从URL中提取文件名
            filename = url.split("/")[-1]

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            print(f"file{i + 1}/{len(download_links)}, filename:{filename}")
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

            # 将文件内容保存到本地
            file_path = os.path.join(filepath, filename)
            if os.path.exists(file_path) and not overwrite:
                progress_bar.update(total_size_in_bytes)
            else:
                with open(file_path, 'wb') as file:
                    for data in response.iter_content(1024):  # 以1024字节的块读取内容
                        progress_bar.update(len(data))  # 更新进度条
                        file.write(data)  # 写入文件
            progress_bar.close()
            print("download successful!")
            files.append(filename)

        else:
            print(f"connection failed, response code: {response.status_code}")

    return len(files), download_links, files

# 函数调用
# download_eui_data(start, end, resolution, wavelength, instrument, filepath, overwrite=False, links_only=False)
# start, end: string. format: YYYYmmddTHHMMSS. example: 20231126T172800
# resolution: integer. time resolution, unit is second. if zero, will download all files between the time range
# wavelength: integer. 174 or 304
# instrument: string. hrieuv or fsi
# filepath: string. path to save the files.
# overwrite: bool. default as False. decide overwrite or not when the file already exist.
# links_only: bool. default as False. if True, the program will not download the files.
# 返回值: 若links_only = True, 只返回下载链接; 否则, 按顺序返回下载文件数目、下载链接、文件名
