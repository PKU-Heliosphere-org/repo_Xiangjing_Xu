import requests
import os
import re
from tqdm.notebook import tqdm
from datetime import datetime


def download_file(url, output_path, overwrite=False):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filepath = os.path.join(output_path, url.split("/")[-1])

    try:
        if os.path.exists(filepath) and not overwrite:  # 检查文件完整性
            response = requests.head(url)
            response.raise_for_status()
            file_size = int(response.headers.get('Content-Length', 0))
            if os.path.getsize(filepath) == file_size:
                return
            else:
                print(f"文件不完整，重新下载: {filepath}")

        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('Content-Length', 0))
        with open(filepath, 'wb') as file, tqdm(
            desc=url.split("/")[-1]+'\n',
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=False
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))

    except requests.exceptions.RequestException as e:
        print(f"下载文件 {url} 失败: {e}")


def extract_datetime(file_name, pattern=r'\d{8}T\d{6}', time_format='%Y%m%dT%H%M%S'):
    match = re.search(pattern=pattern, string=file_name)
    if match:
        return datetime.strptime(match.group(), time_format)
    else:
        return datetime.min


def check_complementary(links, output_path):
    complete = True

    for link in tqdm(links, desc='检查文件完整性'):
        filepath = os.path.join(output_path, link.split("/")[-1])
        response = requests.head(link)
        response.raise_for_status()
        file_size = int(response.headers.get('Content-Length', 0))
        if os.path.exists(filepath):
            if os.path.getsize(filepath) == file_size:
                continue
            else:
                print(f'文件{link.split("/")[-1]}不完整，重新下载')
        else:
            print(f'文件{link.split("/")[-1]}不存在，重新下载')

        download_file(link, output_path, overwrite=True)
        complete = False

    return complete


def download_links(links, output_path, overwrite=False):
    links = sorted(links, key=extract_datetime)
    for link in tqdm(links, desc='downloading'):
        download_file(link, output_path, overwrite)

    while not check_complementary(links, output_path):
        check_complementary(links, output_path)

    print(f'successfully download {len(links)} files to {output_path}!')
