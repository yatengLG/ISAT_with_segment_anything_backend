# -*- coding: utf-8 -*-
# @Author  : LG

from ISAT_SAM_BACKEND.segment_any.model_zoo import model_dict
from tabulate import tabulate
from urllib import request
import time
import shutil
import tqdm
from pathlib import Path


def list_model(checkpoints_dir):
    data = []
    for model_name in model_dict.keys():
        checkpoints_file = checkpoints_dir / f'{model_name}'
        data.append([
            model_name,
            model_dict.get(model_name, {}).get('memory', None),
            model_dict.get(model_name, {}).get('bf16_memory', None),
            model_dict.get(model_name, {}).get('params', None),
            checkpoints_file.exists()
        ])

    print(tabulate(data, headers=['Model name', 'Memory', 'Memory(bf16)' ,'Disk', 'Downloaded'], tablefmt='github'))
    return

def download_model(model_name, checkpoints_dir):
    checkpoints_dir = Path(checkpoints_dir)
    if model_name not in model_dict:
        raise Exception(f'Model {model_name} not found.')

    info_dict = model_dict.get(model_name, {})
    urls = info_dict.get('urls', None)
    block_size = 4096

    tmp_dir = checkpoints_dir / 'tmp'
    if not tmp_dir.exists():
        tmp_dir.mkdir()
    tmp_file = checkpoints_dir / 'tmp' / f'{model_name}'
    save_file = checkpoints_dir / f'{model_name}'
    if save_file.exists():
        print(f'Model {model_name} already exists.')
        return

    # 寻找最佳下载链接
    print(f'Checking best download url for {model_name}')

    best_time = 1e8
    best_url = urls[0]
    for url in urls:
        try:
            start_time = time.time()
            req = request.Request(url, headers={"Range": "bytes=0-10"})
            request.urlopen(req, timeout=5)
            cost_time = time.time() - start_time
        except:
            cost_time = 1e8
        if cost_time < best_time:
            best_time = cost_time
            best_url = url

    print(f'Download {model_name} from {best_url}')

    # 检查缓存

    downloaded_size = 0
    if tmp_file.exists():
        with open(tmp_file, 'rb') as f:
            downloaded_size = len(f.read())

    req = request.Request(best_url, headers={"Range": "bytes=0-"})
    try:
        response = request.urlopen(req, timeout=10)
        total_size = int(response.headers['Content-Length'])
    except Exception as e:
        raise RuntimeError('When download {} from {}, {}'.format(model_name, best_url, e))

    # 存在缓存
    if downloaded_size != 0:
        # 判断缓存文件是否下载完
        if downloaded_size >= total_size:
            try:
                shutil.move(tmp_file, save_file)
                return
            except Exception as e:
                raise RuntimeError('Error when move {} to {}, {}'.format(tmp_file, save_file, e))

        # 断点续传
        content_range = response.headers.get('Content-Range', None)
        if content_range is not None:
            req = request.Request(best_url, headers={"Range": "bytes={}-".format(downloaded_size)})
            response = request.urlopen(req)
            content_range = response.headers.get('Content-Range', None)
            if content_range is not None:
                print('Resume downloading: ', content_range)
        else:
            print('Not supprot resume download.')
            downloaded_size = 0

    open_mode = 'wb' if downloaded_size == 0 else 'ab'
    with open(tmp_file, open_mode) as f:
        bar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True)
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            f.write(buffer)
            downloaded_size += len(buffer)
            bar.update(len(buffer))

    try:
        shutil.move(tmp_file, save_file)
    except Exception as e:
        raise RuntimeError('Error when move {} to {}, {}'.format(tmp_file, save_file, e))
    return

def remove_model(model_name, checkpoints_dir):
    model_file = Path(checkpoints_dir / f'{model_name}')
    if model_file.exists():
        try:
            model_file.unlink()
            print(f'Removed {model_name} finished.')
        except Exception as e:
            raise RuntimeError(f'Error when remove {model_name}: {e}')
    else:
        raise FileExistsError(f'Model {model_name} not found.')
    return
