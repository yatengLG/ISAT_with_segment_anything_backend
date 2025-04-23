# -*- coding: utf-8 -*-
# @Author  : LG

import torch
from fastapi import FastAPI, Request, File, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import json
from pathlib import Path
from tabulate import tabulate
from ISAT_SAM_BACKEND.segment_any.segment_any import SegAny
from ISAT_SAM_BACKEND.segment_any.model_zoo import model_dict
import os.path
import argparse
from urllib import request
import time
import shutil
import tqdm
import sys


app = FastAPI()
segany:SegAny = None
templates = Jinja2Templates(directory="ISAT_SAM_BACKEND/templates")
app.mount("/static", StaticFiles(directory="ISAT_SAM_BACKEND/static"), name="static")

BASE_DIR = Path(__file__).resolve().parent

# 加载语言文件
LOCALES = {
    'en': json.loads(Path(BASE_DIR/'static'/'locales'/'en.json').read_text(encoding='utf-8')),
    'zh': json.loads(Path(BASE_DIR/'static'/'locales'/'zh.json').read_text(encoding='utf-8'))
}

def get_locale(request: Request):
    # 获取语言参数，默认英语
    lang = request.query_params.get('lang', 'en')
    return LOCALES.get(lang, 'en')


# init sam
def sam_init(model_name='mobile_sam.pt', use_bfloat16=True):
    if not torch.cuda.is_available():
        use_bfloat16 = False
    segany = SegAny(str(BASE_DIR/'checkpoints'/f'{model_name}'), use_bfloat16=use_bfloat16)
    return segany


def list_model():
    checkpoints_dir = BASE_DIR/'checkpoints'
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

def download_model(model_name):
    if model_name not in model_dict:
        raise Exception(f'Model {model_name} not found.')

    info_dict = model_dict.get(model_name, {})
    urls = info_dict.get('urls', None)
    block_size = 4096

    tmp_dir = BASE_DIR / 'checkpoints' / 'tmp'
    if not tmp_dir.exists():
        tmp_dir.mkdir()
    tmp_file = BASE_DIR / 'checkpoints' / 'tmp' / f'{model_name}'
    save_file = BASE_DIR / 'checkpoints' / f'{model_name}'
    if save_file.exists():
        print(f'Model {model_name} already exists.')
        return

    # 寻找最佳下载链接
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
        print('When download {} from {}, {}'.format(model_name, best_url, e))
        return
    # 存在缓存
    if downloaded_size != 0:
        # 判断缓存文件是否下载完
        if downloaded_size >= total_size:
            try:
                shutil.move(tmp_file, save_file)
                return
            except Exception as e:
                print('Error when move {} to {}, {}'.format(tmp_file, save_file, e))

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
        return
    except Exception as e:
        print('Error when move {} to {}, {}'.format(tmp_file, save_file, e))
    return

def remove_model(model_name):
    model_file = BASE_DIR / 'checkpoints' / f'{model_name}'
    if model_file.exists():
        try:
            model_file.unlink()
            print(f'Removed {model_name} finished.')
        except Exception as e:
            print(f'Error when remove {model_name}: {e}')
    else:
        print(f'Model {model_name} not found.')
    return

@torch.no_grad()
async def sam_encode(image: np.ndarray):
    with torch.inference_mode(), torch.autocast(segany.device,
                                                dtype=segany.model_dtype,
                                                enabled=torch.cuda.is_available()):
        if 'sam2' in segany.model_type:
            _orig_hw = tuple([image.shape[:2]])
            input_image = segany.predictor_with_point_prompt._transforms(image)
            input_image = input_image[None, ...].to(segany.predictor_with_point_prompt.device)
            backbone_out = segany.predictor_with_point_prompt.model.forward_image(input_image)
            _, vision_feats, _, _ = segany.predictor_with_point_prompt.model._prepare_backbone_features(
                backbone_out)
            if segany.predictor_with_point_prompt.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[
                                       -1] + segany.predictor_with_point_prompt.model.no_mem_embed
            feats = [
                        feat.permute(1, 2, 0).view(1, -1, *feat_size)
                        for feat, feat_size in
                        zip(vision_feats[::-1], segany.predictor_with_point_prompt._bb_feat_sizes[::-1])
                    ][::-1]
            _features = {"image_embed": feats[-1], "high_res_feats": tuple(feats[:-1])}
            return _features, _orig_hw, _orig_hw
        else:
            input_image = segany.predictor_with_point_prompt.transform.apply_image(image)
            input_image_torch = torch.as_tensor(input_image,
                                                device=segany.predictor_with_point_prompt.device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            original_size = image.shape[:2]
            input_size = tuple(input_image_torch.shape[-2:])

            input_image = segany.predictor_with_point_prompt.model.preprocess(input_image_torch)
            features = segany.predictor_with_point_prompt.model.image_encoder(input_image)
        return features, original_size, input_size


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, _: dict = Depends(get_locale)):

    return templates.TemplateResponse("index.html", {
        "request": request,
        "_": _,
        "current_lang": request.query_params.get('lang', 'en'),
        "current_checkpoint": os.path.split(segany.checkpoint)[-1]
    })

@app.get("/model", response_class=HTMLResponse)
async def model(request: Request, _: dict = Depends(get_locale)):
    model_dict_update = model_dict.copy()
    for model_name in model_dict_update:
        model_dict_update[model_name]['downloaded'] = os.path.exists(f'checkpoints/{model_name}')

    return templates.TemplateResponse("model.html", {
        "request": request,
        "_": _,
        "current_lang": request.query_params.get('lang', 'en'),
        "model_dict": model_dict_update
    })

@app.get("/api", response_class=HTMLResponse)
async def api(request: Request, _: dict = Depends(get_locale)):
    # todo 补充页面
    return templates.TemplateResponse("api.html", {
        "request": request,
        "_": _,
        "current_lang": request.query_params.get('lang', 'en')
    })

@app.post("/encode")
async def encode(file: bytes=File(...), shape: str=Form(...), dtype: str=Form(...)):
    try:
        #
        shape = tuple(map(int, shape.split(',')))
        image_data = np.frombuffer(file, eval(f'np.{dtype}')).reshape(shape)
        # process
        features, original_size, input_size = await sam_encode(image_data)
        if segany.model_source == 'sam_hq':
            # sam_hq features is a tuple. include features: Tensor and interm_features: List[Tensor, ...]
            features, interm_features = features

            features = features.detach().to(torch.float32).cpu().numpy().tolist()
            interm_features = [interm_feature.detach().to(torch.float32).cpu().numpy().tolist() for interm_feature in interm_features]
            features = (features, interm_features)

        elif 'sam2' in segany.model_source:
            # sam2 features is a dict. include image_embed: Tensor and high_res_feats: Tuple[Tensor, ...]
            image_embed = features['image_embed']
            high_res_feats = features['high_res_feats']

            image_embed = image_embed.detach().to(torch.float32).cpu().numpy().tolist()
            high_res_feats = [high_res_feat.detach().to(torch.float32).cpu().numpy().tolist() for high_res_feat in high_res_feats]
            features['image_embed'] = image_embed
            features['high_res_feats'] = high_res_feats
        else:
            features = features.detach().cpu().numpy().tolist()

        print('features', sys.getsizeof(features))
        return {
            "features": features,
            "original_size": original_size,
            "input_size": input_size
        }

    except Exception as e:
        raise e

@app.get("/info")
async def info():
    return {
        'checkpoint': segany.checkpoint if isinstance(segany, SegAny) else None,
        'device': segany.device if isinstance(segany, SegAny) else None,
        'dtype': f'{segany.model_dtype}' if isinstance(segany, SegAny) else None
    }

def main():
    parser = argparse.ArgumentParser(description="ISAT SAM backend, supporting ISAT use a remote server for SAM encoding.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_run = subparsers.add_parser("run", description="ISAT SAM backend, run server")
    parser_run.add_argument("--checkpoint", type=str, default="mobile_sam.pt", help="SAM checkpoint name")
    parser_run.add_argument("--host", type=str, default="127.0.0.1", help="Ip")
    parser_run.add_argument("--port", type=int, default=8000, help="Port")
    parser_run.add_argument("--workers", type=int, default=1, help="Num of workers")

    parser_model = subparsers.add_parser("model", description="ISAT SAM backend model manage")
    model_group = parser_model.add_mutually_exclusive_group(required=True)      # 互斥组
    model_group.add_argument("--list", action="store_true", help="List model")
    model_group.add_argument("--download", type=str, help="The model name will download")
    model_group.add_argument("--remove", type=str, help="The model name will remove")

    args = parser.parse_args()

    if args.command == "run":
        checkpoint = args.checkpoint
        if not os.path.exists(BASE_DIR/'checkpoints'/f'{checkpoint}'):
            raise FileExistsError(checkpoint)

        # init sam
        global segany
        try:
            segany = sam_init(args.checkpoint)
        except Exception as e:
            print(f"init sam failed: {e}")

        # start server
        uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)

    elif args.command == "model":
        if args.list:
            list_model()
        elif args.download:
            download_model(args.download)
        elif args.remove:
            remove_model(args.remove)


if __name__ == '__main__':
    main()
