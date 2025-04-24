# -*- coding: utf-8 -*-
# @Author  : LG

from pathlib import Path
from fastapi import Request
import json
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# 加载语言文件
LOCALES = {
    'en': json.loads(Path(BASE_DIR/'static'/'locales'/'en.json').read_text(encoding='utf-8')),
    'zh': json.loads(Path(BASE_DIR/'static'/'locales'/'zh.json').read_text(encoding='utf-8'))
}

def get_locale(request: Request):
    # 获取语言参数，默认英语
    lang = request.query_params.get('lang', 'en')
    return LOCALES.get(lang, 'en')

# 模版
templates = Jinja2Templates(directory=BASE_DIR / "templates")
