# -*- coding: utf-8 -*-
# @Author  : LG

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from ISAT_SAM_BACKEND.config import get_locale, templates, CHECKPOINT_DIR
from ISAT_SAM_BACKEND.segment_any.model_zoo import model_dict
from ISAT_SAM_BACKEND.model import sam
import os.path


router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def index(request: Request, _: dict = Depends(get_locale)):

    return templates.TemplateResponse("pages/index.html", {
        "request": request,
        "_": _,
        "current_lang": request.query_params.get('lang', 'en'),
        "current_checkpoint": os.path.split(sam.segany.checkpoint)[-1] if sam.segany is not None else None,
    })

@router.get("/model", response_class=HTMLResponse)
async def model(request: Request, _: dict = Depends(get_locale)):
    model_dict_update = model_dict.copy()
    for model_name in model_dict_update:
        model_path = CHECKPOINT_DIR / f'{model_name}'
        model_dict_update[model_name]['downloaded'] = model_path.exists()

    return templates.TemplateResponse("pages/model.html", {
        "request": request,
        "_": _,
        "current_lang": request.query_params.get('lang', 'en'),
        "model_dict": model_dict_update
    })

@router.get("/description", response_class=HTMLResponse)
async def description(request: Request, _: dict = Depends(get_locale)):
    return templates.TemplateResponse("pages/description.html", {
        "request": request,
        "_": _,
        "current_lang": request.query_params.get('lang', 'en')
    })