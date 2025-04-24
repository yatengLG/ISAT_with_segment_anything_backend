# -*- coding: utf-8 -*-
# @Author  : LG
import os.path

from fastapi import APIRouter, Depends, Request, File, Form, HTTPException
from ISAT_SAM_BACKEND.model import sam
import numpy as np


router = APIRouter()

@router.post("/encode")
async def encode(file: bytes=File(...), shape: str=Form(...), dtype: str=Form(...)):
    if sam.segany is None:
        raise HTTPException(status_code=404, detail="No segany model")
    try:
        #
        shape = tuple(map(int, shape.split(',')))
        image_data = np.frombuffer(file, eval(f'np.{dtype}')).reshape(shape)
        # process
        features, original_size, input_size = sam.predict(image_data)

        return {
            "features": features,
            "original_size": original_size,
            "input_size": input_size
        }

    except Exception as e:
        raise e

@router.get("/info")
async def info():
    return {
        'checkpoint': os.path.split(sam.segany.checkpoint)[-1] if sam.segany is not None else None,
        'device': sam.segany.device if sam.segany is not None else None,
        'dtype': f'{sam.segany.model_dtype}' if sam.segany is not None else None
    }