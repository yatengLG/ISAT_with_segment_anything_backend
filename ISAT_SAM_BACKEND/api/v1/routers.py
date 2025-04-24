# -*- coding: utf-8 -*-
# @Author  : LG

from fastapi import APIRouter
from ISAT_SAM_BACKEND.api.v1.endpoints import pages, api

api_router = APIRouter()
api_router.include_router(pages.router, tags=["pages"])
api_router.include_router(api.router, prefix='/api', tags=["api"])