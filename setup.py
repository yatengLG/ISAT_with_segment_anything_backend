# -*- coding: utf-8 -*-
# @Author  : LG

import os
import codecs
from setuptools import setup, find_packages


def get_version():
    try:
        from ISAT_SAM_BACKEND.__init__ import __version__
        return __version__

    except FileExistsError:
        FileExistsError('ISAT_SAM_BACKEND.__init__.py not exists.')

setup(
    name="isat-sam-backend",                                        # 包名
    version=get_version(),                                  # 版本号
    author="yatengLG",
    author_email="yatenglg@foxmail.com",
    description="ISAT SAM encoding backend.",
    long_description=(codecs.open("README.md", encoding='utf-8').read()),
    long_description_content_type="text/markdown",

    url="https://github.com/yatengLG/ISAT_with_segment_anything",  # 项目相关文件地址

    keywords=["annotation tool", "segment anything", "image annotation", "semantic segmentation", 'instance segmentation'],
    license="Apache2.0",

    packages=find_packages(),
    package_dir={"ISAT_SAM_BACKEND": "ISAT_SAM_BACKEND"},
    package_data={'ISAT_SAM_BACKEND': ['api/**',
                                       'checkpoints/mobile_sam.pt',
                                       'segment_any/**',
                                       'services/**'
                                       'static/**',
                                       'templates/**',
                                       'utils/**',
                                       ]},
    exclude_package_data={'': ['test.py']},

    include_package_data=True,

    python_requires=">=3.8",                            # python 版本要求
    install_requires=[
        'fastapi',
        'python-multipart',
        'uvicorn',
        'torch>=2.1.1',
        'torchvision',
        'timm',
        'hydra-core>=1.3.2',
        'tqdm>=4.66.1',
        'iopath',
        'tqdm',
        'tabulate'
        ],

    classifiers=[
        "Intended Audience :: Developers",              # 目标用户:开发者
        "Intended Audience :: Science/Research",        # 目标用户:学者
        'Development Status :: 5 - Production/Stable',
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
        'License :: OSI Approved :: Apache Software License',

        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "isat-sam-backend=ISAT_SAM_BACKEND.main:main",
        ],
    },
)
