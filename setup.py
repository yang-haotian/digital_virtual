# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 19:51
# @Project : AnimateMaster
# @FileName: setup.py

from setuptools import setup

setup(
    name='animate_master',
    version='1.0.0',
    description='Animate Master',
    author='wenshao',
    author_email='wenshaoguo1026@gmail.com',
    packages=[
        'animate_master',
        'animate_master.models',
        'animate_master.pipelines',
        'animate_master.common',
        'animate_master.infer_models'
    ],
    install_requires=[],
    data_files=[]
)