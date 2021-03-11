#!/usr/bin/env python
from setuptools import setup

setup(
   name='asplstm',
   version='1.0',
   description='Inno LSTM V1',
   author='Ng Jian Lai',
   author_email='ngjianlai.jeph@gmail.com',
   packages=['asplstm'],  #same as name
   install_requires=['pandas == 0.24.1', 'numpy == 1.18.1','matplotlib == 3.3.2','seaborn == 0.11.0','datetime','csv == 1.0','json == 2.0.9','argparse == 1.1','parser','tf == 2.1.0','re','sklearn']
)