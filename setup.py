#!/usr/bin/env python3

from setuptools import setup, Distribution, Extension

version = '3.1.5'

define_macros=[
  ('PYIQE_VERSION', f'"{version}"'),
  ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
]

extra_compile_args=[
  '-std=c99',
  '-Wno-unused-but-set-variable',
  '-Wno-unused-variable',
]

setup(
  name='iqe',
  version=version,
  description='Image Quality Estimator',
  long_description="Plugin to ESO's skycat pick-object.",
  author='Matthieu Bec',
  author_email='mdcb808@gmail.com',
  url='https://github.com/mdcb/python-iqe',
  license='GNU General Public License',
  ext_modules=[
    Extension(
      name='iqe',
      sources = [
        'iqefunc.c',
        'mpfit.c',
        'python-iqe.c',
      ],
      include_dirs=['.'],
      define_macros=define_macros,
      libraries = ['m'],
      extra_compile_args = extra_compile_args,
    )
  ]
)

