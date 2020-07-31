#!/usr/bin/env python3

from numpy.distutils.core import setup, Extension

version = '1.1'

undef_macros=[]
 
define_macros=[
  ('PYIQE_VERSION', f'"{version}"'),
  ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
]

extra_compile_args=['-Wno-maybe-uninitialized'] # ['-Wall','-pedantic']

setup(name='python3-iqe',
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
        'covsrt.c',
        'gaussj.c',
        'indexx.c',
        'iqefunc.c',
        'mrqfit.c',
        'sort.c',
        'python-iqe.c',
      ],
      define_macros=define_macros,
      libraries = ['m'],
      extra_compile_args = extra_compile_args,
    )
  ]
)
     
