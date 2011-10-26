#!/usr/bin/env python

from numpy.distutils.core import setup, Extension
from os import environ as env

version = '1.1'

undef_macros=[
   'PY_IQE_DEBUG'
  ]
 
define_macros=[
   ('PY_IQE_VERSION',          '\\\"%s\\\"' % version),
   ]

extra_compile_args=['-Wall']

setup(name='python-iqe',
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
     
