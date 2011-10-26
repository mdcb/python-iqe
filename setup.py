#!/usr/bin/env python
# setenv DISTUTILS_DEBUG

from numpy.distutils.core import setup, Extension
from os import environ as env

version = '1.1'                   # module version

undef_macros=[
   'PY_IQE_DEBUG'
  ]
 
define_macros=[
   ('PY_IQE_VERSION',          '\\\"%s\\\"' % version),
   ]

# override default OPT in /usr/lib/python*.*/config/Makefile
extra_compile_args=['-Wall']

setup(name="python-iqe",
      version=version,
      description="Image Quality routine",
      author="Matthieu Bec",
      author_email="mdcb808@gmail.com",
      url='http://mdcb.github.com/python-iqe',
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
            #include_dirs=[],
            #undef_macros=undef_macros,
            define_macros=define_macros,
            libraries = ['m'],
            #runtime_library_dirs = [string],
            #extra_objects = [string],
            extra_compile_args = extra_compile_args,
            #extra_link_args = ['-Wl,rpath %s/lib' % skycat],
            #export_symbols = [string],
            #depends = [string],
            #language = string,
            )
         ]
      )
     
