#!/usr/bin/env python3

from numpy.distutils.core import setup, Distribution, Extension

dist = Distribution()
dist.parse_config_files()
dist.parse_command_line()

release = dist.get_option_dict('bdist_rpm')['release'][1]
version = dist.get_option_dict('command')['version'][1]

define_macros=[
  ('PYIQE_VERSION', f'"{version}-{release}"'),
  ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
]

extra_compile_args=[
  '-Wno-unused-but-set-variable',
  '-Wno-unused-variable',
]

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

