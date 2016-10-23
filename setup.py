from distutils.core import setup

import penfmb

setup(name='penfmb',
      version='0.1',
      description='penfmb',
      url='https://github.com/nuffe/penfmb',
      author='nuffe',
      license='MIT',
      py_modules=['penfmb'],
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=['arch, statsmodels, pandas'])
