from setuptools import setup, find_packages

setup(name='u8timeseries',
      version='0.1',
      description='A collection of easy-to-use timeseries forecasting models',
      url='http://github.com/unit8co/u8timeseries',
      author='Unit8',
      author_email='info@unit8.co',
      license='Apache License 2.0',
      packages=find_packages(),
      install_requires=[
          'dateutils>=0.6.6',
          'scipy==1.2',  # latest cause issue with pmdarima imports
          'statsmodels>=0.9.0',
          'pmdarima',
          'fbprophet>=0.5',
          'scikit->=0.21.2',
          'pandas>=0.23.1',
          'tqdm>=4.32.1',
          'numpy>=1.15.4'
      ],
      zip_safe=False)
