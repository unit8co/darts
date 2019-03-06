from setuptools import setup

setup(name='u8timeseries',
      version='0.1',
      description='A collection of easy-to-use timeseries forecasting models',
      url='http://github.com/unit8co/u8timeseries',
      author='Unit8',
      author_email='info@unit8.co',
      license='Apache License 2.0',
      packages=['u8timeseries'],
      install_requires=[
          'dateutils',
          'statsmodels',
          'pyramid-arima',
          'fbprophet',
          'scikit-learn',
          'pandas',
          'numpy'
      ],
      zip_safe=False)

