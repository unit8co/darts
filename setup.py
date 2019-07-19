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
          'dateutils',
          'scipy<1.3',
          'statsmodels==0.9.0',
          'pmdarima',
          'plotly',
          'fbprophet>=0.5',
          'scikit-learn',
          'pandas',
          'tqdm',
          'numpy'
      ],
      zip_safe=False)
