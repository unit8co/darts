from setuptools import setup, find_packages

setup(name='u8timeseries',
      version='0.0.1',
      description='A collection of easy-to-use timeseries forecasting models',
      url='http://github.com/unit8co/u8timeseries',
      author='Unit8 SA',
      author_email='info@unit8.co',
      license='Apache License 2.0',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.15.4',
          'dateutils',
          'scipy<1.3',
          'statsmodels==0.9.0',
          'pmdarima<1.3',
          'plotly',
          'fbprophet>=0.5',
          'scikit-learn>=0.21.2',
          'pandas>=0.23.1',
          'tqdm>=4.32.1',
          'ipywidgets'
      ],
      zip_safe=False,
      python_requires='>=3.6'
      )
