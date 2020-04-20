from setuptools import setup, find_packages

setup(name='u8timeseries',
      version='0.0.2',
      description='A collection of easy-to-use timeseries forecasting models',
      url='http://github.com/unit8co/u8timeseries',
      author='Unit8 SA',
      author_email='info@unit8.co',
      license='Apache License 2.0',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.18.2',
          'dateutils>=0.6.8',
          'scipy>=1.4.1',
          'statsmodels>=0.11.1',
          'pmdarima>=1.5.3',
          'plotly>=4.6.0',
          'matplotlib>=3.2.1',
          'fbprophet>=0.5',
          'scikit-learn>=0.21.2',
          'pandas>=0.23.1',
          'tqdm>=4.32.1',
          'ipywidgets'
      ],
      zip_safe=False,
      python_requires='>=3.6'
      )
