import os
from setuptools import setup, find_packages
from pathlib import Path

requirements = {}
for extra in ["docs", "main"]:
    requirements[extra] = [
        r for r in Path(f"requirements/{extra}.txt").read_text().splitlines()
    ]

setup(name='u8timeseries',
      version=open("u8timeseries/VERSION", "r").read(),
      description='A collection of easy-to-use timeseries forecasting models',
      url='http://github.com/unit8co/u8timeseries',
      author='Unit8 SA',
      author_email='info@unit8.co',
      license='Apache License 2.0',
      packages=find_packages(),
<<<<<<< HEAD
      install_requires=requirements["main"],
      extras_require={"docs": requirements["docs"]},
=======
      install_requires=[
          'numpy>=1.18.1',
          'scipy>=1.4.1',
          'statsmodels>=0.11.1',
          'pmdarima>=1.5.3',
          'matplotlib>=3.2.1',
          'fbprophet>=0.5',
          'pandas>=0.23.1',
          'tqdm>=4.32.1'
      ],
>>>>>>> conda currently has numpy up to 1.18.1
      zip_safe=False,
      python_requires='>=3.6',
      package_data={'u8timeseries': ['VERSION']}
      )
