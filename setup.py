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
      install_requires=requirements["main"],
      extras_require={"docs": requirements["docs"]},
      zip_safe=False,
      python_requires='>=3.6',
      package_data={'u8timeseries': ['VERSION']}
      )
