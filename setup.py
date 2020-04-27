import os

import yaml
from setuptools import setup, find_packages


def read_requirements_from_conda():
    path = os.path.join('conda_recipe', 'u8timeseries', 'meta.yaml')
    with open(path, 'r') as stream:
        requirements = yaml.safe_load(stream)['requirements']['run']
        # remove the first python requirement
        requirements = requirements[1:]

        return requirements


setup(name='u8timeseries',
      version=open("u8timeseries/VERSION", "r").read(),
      description='A collection of easy-to-use timeseries forecasting models',
      url='http://github.com/unit8co/u8timeseries',
      author='Unit8 SA',
      author_email='info@unit8.co',
      license='Apache License 2.0',
      packages=find_packages(),
      install_requires=read_requirements_from_conda(),
      zip_safe=False,
      python_requires='>=3.6',
      package_data={'u8timeseries': ['VERSION']}
      )
