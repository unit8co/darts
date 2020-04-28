from pathlib import Path

from setuptools import setup, find_packages

requirements = {}
for extra in ['docs', 'main', 'tests', 'examples', 'publish']:
    requirements[extra] = [
        r for r in Path(f'requirements/{extra}.txt').read_text().splitlines()
    ]
main_requirements = requirements.pop('main')


setup(
      name='u8timeseries',
      version=open('u8timeseries/VERSION', 'r').read(),
      description='A collection of easy-to-use timeseries forecasting models',
      url='http://github.com/unit8co/u8timeseries',
      author='Unit8 SA',
      author_email='info@unit8.co',
      license='Apache License 2.0',
      packages=find_packages(),
      install_requires=main_requirements,
      extras_require=requirements,
      zip_safe=False,
      python_requires='>=3.6',
      package_data={'u8timeseries': ['VERSION']}
)
