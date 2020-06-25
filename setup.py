from pathlib import Path
from setuptools import setup, find_packages


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()


URL = 'https://unit8co.github.io/darts/'


PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/unit8co/darts/issues',
    'Documentation': URL,
    'Source Code': 'https://github.com/unit8co/darts'
}


setup(
      name='u8darts',
      version="0.2.0",
      description='A python library for easy manipulation and forecasting time series.',
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      project_urls=PROJECT_URLS,
      url=URL,
      maintainer='Unit8 SA',
      maintainer_email='darts@unit8.co',
      license='Apache License 2.0',
      packages=find_packages(),
      install_requires=read_requirements('requirements/main.txt'),
      zip_safe=False,
      python_requires='>=3.6',
      classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            ('Programming Language :: Python :: '
             'Implementation :: PyPy')
      ],
      keywords='timeseries forecasting arima'
)
