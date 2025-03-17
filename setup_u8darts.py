from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


base_reqs = read_requirements("requirements/core.txt")
torch_reqs = read_requirements("requirements/torch.txt")
no_torch_reqs = read_requirements("requirements/notorch.txt")

all_reqs = base_reqs + torch_reqs + no_torch_reqs

with open("README.md") as fh:
    LONG_DESCRIPTION = fh.read()


URL = "https://unit8co.github.io/darts/"


PROJECT_URLS = {
    "Bug Tracker": "https://github.com/unit8co/darts/issues",
    "Documentation": URL,
    "Source Code": "https://github.com/unit8co/darts",
}


setup(
    name="u8darts",
    version="0.34.0",
    description="A python library for easy manipulation and forecasting of time series.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    project_urls=PROJECT_URLS,
    url=URL,
    maintainer="Unit8 SA",
    maintainer_email="darts@unit8.co",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=base_reqs,
    extras_require={"all": all_reqs, "torch": torch_reqs, "notorch": no_torch_reqs},
    package_data={
        "darts": ["py.typed"],
    },
    zip_safe=False,
    python_requires=">=3.9",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="time series forecasting",
)
