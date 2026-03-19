# Available at setup time due to pyproject.toml
import glob

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension(
        "darts._internal",
        sources=glob.glob("src/*.cpp"),
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
)
