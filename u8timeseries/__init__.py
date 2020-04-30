import os

path = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(path, "r").read()
