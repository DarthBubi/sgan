from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
  packages=['sgan'],
  scripts=['scripts/predict_tracks.py'],
  package_dir={'': 'src'}
)

setup(**d)
