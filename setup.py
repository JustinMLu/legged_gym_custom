from setuptools import find_packages
from distutils.core import setup

setup(
    name='arcad_gym',
    version='1.0.0',
    author='Justin Lu',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='lujust@umich.edu',
    description='Custom Isaac Gym environments for Legged Robots',
    install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.21', 'tensorboard', 'mujoco==3.2.3', 'pyyaml']
    )
