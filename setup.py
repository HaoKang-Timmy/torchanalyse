
from setuptools import find_packages, setup

from torchanalyse import __version__

setup(
    name='torchanalyse',
    version=__version__,
    packages=find_packages(exclude=['examples']),
    install_requires=[
        'numpy>=1.14',
        'torch>=1.4',
        'torchvision>=0.4',
        'pandas>=2.0.0'
    ],
    url='https://github.com/HaoKang-Timmy/torchanalyse',
    # license='MIT',
)
