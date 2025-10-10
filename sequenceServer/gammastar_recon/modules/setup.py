"""!
@brief Setup file for modules
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent

setup(
    name='gammastar_recon_modules',
    url='',
    author='Joern Huber',
    author_email='joern.huber@mevis.fraunhofer.de',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['mri-nufft[finufft]'],
    version='1.0.1',
    description='Modules of gammaSTAR reconstruction',
)