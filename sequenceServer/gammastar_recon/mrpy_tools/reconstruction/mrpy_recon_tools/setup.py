"""!
@brief Setup file for mrpy_recon_tools
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent

setup(
    name='mrpy_recon_tools',
    long_description='MRPy Reconstruction Tools',
    url='',
    author='Joern Huber, Vincent Kuhlen, Tom Luetjen',
    author_email='joern.huber@mevis.fraunhofer.de',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['numpy', 'pymapvbvd', 'matplotlib', 'pydicom', 'sigpy', 'xmltodict', 'ismrmrd', 'mri-nufft[finufft]', 'scikit-learn'],
    version='1.3',
    description='Python tools collecting basic MRI reconstruction algorithms',
)