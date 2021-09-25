# @Author            : FederalLab
# @Date              : 2021-09-26 00:35:18
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:35:18
# Copyright (c) FederalLab. All rights reserved.
"""python setup.py sdist bdist_wheel python -m twine upload dist/*"""

from openfed import __version__
from setuptools import find_packages, setup

with open('requirements.txt', 'r', encoding='utf-8') as fh:
    install_requires = fh.read()

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    install_requires=install_requires,
    name='benchmark',
    version=__version__,
    author='FederalLab',
    author_email='densechen@foxmail.com',
    description='A high speed simulation benchmark of federated learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/FederalLab/benchmark-lightly',
    download_url=
    'https://github.com/FederalLab/benchmark-lightly/archive/main.zip',
    packages=find_packages(),
    # https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Environment :: GPU :: NVIDIA CUDA :: 10.2',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Documentation :: Sphinx',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Security',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='MIT License',
    python_requires='>=3.7',
)
