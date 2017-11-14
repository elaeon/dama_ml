# -*- coding: utf-8 -*-
#import re
from setuptools import find_packages, setup
from os.path import abspath, dirname, join

install_requires = [
    'matplotlib==1.5.1',
    'networkx==1.11',
    'numpy==1.12',
    'pandas==0.19.2',
    'Pillow==3.4.2',
    'scikit-image==0.12.3',
    'scikit-learn==0.18',
    'scipy==0.18.1',
    'tabulate==0.7.5',
    'tensorflow==1.0.1',
    'tflearn==0.2.1',
    'tqdm==4.11.2',
    'h5py==2.6.0',
    'GPy==1.5.6',
    'climin==0.1a1',
    'xmltodict==0.10.2',
    'dill==0.2.6.',
    'keras==2.0.2',
    'seaborn==0.7.1'
]

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.rst')) as f:
    long_description = f.read()

docs_require = [
    'sphinx>=1.4.0',
]

tests_require = [
    
]

setup(
    name='ml',
    version='0.4.0',
    description='A modern machine learning framework',
    long_description=long_description,
    author="Alejandro G. Martínez Romero",
    author_email="mara80@gmail.com",
    url='https://github.com/elaeon/ML',

    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={
        'docs': docs_require,
        'test': tests_require,
    },
    entry_points={
        'console_scripts': [
            'ml=ml.cli:main'
        ]
    },
    package_dir={'': 'src'},
    packages=find_packages('src', exclude=['docs', 'tests*']),  
    package_data = {'ml': ['data/settings.cfg.example'],},
    #include_package_data=True,

    license='Apache',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache 2 License',
        'Programming Language :: Python :: 2.7',
    ],
    zip_safe=False,
    keywords = 'ml dataset inference machine learning',
)


