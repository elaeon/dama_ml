import re
from setuptools import find_packages, setup

install_requires = [
#    'GPy>=1.5.3',
    'matplotlib>=1.5.1',
    'networkx>=1.11',
    'numpy>=1.11.2',
    'pandas>=0.18.0',
    'Pillow==3.4.2',
    'scikit-image>=0.12.3',
    'scikit-learn>=0.18',
    'scipy>=0.18.1',
    'tabulate>=0.7.5',
    'tensorflow>=0.10.0',
    'tflearn>=0.2.1',
    'tqdm>=4.5.0',
    'h5py>=2.6.0'
]

docs_require = [
    'sphinx>=1.4.0',
]

tests_require = [
    
]

setup(
    name='ml',
    version='0.1.0',
    description='A modern machine learning framework',
    long_description='',
    author="Alejandro G. Martínez Romero",
    author_email="mara80@gmail.com",
    url='https://github.com/elaeon/ML',

    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={
        'docs': docs_require,
        'test': tests_require,
    },
    entry_points={},
    package_dir={'': 'src'},
    packages=find_packages('src'),
    include_package_data=True,

    license='Apache',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache 2 License',
        'Programming Language :: Python :: 2.7',
    ],
    zip_safe=False,
)


