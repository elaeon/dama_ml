from setuptools import find_packages, setup
from os.path import abspath, dirname, join


with open("requirements.txt") as f:
    req = f.read()
    install_requires = req.split(",")

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.rst')) as f:
    long_description = f.read()

docs_require = [
    'sphinx>=1.4.0',
]

tests_require = [
    
]

setup(
    name='soft stream',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description='A framework for building machine learning pipelines',
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
            'soft=ml.cli:main',
            'sst=ml.cli:main'
        ]
    },
    package_dir={'': 'src'},
    packages=find_packages('src', exclude=['docs', 'tests*']),  
    package_data = {'ml': ['config/settings.cfg.example'],},
    #include_package_data=True,

    license='Apache',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache 2 License',
        'Programming Language :: Python :: 3.5',
    ],
    zip_safe=False,
    keywords = 'pipeline stream dataset inference machine learning bigdata',
)


