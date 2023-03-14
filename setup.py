# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
# import sphinx_pypi_upload
import sys

if sys.version_info[0] < 3:
    import codecs

    with codecs.open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()

else:
    import io

    with io.open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()

setup(
    name='scikit-multilearn',
    version='0.2.0',
    packages=find_packages(exclude=['docs', 'tests', '*.tests']),
    install_requires=[
        "scipy>=1.1.0",
        "numpy>=1.15.1",
        "liac-arff>=2.2.1",
        "networkx>=2.1",
        "python-louvain>=0.11",
        "future>=0.16.0",
        "scikit_learn>=0.19.2",
        "requests>=2.18.4",
    ],
    extras_require={
        "gpl":  [
            "python-igraph>=0.7.1.post6",
        ],
        "dev":  [
            "pytest>=3.3.1",
            "PyHamcrest>=1.9.0",
            "ipython",
            "ipython-genutils",
        ],
        "keras": [
            "keras",
            "tensorflow",
        ],
        "meka": [
            "whichcraft>=0.4.1",
        ]
    },
    author=u'Piotr Szymański',
    author_email=u'niedakh@gmail.com',
    license=u'BSD',
    long_description=readme,
    url=u'http://scikit.ml/',
    description=u'Scikit-multilearn is a BSD-licensed library for multi-label classification that is built on top of the well-known scikit-learn ecosystem.',
    classifiers=[
        u'Development Status :: 5 - Production/Stable',
        u'Environment :: Console',
        u'Environment :: Web Environment',
        u'Intended Audience :: Developers',
        u'Intended Audience :: Education',
        u'Intended Audience :: Science/Research',
        u'License :: OSI Approved :: BSD License',
        u'Operating System :: MacOS :: MacOS X',
        u'Operating System :: Microsoft :: Windows',
        u'Operating System :: POSIX',
        u'Programming Language :: Python',
        u'Topic :: Scientific/Engineering',
        u'Topic :: Scientific/Engineering :: Information Analysis',
        u'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
