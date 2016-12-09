# -*- coding: utf-8 -*-
from setuptools import find_packages
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
# import sphinx_pypi_upload
import numpy

setup(
    name='scikit-multilearn',
    version='0.0.3.89',
    packages=find_packages(exclude=['docs', 'tests', '*.tests']),
    author=u'Piotr Szymański',
    author_email='niedakh@gmail.com',
    license='BSD',
    #    long_description=open('README.md').read(),
    url='http://scikit-multilearn.github.io/',
    description='A set of python modules for multi-label classification',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    ext_modules=[
        Extension("skmultilearn.tree.pctcriterium",
                  ["skmultilearn/tree/pctcriterium.c"],
                  extra_compile_args=["-Zi", "/Od"],
                  include_dirs=[numpy.get_include()]),
        Extension("skmultilearn.tree.utils",
                  ["skmultilearn/tree/utils.c"],
                  extra_compile_args=["-Zi", "/Od"],
                  include_dirs=[numpy.get_include()]),
        Extension("skmultilearn.tree.pctsplitter",
                  ["skmultilearn/tree/pctsplitter.c"],
                  extra_compile_args=["-Zi", "/Od"],
                  include_dirs=[numpy.get_include()])
    ]
)
