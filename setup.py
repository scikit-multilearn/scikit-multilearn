# -*- coding: utf-8 -*-
from distutils.core import setup

setup(
    name='scikit-multilearn',
    version='0.0.1',
    packages=['skmultilearn','meka', 'data'],
    author=u'Piotr Szymański',
    author_email='niedakh@gmail.com',
    license='BSD',
    long_description=open('README.md').read(),
    url='http://scikit-multilearn.github.io/',
    description= 'A set of python modules for multi-label classification',
    package_data = {'data': ['*.dump.bz2']},
    include_package_data = True,
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
)
