from distutils.core import setup

setup(
    name='scikit-multilearn',
    version='0.0.1',
    packages=['skmultilearn','meka',],
    author=u'Piotr Szymański',
    author_email='niedakh@gmail.com',
    license='GPL',
    long_description=open('README.md').read(),
    url='http://scikit-multilearn.github.io/scikit-multilearn/',
    description= 'A set of python modules for multi-label classification',
)