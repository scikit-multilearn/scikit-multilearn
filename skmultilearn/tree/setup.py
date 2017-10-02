try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
ext_modules=[
    Extension("pctcriterium",
              ["pctcriterium.c"],
              extra_compile_args=["-Zi", "/Od"],
              extra_link_args=["-debug"],
              include_dirs = [numpy.get_include()]),
    Extension("utils",
              ["utils.c"],
              extra_compile_args=["-Zi", "/Od"],
              extra_link_args=["-debug"],
              include_dirs=[numpy.get_include()]),
    Extension("pctsplitter",
              ["pctsplitter.c"],
              extra_compile_args=["-Zi", "/Od"],
              extra_link_args=["-debug"],
              include_dirs=[numpy.get_include()]),
]
setup(
    ext_modules = ext_modules,
)