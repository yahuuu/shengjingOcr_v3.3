##python setup_pyd.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
               Extension("billTitleOCR", ["billTitleOCR.py"]),
               
               
               ]

setup(
  name = 'service',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)


#setup(name='c',
#      ext_modules=[cythonize("c.py")])
