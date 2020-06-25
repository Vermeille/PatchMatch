from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

setup(
    name='PatchMatch',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    packages=find_packages(),
    ext_modules=cythonize([
        Extension(
            "patchmatch.pm",
            ["patchmatch/pm.pyx"],
            extra_compile_args=['-O3'])
        ]),
    zip_safe=False,
    install_requires=['cython', 'numpy', 'pillow'],
)
