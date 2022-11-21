from pathlib import Path

from setuptools import setup

long_description = Path('README.md').read_text()

setup(
    name='awsmfunc',
    version='1.3.3',
    description='awesome VapourSynth functions',
    long_description_content_type="text/markdown",
    long_description=long_description,
    url='https://github.com/OpusGang/awsmfunc',
    license='MIT',
    author='OpusGang',
    packages=['awsmfunc', 'awsmfunc.types'],
    package_data={
        'awsmfunc': ['py.typed'],
    },
    install_requires=[
        'VapourSynth>=57',
        'numpy',
        'vs-rekt>=1.0.0',
        'vsutil>=0.7.0',
    ],
    zip_safe=False,
    python_requires='>=3.9',
)
