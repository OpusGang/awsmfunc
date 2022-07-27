from setuptools import setup

setup(
    name='awsmfunc',
    version='1.3.2',
    description='awesome VapourSynth functions',
    url='https://github.com/OpusGang/awsmfunc',
    license='MIT',
    author='OpusGang',
    packages=['awsmfunc', 'awsmfunc.types'],
    package_data={
        'awsmfunc': ['py.typed'],
    },
    install_requires=[
        'VapourSynth>=57',
        'vs-rekt>=1.0.0',
        'vsutil==0.7.0',
    ],
    zip_safe=False,
    python_requires='>=3.8',
)
