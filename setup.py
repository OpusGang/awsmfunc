from setuptools import setup

setup(
    name='awsmfunc',
    version='1.3.1',
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
        'rekt @ git+https://github.com/OpusGang/rekt.git',
        'vsutil @ git+https://github.com/Irrational-Encoding-Wizardry/vsutil.git@956fa579406ca9edf6e0b6a834defae28efb51ce',
    ],
    zip_safe=False,
    python_requires='>=3.8',
)
