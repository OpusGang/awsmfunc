from setuptools import setup

with open('requirements.txt') as fh:
    install_requires = fh.read()

setup(
    name='awsmfunc',
    version='1.2.0',
    url='https://github.com/OpusGang/awsmfunc',
    author='OpusGang',
    packages=['awsmfunc'],
    package_data={
        'awsmfunc': ['py.typed'],
    },
    install_requires=install_requires,
    zip_safe=False,
)
