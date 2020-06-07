from setuptools import setup

with open("requirements.txt") as fh:
    install_requires = fh.read()

setup(
    name='awsmfunc',
    version='0.2.0',
    url='https://github.com/OpusGang/awsmfunc',
    author='OpusGang',
    packages=["awsmfunc"],
    instal_requires=install_requires,
)