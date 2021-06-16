from setuptools import setup

with open("requirements.txt") as fh:
    install_requires = fh.read()

setup(
    name='awsmfunc',
    version='1.0.0',
    url='https://github.com/OpusGang/awsmfunc',
    author='OpusGang',
    packages=["awsmfunc"],
    install_requires=install_requires,
    zip_safe=False,
)
