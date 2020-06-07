from setuptools import setup

with open("requirements.txt") as fh:
    install_requires = fh.read()

setup(
    name='awsmfunc',
    version='0.2.0',
    url='https://git.concertos.live/AHD/awsmfunc',
    author='AHD',
    packages=["awsmfunc"],
    instal_requires=install_requires,
)