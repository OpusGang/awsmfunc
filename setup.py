from setuptools import setup

with open('requirements.txt') as fh:
    install_requires = fh.read()

setup(
    name='awsmfunc',
    version='1.3.2',
    url='https://git.concertos.live/AHD/awsmfunc',
    author='AHD',
    packages=['awsmfunc'],
    package_data={
        'awsmfunc': ['py.typed'],
    },
    install_requires=install_requires,
    zip_safe=False,
)
