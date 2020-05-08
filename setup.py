# Initialize dependencies with `git submodule update --init --recursive`
# Run install.py to install

from setuptools import setup
from distutils.command import build as build_module

import argparse
import sys
import subprocess as sp
from pathlib import Path

created = []

def setup_deps():
    global created

    deps = Path("./dependencies")
    skipped = []

    # Create a shitty __init__ file because vs people suck at python
    for d in deps.iterdir():
        init = Path.joinpath(d, "__init__.py")
        func_module = Path.joinpath(d, d.name)
        func_target_file = Path.joinpath(d, f"{d.name}.py")

        if func_module.is_dir() or init.is_file():
            skipped.append(d.name)
        elif func_target_file.is_file():
            created.append(init)
            with open(init, "w") as f:
                f.write(f"from .{d.name} import *")

    print(f"Skipped custom packaging: {skipped}")

def cleanup_inits(created):
    if created:
        # Remove created inits
        for d in created:
            d.unlink()

packages = {
    'awsmfunc': 'awsmfunc',
    'adjust': 'dependencies/adjust',
    'fvsfunc': 'dependencies/fvsfunc',
    'havsfunc': 'dependencies/havsfunc',
    'muvsfunc': 'dependencies/muvsfunc',
    'mvsfunc': 'dependencies/mvsfunc',
    'nnedi3_resample': 'dependencies/nnedi3_resample',
    'nnedi3_rpow2': 'dependencies/nnedi3_rpow2',
    'rekt': 'dependencies/rekt/rekt',
    'vsutil': 'dependencies/vsutil',
}

optional = {
    'adptvgrnMod': 'dependencies/adptvgrnMod',
    'edi_rpow2': 'dependencies/edi_rpow2',
    'kagefunc': 'dependencies/kagefunc',
    'vsTAAmbk': 'dependencies/vsTAAmbk',
}

# If full is specified, add optional packages
if '--full' in sys.argv:
    packages.update(optional)
    sys.argv.remove('--full')

class CustomBuild(build_module.build):
    def run(self):
        setup_deps()

setup(
    name='awsmfunc',
    version='0.1.0',
    url='https://git.concertos.live/AHD/awsmfunc',
    author='AHD',
    package_dir=packages,
    packages=[*packages],
    cmdclass={
        'build': CustomBuild,
    },
)

cleanup_inits(created)
