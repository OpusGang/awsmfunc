from setuptools import setup
from distutils.command import build as build_module

import sys
import subprocess as sp
import shutil
from pathlib import Path

packages = {
    'awsmfunc': 'awsmfunc',
    'adjust': 'dependencies/adjust',
    'rekt': 'dependencies/rekt/rekt',
    'vsutil': 'dependencies/vsutil',
}

inits = {
    'adjust': 'dependencies/inits/adjust.py',
    'vsutil': 'dependencies/inits/vsutil.py',
}

created = []

def setup_deps():
    global created

    deps = Path("./dependencies")

    # Create a shitty __init__ file because vs people suck at python
    for d in deps.iterdir():
        if d.name == "inits":
            continue 

        final_path = Path.joinpath(d, "__init__.py")
        func_target_file = Path.joinpath(d, f"{d.name}.py")

        if func_target_file.is_file():
            created.append(final_path)
            init_path = Path(inits[d.name]).resolve()

            # Copy init file to dependency
            shutil.copyfile(init_path, final_path)

def cleanup_inits(created):
    if created:
        # Remove created inits
        for d in created:
            d.unlink()

class CustomBuild(build_module.build):
    def run(self):
        setup_deps()

setup(
    name='awsmfunc',
    version='0.2.0',
    url='https://git.concertos.live/AHD/awsmfunc',
    author='AHD',
    package_dir=packages,
    packages=[*packages],
    cmdclass={
        'build': CustomBuild,
    },
)

cleanup_inits(created)
