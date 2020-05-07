#!/usr/bin/env python3

# Initialize dependencies with `git submodule update --init --recursive`
# Run this

import sys
import subprocess as sp
from pathlib import Path

deps = Path("./dependencies")

created = []
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

cmd = [sys.executable, "-m", "pip", "install", "."]
opts = {"stdin": None,
        "stderr": sp.PIPE,
        "universal_newlines": True}

proc = sp.Popen(cmd, **opts)

out, err = proc.communicate()
if proc.returncode and err:
    raise Exception(f"something is wrong with {cmd}, got: \"{err}\"")
elif created:
    # Remove created inits
    for d in created:
        d.unlink()
