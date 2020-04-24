#!/usr/bin/env python3

import sys
import subprocess as sp
from pathlib import Path

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
        with open(init, "w") as f:
            f.write(f"from .{d.name} import *")

print(f"Skipped custom packaging: {skipped}")

cmd = [sys.executable, "-m", "pip", "install", "."]
opts = {"stdin": None,
        "stderr": sp.PIPE,
        "universal_newlines": True}

proc = sp.Popen(cmd, **opts)

out, err = proc.communicate()
if err:
    raise Exception(f"something is wrong with {cmd}, got: \"{err}\"")