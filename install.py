#!/usr/bin/env python3

import sys
import subprocess as sp
from pathlib import Path

deps = Path("./dependencies")

# Create a shitty __init__ file because vs people suck at python
for d in deps.iterdir():
    init = Path.joinpath(d, "__init__.py")
    with open(init, "w") as f:
        f.write(f"from .{d.name} import *")

cmd = [sys.executable, "-m", "pip", "install", "."]
opts = {"stdin": None,
        "stderr": sp.PIPE,
        "universal_newlines": True}

proc = sp.Popen(cmd, **opts)

out, err = proc.communicate()
if err:
    raise Exception(f"something is wrong with {cmd}, got: \"{err}\"")