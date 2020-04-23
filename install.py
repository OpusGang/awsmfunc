import sys
import subprocess
from pathlib import Path

deps = Path("./dependencies")

# Create shitty __init__ files into each dep
for d in deps.iterdir():
    name = d.name
    init = Path.joinpath(d, "__init__.py")
    with open(init, "w") as f:
        f.write("from .{} import *".format(name))

subprocess.check_call([sys.executable, "-m", "pip", "install", "."])