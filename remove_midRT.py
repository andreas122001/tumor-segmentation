import os
from shutil import rmtree

# Remove the midRT data
for top, dirs, files in os.walk("data/train"):
    for d in dirs:
        if d == "midRT":
            rmtree(os.path.join(top, d))
