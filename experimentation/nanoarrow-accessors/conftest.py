import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.normpath(os.path.join(HERE, "..", "arrow-strings-in-tree"))):
    if p not in sys.path:
        sys.path.insert(0, p)
