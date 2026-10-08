import sys
from pathlib import Path

# Examples import each other by package name (`credit_core`, ...), so every project directory is a path root.
for project in sorted(Path(__file__).parent.iterdir()):
    if project.is_dir() and str(project) not in sys.path:
        sys.path.append(str(project))
