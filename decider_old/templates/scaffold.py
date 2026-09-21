"""
Template renderer and project scaffolder.

All public functions take explicit paths and return Path objects of what was
written.  They are deliberately side-effect-free beyond filesystem writes so
they can be called from the magic, a CLI, or a test.
"""

import re
import textwrap
from pathlib import Path
from string import Template

_STATIC = Path(__file__).parent / "static"


# ── low-level helpers ─────────────────────────────────────────────────────────

def _render(template_path: Path, **kwargs) -> str:
    return Template(template_path.read_text()).substitute(**kwargs)


def to_snake(name: str) -> str:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


def extract_function_names(source: str) -> list[str]:
    """Top-level `def` names in order."""
    return re.findall(r"^def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", source, re.MULTILINE)


# ── inline extension (no package) ────────────────────────────────────────────

def write_inline_module(ext_dir: Path, class_name: str, user_code: str) -> Path:
    """
    Write decider_extensions/<snake_name>/__init__.py.
    Returns the path written.
    """
    snake = to_snake(class_name)
    fn_names = extract_function_names(user_code)
    if not fn_names:
        raise ValueError(
            f"No top-level functions found in cell body for {class_name!r}. "
            "Define at least one `def` returning pl.Expr."
        )

    pkg_dir = ext_dir / snake
    pkg_dir.mkdir(parents=True, exist_ok=True)
    out = pkg_dir / "__init__.py"
    out.write_text(
        _render(
            _STATIC / "extension_module.py",
            user_code=textwrap.dedent(user_code).strip(),
            class_name=class_name,
            type_id=snake,
            function_names=",\n    ".join(fn_names),
        )
    )
    return out


# ── package extension (uv src-layout) ────────────────────────────────────────

def _package_src_dir(ext_dir: Path, package_name: str) -> Path:
    return ext_dir / package_name / "src" / package_name


def _package_init_path(ext_dir: Path, package_name: str) -> Path:
    return _package_src_dir(ext_dir, package_name) / "__init__.py"


def _rebuild_package_init(src_dir: Path, package_name: str) -> None:
    """Re-generate __init__.py by importing every sibling .py that isn't __init__."""
    modules = sorted(
        p.stem for p in src_dir.glob("*.py") if p.stem != "__init__"
    )
    lines = [f"from .{m} import *" for m in modules]
    (src_dir / "__init__.py").write_text("\n".join(lines) + "\n" if lines else "")


def write_package_module(
    ext_dir: Path,
    class_name: str,
    package_name: str,
    user_code: str,
) -> tuple[Path, Path]:
    """
    Ensure ext_dir/<package_name>/ exists as a uv src-layout package, write
    (or overwrite) the module file for class_name, and regenerate __init__.py.

    Returns (module_file, init_file).
    """
    snake_class = to_snake(class_name)
    fn_names = extract_function_names(user_code)
    if not fn_names:
        raise ValueError(
            f"No top-level functions found in cell body for {class_name!r}."
        )

    src_dir = _package_src_dir(ext_dir, package_name)
    src_dir.mkdir(parents=True, exist_ok=True)

    # pyproject.toml — only create if missing
    pyproject = ext_dir / package_name / "pyproject.toml"
    if not pyproject.exists():
        pyproject.write_text(
            _render(_STATIC / "extension_package" / "pyproject.toml", package_name=package_name)
        )

    module_file = src_dir / f"{snake_class}.py"
    module_file.write_text(
        _render(
            _STATIC / "extension_package" / "module.py",
            user_code=textwrap.dedent(user_code).strip(),
            class_name=class_name,
            type_id=snake_class,
            function_names=",\n    ".join(fn_names),
        )
    )

    _rebuild_package_init(src_dir, package_name)
    return module_file, src_dir / "__init__.py"


# ── new project scaffold ──────────────────────────────────────────────────────

def write_project(projects_dir: Path, project_name: str) -> Path:
    """
    Scaffold a new project directory from the project template.
    Returns the created project directory.
    """
    snake = to_snake(project_name)
    project_dir = projects_dir / snake
    if project_dir.exists():
        raise FileExistsError(f"Project directory already exists: {project_dir}")

    project_dir.mkdir(parents=True)
    (project_dir / "decider_extensions").mkdir()
    (project_dir / "configs").mkdir()

    vars_ = dict(
        project_title=project_name.replace("_", " ").title(),
        project_dir=f"projects/{snake}",
    )

    for tmpl in (_STATIC / "project").iterdir():
        if tmpl.is_file():
            (project_dir / tmpl.name).write_text(
                _render(tmpl, **vars_)
            )

    return project_dir
