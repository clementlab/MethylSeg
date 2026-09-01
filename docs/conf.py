import sys
import warnings
from shutil import copy2
from importlib import metadata
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = Path(__file__).resolve().parent
GENERATED_ROOT = DOCS_ROOT / "_generated"
GENERATED_TUTORIALS = DOCS_ROOT / "tutorials" / "generated"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

warnings.filterwarnings(
    "ignore",
    message=r"You are using an unsupported version of pandoc.*",
    category=RuntimeWarning,
)

project = "methylseg"
author = "Clement Lab"

try:
    release = metadata.version("methylseg")
except metadata.PackageNotFoundError:
    release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_generated", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
nbsphinx_execute = "never"

html_theme = "sphinx_rtd_theme"
html_static_path = []


def _write(path: Path, content: str) -> None:
    """Write a generated Sphinx source file only when its content changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def _manual_rst_pages() -> list[str]:
    """Return hand-authored pages that belong in the generated root toctree."""
    pages = []
    for path in sorted(DOCS_ROOT.rglob("*.rst")):
        relative = path.relative_to(DOCS_ROOT)
        if relative.parts[0] in {"_build", "_generated", "generated"}:
            continue
        if relative.parts[:2] == ("tutorials", "generated"):
            continue
        if relative == Path("index.rst"):
            continue
        pages.append(relative.with_suffix("").as_posix())
    return pages


def _generate_docs_sources(app) -> None:
    """Stage public notebooks and navigation before Sphinx reads source files."""
    examples_dir = PACKAGE_ROOT / "examples"
    GENERATED_TUTORIALS.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(examples_dir.glob("*.ipynb"))
    expected = {notebook.name for notebook in notebooks}
    for staged in GENERATED_TUTORIALS.glob("*.ipynb"):
        if staged.name not in expected:
            staged.unlink()
    for notebook in notebooks:
        copy2(notebook, GENERATED_TUTORIALS / notebook.name)

    tutorial_entries = [f"   {notebook.stem}" for notebook in notebooks]
    _write(
        GENERATED_TUTORIALS / "index.rst",
        "Example Notebooks\n=================\n\n"
        "These pages are generated from ``examples/*.ipynb``. Edit the "
        "notebooks in the package source, not the staged copies.\n\n"
        ".. toctree::\n   :maxdepth: 1\n\n"
        + "\n".join(tutorial_entries)
        + "\n",
    )

    entries = [f"   /{page}" for page in _manual_rst_pages()]
    _write(
        GENERATED_ROOT / "contents.rst",
        "Documentation\n=============\n\n"
        ".. toctree::\n   :maxdepth: 2\n   :caption: Contents\n\n"
        + "\n".join(entries)
        + "\n",
    )


def setup(app):
    app.connect("builder-inited", _generate_docs_sources)


# Sphinx discovers source files after loading this configuration module, so
# stage notebook pages now as well as before each builder initialization.
_generate_docs_sources(None)
