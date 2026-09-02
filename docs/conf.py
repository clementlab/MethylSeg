import sys
import warnings
from shutil import copy2
from importlib import metadata
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = Path(__file__).resolve().parent
GENERATED_ROOT = DOCS_ROOT / "_generated"
GENERATED_TUTORIALS = DOCS_ROOT / "tutorials" / "generated"
QUICKSTART_NOTEBOOK = "01_quickstart.ipynb"
STAGED_QUICKSTART = DOCS_ROOT / "quickstart.ipynb"
STAGED_README = DOCS_ROOT / "readme.md"
STAGED_TROUBLESHOOTING = DOCS_ROOT / "troubleshooting.md"
README_IMAGE = "quickstart.png"
REPOSITORY_URL = "https://github.com/clementlab/MethylSeg"
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
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_generated",
    "tutorials/generated/toctree.rst",
    "Thumbs.db",
    ".DS_Store",
]

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
    # Quickstart is a staged notebook rather than a hand-authored RST page.
    pages = ["quickstart", "readme", "troubleshooting"]
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


def _stage_readme() -> None:
    """Copy the README while retargeting links that are relative to the repo."""
    readme = (PACKAGE_ROOT / "README.md").read_text()
    readme = readme.replace(
        "(TROUBLESHOOTING.md)", "(troubleshooting.md)"
    )
    readme = readme.replace(
        "(examples/run_full_pipeline.ipynb)",
        "(tutorials/generated/02_run_full_pipeline.html)",
    )
    readme = readme.replace("(LICENSE.md)", f"({REPOSITORY_URL}/blob/main/LICENSE)")
    _write(STAGED_README, readme)
    _write(STAGED_TROUBLESHOOTING, (PACKAGE_ROOT / "TROUBLESHOOTING.md").read_text())
    copy2(PACKAGE_ROOT / README_IMAGE, DOCS_ROOT / README_IMAGE)


def _generate_docs_sources(app) -> None:
    """Stage public notebooks and navigation before Sphinx reads source files."""
    examples_dir = PACKAGE_ROOT / "examples"
    GENERATED_TUTORIALS.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(examples_dir.glob("*.ipynb"))
    quickstart = next(
        (notebook for notebook in notebooks if notebook.name == QUICKSTART_NOTEBOOK),
        None,
    )
    if quickstart is None:
        raise RuntimeError(f"Missing canonical quickstart notebook: {QUICKSTART_NOTEBOOK}")

    tutorial_notebooks = [notebook for notebook in notebooks if notebook != quickstart]
    expected = {notebook.name for notebook in tutorial_notebooks}
    for staged in GENERATED_TUTORIALS.glob("*.ipynb"):
        if staged.name not in expected:
            staged.unlink()
    for notebook in tutorial_notebooks:
        copy2(notebook, GENERATED_TUTORIALS / notebook.name)

    # The notebook itself owns the Quickstart URL; the tutorial list links to it.
    copy2(quickstart, STAGED_QUICKSTART)
    _stage_readme()
    (GENERATED_TUTORIALS / "index.rst").unlink(missing_ok=True)
    tutorial_entries = [
        f"   /tutorials/generated/{notebook.stem}"
        for notebook in tutorial_notebooks
    ]
    _write(
        GENERATED_TUTORIALS / "toctree.rst",
        "* :doc:`Quickstart </quickstart>`\n\n"
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
