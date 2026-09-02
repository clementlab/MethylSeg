import sys
import warnings
from shutil import copy2
from importlib import metadata
from pathlib import Path
from re import MULTILINE, compile as re_compile

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = Path(__file__).resolve().parent
GENERATED_ROOT = DOCS_ROOT / "_generated"
GENERATED_TUTORIALS = DOCS_ROOT / "tutorials" / "generated"
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
myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"


def _write(path: Path, content: str) -> None:
    """Write a generated Sphinx source file only when its content changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


GITHUB_ALERT_PATTERN = re_compile(
    r"^> \[!(?P<kind>IMPORTANT|NOTE|WARNING|TIP|CAUTION)\]\n"
    r"(?P<body>(?:>.*(?:\n|$))*)",
    MULTILINE,
)


def _convert_github_alerts(content: str) -> str:
    """Convert GitHub alerts to MyST directives in staged documentation."""

    def replace_alert(match) -> str:
        body = "\n".join(
            line[2:] if line.startswith("> ") else line[1:]
            for line in match.group("body").splitlines()
        ).strip()
        return f":::{{{match.group('kind').lower()}}}\n{body}\n:::"

    return GITHUB_ALERT_PATTERN.sub(replace_alert, content)


def _manual_rst_pages() -> list[str]:
    return [
        "readme",
        "methylseg_methodology",
        "tutorials",
        "api",
        "troubleshooting",
    ]


def _stage_readme() -> None:
    """Copy the README while retargeting links that are relative to the repo."""
    readme = (PACKAGE_ROOT / "README.md").read_text()
    readme = _convert_github_alerts(readme)
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
    expected = {notebook.name for notebook in notebooks}
    for staged in GENERATED_TUTORIALS.glob("*.ipynb"):
        if staged.name not in expected:
            staged.unlink()
    for notebook in notebooks:
        copy2(notebook, GENERATED_TUTORIALS / notebook.name)

    _stage_readme()
    (GENERATED_TUTORIALS / "index.rst").unlink(missing_ok=True)
    tutorial_entries = [
        f"   /tutorials/generated/{notebook.stem}"
        for notebook in notebooks
    ]
    _write(
        GENERATED_TUTORIALS / "toctree.rst",
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
