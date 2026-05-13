import sys
import warnings
from importlib import metadata
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
