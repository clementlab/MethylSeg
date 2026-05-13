#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python -m pip install -e ".[docs]"
rm -rf docs/_build
python -m sphinx -b html docs docs/_build/html

printf '\nBuilt docs: %s\n' "${SCRIPT_DIR}/docs/_build/html/index.html"
