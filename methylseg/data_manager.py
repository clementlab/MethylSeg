import gdown
import shutil
import tarfile
from pathlib import Path

from .helper_classes import DATA_DIR

def is_lfs_pointer(path):
    """
    Check whether a file is still a Git LFS pointer stub.

    Parameters
    ----------
    path
        File to inspect.

    Returns
    -------
    bool
        ``True``` when the file begins with the standard Git LFS pointer
        header, otherwise ``False``.
    """
    try:
        with open(path, "r") as f:
            first_line = f.readline()
        return first_line.startswith("version https://git-lfs.github.com")
    except Exception:
        return False


def download_data_files(cleanup_existing=False):
    """
    Download and extract data files into DATA_DIR directory.

    Parameters
    ----------
    cleanup_existing
        If ``True``, remove existing extracted contents in ``DATA_DIR`` before
        downloading. If ``False``, skip the download when the directory is
        already populated.

    Returns
    -------
    None
        Downloads, extracts, and cleans up the packaged reference archive.
    """
    FILE_ID = "1pylU2nyidkmrhp8Gwgz5-VjjvgAMf4Gv"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / "data.tar.gz"

    # --- check existing ---
    if any(DATA_DIR.iterdir()) and not cleanup_existing:
        print(
            "Reference file directory is not empty. Use --cleanup_existing to force re-download."
        )
        return

    # --- Step 1: download ---
    print("Downloading data...")
    gdown.download(
        id=FILE_ID,
        output=str(archive_path),
        quiet=False,
    )

    # --- Step 2: clear existing contents ---
    for item in DATA_DIR.iterdir():
        if item == archive_path:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # --- Step 3: extract ---
    print("Extracting data...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)

    # --- Step 4: cleanup archive ---
    archive_path.unlink()

    print("Done.")
