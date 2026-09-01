from .data_manager import download_data_files
import argparse

#TODO - add subcommands for pathway running, segmentation and plotting
def main():
    """
    Run the ``methylseg`` command-line interface.

    Returns
    -------
    None
        Parses CLI arguments, executes the requested subcommand, and prints
        help text when no subcommand is provided.
    """
    parser = argparse.ArgumentParser(prog="methylseg")

    subparsers = parser.add_subparsers(dest="command")

    # subcommand: download_data_files
    dl_parser = subparsers.add_parser("download_data_files")
    dl_parser.add_argument(
        "--cleanup_existing",
        action="store_true",
        help="Delete existing files before downloading",
    )

    args = parser.parse_args()

    if args.command == "download_data_files":
        download_data_files(cleanup_existing=args.cleanup_existing)
    else:
        parser.print_help()
