import argparse

from .data_manager import download_data_files

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
    aggregate_parser = subparsers.add_parser("aggregate", help="Launch a multisample aggregation run.")
    aggregate_parser.add_argument("--config", required=True, help="Aggregation run YAML.")
    worker_parser = subparsers.add_parser("aggregate-worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--launch-spec", required=True)
    worker_group = worker_parser.add_mutually_exclusive_group(required=True)
    worker_group.add_argument("--task-index", type=int)
    worker_group.add_argument("--array-index", type=int)
    worker_parser.add_argument("--array-count", type=int)
    finalizer_parser = subparsers.add_parser("aggregate-finalize", help=argparse.SUPPRESS)
    finalizer_parser.add_argument("--launch-spec", required=True)

    args = parser.parse_args()

    if args.command == "download_data_files":
        download_data_files(cleanup_existing=args.cleanup_existing)
    elif args.command == "aggregate":
        from .aggregator import AggregationLauncher
        print(AggregationLauncher.from_yaml(args.config).launch())
    elif args.command == "aggregate-worker":
        from .aggregator import AggregationLauncher
        launcher = AggregationLauncher.from_launch_spec(args.launch_spec)
        if args.array_index is not None:
            if args.array_count is None:
                parser.error("--array-count is required with --array-index.")
            print(launcher.run_task_batch_from_spec(args.launch_spec, args.array_index, args.array_count))
        else:
            print(launcher.run_task_from_spec(args.launch_spec, args.task_index))
    elif args.command == "aggregate-finalize":
        from .aggregator import AggregationLauncher
        print(AggregationLauncher.from_launch_spec(args.launch_spec).finalize_from_spec(args.launch_spec))
    else:
        parser.print_help()
