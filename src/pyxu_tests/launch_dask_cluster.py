#!/usr/bin/env python3

import argparse
import os
import pathlib as plib
import subprocess as subp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a Dask-distributed local cluster. "
        "Assumes the local Python environment has `dask.distributed` available."
    )
    parser.add_argument(
        "--host",
        help="URI, IP or hostname of this server. (default: %(default)s)",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--port",
        help="Serving port. (default: %(default)d)",
        type=int,
        default=8786,
    )
    parser.add_argument(
        "--dashboard-address",
        help="Address on which to listen for diagnostics dashboard. (default: %(default)d)",
        type=int,
        default=8787,
    )
    parser.add_argument(
        "--nworkers",
        help="Number of independant workers. (default: %(default)d)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--nthreads",
        help="Number of threads per worker. (default: %(default)d)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--log_dir",
        help="Directory where scheduler/worker logs are stored. (default: %(default)s)",
        type=str,
        default=os.getenv("XDG_CACHE_HOME", "~/.cache") + "/launch_dask_cluster",
    )

    args = parser.parse_args()

    # Max thread limits
    assert args.nthreads >= 1
    assert args.nworkers >= 1
    max_threads = os.cpu_count()
    requested_threads = args.nworkers * args.nthreads
    assert requested_threads <= max_threads, f"max_threads={max_threads}, but asked for {requested_threads}."

    # Create log directory
    args.log_dir = plib.Path(args.log_dir).expanduser().resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    return args


def launch_scheduler(args: argparse.Namespace) -> subp.Popen:
    log_file = args.log_dir / "scheduler.log"
    with open(log_file, mode="w") as f:
        p = subp.Popen(
            args=[
                "dask",
                "scheduler",
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--dashboard-address",
                str(args.dashboard_address),
                "--dashboard",
                "--no-show",
            ],
            stdout=f,
            stderr=subp.STDOUT,
        )
    print(f"Scheduler running at {args.host}:{args.port}")
    print(f"Dashboard running at {args.host}:{args.dashboard_address}")
    return p


def launch_workers(args: argparse.Namespace) -> subp.Popen:
    log_file = args.log_dir / "worker.log"
    with open(log_file, mode="w") as f:
        p = subp.Popen(
            args=[
                "dask",
                "worker",
                f"{args.host}:{args.port}",
                "--nworkers",
                str(args.nworkers),
                "--nthreads",
                str(args.nthreads),
            ],
            stdout=f,
            stderr=subp.STDOUT,
        )
    return p


if __name__ == "__main__":
    # Kill any old dask cluster running
    subp.run(args=["pkill", "-9", "dask"])

    args = parse_args()
    p_scheduler = launch_scheduler(args)
    p_worker = launch_workers(args)

    print("Cluster takedown: `pkill -9 dask`.")
