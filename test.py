#!/usr/bin/env python3

# #############################################################################
# test.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

"""
Run tests.
"""

import argparse
import pathlib
import subprocess
import sys

project_root_dir = pathlib.Path(__file__).parent.absolute()
cmds = dict(
    doctest=[f'sphinx-build -b doctest "{project_root_dir}/doc" "{project_root_dir}/build/doctest"']
)

parser = argparse.ArgumentParser(
    description="pycsou test runner.",
    epilog="When run with no arguments, all tests are executed.",
)
parser.add_argument("-e", help="Name of test to run.", type=str, choices=cmds.keys())
args = parser.parse_args()


def run_test(test_name):
    """
    Execute a test on the command line.

    Parameters
    ----------
    test_name : str
        Name of a key in `cmds`.
    """
    if not isinstance(test_name, str):
        raise ValueError("Parameter[test_name] must be a str.")
    if test_name not in cmds:
        raise ValueError("Parameter[test_name] is not a valid test.")

    status = subprocess.run(
        "; ".join(cmds[test_name]),
        stdin=None,
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=True,
        cwd=project_root_dir,
    )
    return status


if __name__ == "__main__":
    status = []

    if args.e is None:
        for test in cmds:
            s = run_test(test)
            status.append(s)
    else:
        s = run_test(args.e)
        status.append(s)

    print("\nSummary\n=======")
    for s in status:
        print("Success" if (s.returncode == 0) else "Failure")
        cmd_list = s.args.split("; ")
        for c in cmd_list:
            print(f"   {c}")
