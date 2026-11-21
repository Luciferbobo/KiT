"""Minimal Windows DDP launcher that avoids torchrun's libuv rendezvous."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = list(args.train_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    root = Path(__file__).resolve().parents[2]
    port = free_port()
    base_env = os.environ.copy()
    base_env.update(
        {
            "USE_LIBUV": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "WORLD_SIZE": str(args.nproc),
            "PYTHONPATH": str(root)
            + (
                os.pathsep + base_env["PYTHONPATH"]
                if base_env.get("PYTHONPATH")
                else ""
            ),
        }
    )
    processes = []
    for rank in range(args.nproc):
        env = base_env.copy()
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "kit_v1.train.cached_distill",
                    *forwarded,
                ],
                cwd=root,
                env=env,
            )
        )

    exit_code = 0
    try:
        for process in processes:
            code = process.wait()
            if code != 0 and exit_code == 0:
                exit_code = code
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        if exit_code != 0:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
