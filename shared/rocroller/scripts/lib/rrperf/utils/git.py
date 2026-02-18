# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Git utilities."""

import subprocess
from pathlib import Path


def top(loc: str | None = None) -> Path:
    path_arg = ["-C", loc] if loc is not None else []
    command = ["git"] + path_arg + ["rev-parse", "--show-toplevel"]
    p = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(p.stdout.strip()).resolve()


def clone(remote: str | Path, repo: Path) -> None:
    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            str(remote),
            str(repo),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "init", "--cone"], cwd=str(repo), check=True
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", "shared/rocroller", "shared/mxdatagenerator"],
        cwd=str(repo),
        check=True,
    )


def checkout(repo: Path, commit: str) -> None:
    subprocess.run(["git", "checkout", commit], cwd=str(repo), check=True)


def is_dirty(repo: Path) -> bool:
    p = subprocess.run(
        ["git", "diff-index", "--quiet", "HEAD"], check=False, cwd=str(repo)
    )
    return p.returncode != 0


def branch(repo: Path) -> str:
    p = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        encoding="ascii",
        check=True,
    )
    return p.stdout.strip()


def short_hash(repo: Path, commit: str = "HEAD") -> str:
    p = subprocess.run(
        ["git", "rev-parse", "--short", commit],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        encoding="ascii",
        check=True,
    )
    return p.stdout.strip()


def full_hash(repo: Path) -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        encoding="ascii",
        check=True,
    )
    return p.stdout.strip()


def rev_list(repo: Path, old_commit: str, new_commit: str) -> list[str]:
    """
    Gets a list of commits, starting with the newest commit and ending
    with the oldest commit, with every commit in between.
    Returns an empty list if there is no path.
    """
    p = subprocess.run(
        [
            "git",
            "rev-list",
            "--ancestry-path",
            f"{old_commit}~..{new_commit}",
        ],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        encoding="ascii",
        check=True,
    )
    return p.stdout.strip().split("\n")


def ls_tree(repo: Path | None = None, consider_staged: bool = False) -> list[Path]:
    """
    Returns a list of all files committed into Git in this repo.
    If `consider_staged` is True, then staged and unmodified files are listed instead.
    """

    if repo is None:
        repo = Path.cwd()

    cmd = [
        "git",
        "ls-tree",
        "--full-tree",
        "--full-name",
        "-r",
        "--name-only",
        "HEAD:shared/rocroller",
    ]

    p = subprocess.run(
        cmd,
        cwd=str(repo),
        stdout=subprocess.PIPE,
        check=True,
    )

    files = [repo / path for path in p.stdout.decode().strip().split("\n")]

    if consider_staged:
        p = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v2",
            ],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            check=True,
        )

        for entry in p.stdout.decode().splitlines():
            fields = entry.split(" ")
            if fields[0] == "1":
                if fields[1] == "A.":
                    files.append(repo / fields[-1])
                elif fields[1] == "D.":
                    files.remove(repo / fields[-1])
            elif fields[0] == "2" and fields[1][0] == "R":
                added, removed = fields[-1].split("\t")
                files.append(repo / added)
                files.remove(repo / removed)

    return files


def try_getting_commit(repo: Path | None) -> str | None:
    if repo is not None:
        try:
            return short_hash(repo)
        except Exception:
            pass
    return None


def get_commit(
    rundir: Path | None = None,
    build_dir: Path | None = None,
) -> str:
    commit = try_getting_commit(build_dir)
    if commit is None:
        commit = try_getting_commit(rundir)
    if commit is None:
        commit = try_getting_commit(Path("."))
    if commit is None:
        commit = try_getting_commit(Path(__file__).resolve().parent)
    if commit is None:
        commit = "NO_COMMIT"
    return commit
