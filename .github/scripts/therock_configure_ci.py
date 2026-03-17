"""
This script determines which build flag and tests to run based on SUBTREES

Required environment variables:
  - SUBTREES

Optional environment variables:
  - BUILD_VARIANT: Build variant to use (e.g., "release", "asan", "tsan"). Defaults to "release".
"""

import fnmatch
import json
import logging
import subprocess
from pathlib import Path
import sys
from therock_matrix import subtree_to_project_map, collect_projects_to_run
import time
from typing import Mapping, Optional, Iterable, List, Tuple
import os
from pr_detect_changed_subtrees import get_valid_prefixes, find_matched_subtrees
from config_loader import load_repo_config


logging.basicConfig(level=logging.INFO)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # .github/scripts -> .github -> repo root

# Importing TheRock amdgpu_family_matrix.py
sys.path.append(str(REPO_ROOT / "TheRock" / "build_tools" / "github_actions"))
from amdgpu_family_matrix import all_build_variants


def retrieve_build_variant(args) -> Tuple[str, str, bool]:
    """
    Retrieves build variant configuration from TheRock's amdgpu_family_matrix.

    Args:
        args: Dictionary containing 'platform' and 'build_variant' keys

    Returns:
        Tuple of (cmake_preset, label, expect_failure)
    """
    platform = args.get("platform")
    build_variant = args.get("build_variant", "release")

    if platform in all_build_variants and build_variant in all_build_variants[platform]:
        build_variant_cmake_preset = all_build_variants[platform][build_variant][
            "build_variant_cmake_preset"
        ]
        build_variant_label = all_build_variants[platform][build_variant][
            "build_variant_label"
        ]
        expect_failure = all_build_variants[platform][build_variant].get(
            "expect_failure", False
        )
    else:
        logging.warning(
            f"Unknown build variant '{build_variant}' for platform '{platform}', using defaults"
        )
        build_variant_cmake_preset = ""
        build_variant_label = ""
        expect_failure = False

    return build_variant_cmake_preset, build_variant_label, expect_failure

# Paths matching any of these patterns are considered to have no influence over
# build or test workflows so any related jobs can be skipped if all paths
# modified by a commit/PR match a pattern in this list.
SKIPPABLE_PATH_PATTERNS = [
    "docs/*",
    ".gitignore",
    "*.md",
    "*.rtf",
    "*.rst",
    "*/.markdownlint-ci2.yaml",
    "*/.readthedocs.yaml",
    "*/.spellcheck.local.yaml",
    "*/.wordlist.txt",
    "projects/*/docs/*",
    "projects/*/.gitignore",
    "shared/*/docs/*",
    "shared/*/.gitignore",
    "dnn-providers/*/docs/*",
    "dnn-providers/*/.gitignore",
    "*.clinerules",
    "*.cursorrules",
    "*.mdc",
]


def is_path_skippable(path: str) -> bool:
    """Determines if a given relative path to a file matches any skippable patterns."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in SKIPPABLE_PATH_PATTERNS)


def get_pr_labels(args) -> List[str]:
    """Gets a list of labels applied to a pull request."""
    data = json.loads(args.get("pr_labels", "{}"))
    labels = []
    for label in data.get("labels", []):
        labels.append(label["name"])
    return labels


def check_for_non_skippable_path(paths: Optional[Iterable[str]]) -> bool:
    """Returns true if at least one path is not in the skippable set."""
    if paths is None:
        return False
    return any(not is_path_skippable(p) for p in paths)


def set_github_output(d: Mapping[str, str]):
    """Sets GITHUB_OUTPUT values.
    See https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/passing-information-between-jobs
    """
    logging.info(f"Setting github output:\n{d}")
    step_output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not step_output_file:
        logging.warning(
            "Warning: GITHUB_OUTPUT env var not set, can't set github outputs"
        )
        return
    with open(step_output_file, "a") as f:
        f.writelines(f"{k}={v}" + "\n" for k, v in d.items())


def retry(max_attempts, delay_seconds, exceptions):
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(
                        f"Exception {str(e)} thrown when attempting to run , attempt {attempt} of {max_attempts}"
                    )
                    attempt += 1
                    if attempt < max_attempts:
                        backoff = delay_seconds * (2 ** (attempt - 1))
                        time.sleep(backoff)
            return func(*args, **kwargs)

        return newfn

    return decorator


@retry(max_attempts=3, delay_seconds=2, exceptions=(TimeoutError))
def get_modified_paths(base_ref: str) -> Optional[Iterable[str]]:
    """Returns the paths of modified files relative to the base reference."""
    return subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
        timeout=60,
    ).stdout.splitlines()


GITHUB_WORKFLOWS_CI_PATTERNS = [
    "therock*",
]


def is_path_workflow_file_related_to_ci(path: str) -> bool:
    return any(
        fnmatch.fnmatch(path, ".github/workflows/" + pattern)
        for pattern in GITHUB_WORKFLOWS_CI_PATTERNS
    ) or any(
        fnmatch.fnmatch(path, ".github/scripts/" + pattern)
        for pattern in GITHUB_WORKFLOWS_CI_PATTERNS
    )


def check_for_workflow_file_related_to_ci(paths: Optional[Iterable[str]]) -> bool:
    if paths is None:
        return False
    return any(is_path_workflow_file_related_to_ci(p) for p in paths)


def get_changed_path_projects(paths: Optional[Iterable[str]]) -> Iterable[str]:
    repo_config_path = Path(SCRIPT_DIR / ".." / "repos-config.json")
    config = load_repo_config(str(repo_config_path))
    valid_prefixes = get_valid_prefixes(config)
    matched_subtrees = find_matched_subtrees(paths, valid_prefixes)
    return matched_subtrees


def retrieve_projects(args):
    # For pushes and pull_requests, we only want to test changed projects
    base_ref = args.get("base_ref")
    modified_paths = get_modified_paths(base_ref)

    # by default, we select full tests
    test_type = "full"

    # Check if CI should be skipped based on modified paths
    # (only for push and pull_request events, not workflow_dispatch or nightly)
    if args.get("is_push") or args.get("is_pull_request"):
        paths_set = set(modified_paths)
        contains_non_skippable_files = check_for_non_skippable_path(paths_set)
        pr_labels = get_pr_labels(args)

        # If only skippable paths were modified, skip CI
        if not contains_non_skippable_files:
            logging.info("Only skippable paths were modified, skipping CI")
            return [], test_type

        if "skip-therockci" in pr_labels:
            logging.info("`skip-therockci` label was added, skipping CI")
            return [], test_type

    subtrees = get_changed_path_projects(modified_paths)

    if args.get("is_workflow_dispatch"):
        if args.get("input_projects") == "all":
            subtrees = list(subtree_to_project_map.keys())
        else:
            subtrees = args.get("input_projects").split()

    # If .github/*/therock* were changed for a push or pull request, run all subtrees
    if args.get("is_push") or args.get("is_pull_request"):
        related_to_therock_ci = check_for_workflow_file_related_to_ci(modified_paths)
        if related_to_therock_ci:
            logging.info(
                "Enabling all projects since a related workflow file was modified"
            )
            subtrees = list(subtree_to_project_map.keys())
            test_type = "smoke"

    # for nightly runs, run everything with full tests
    if args.get("is_nightly"):
        subtrees = list(subtree_to_project_map.keys())

    project_to_run = collect_projects_to_run(subtrees)

    return project_to_run, test_type


def run(args):
    platform = args.get("platform")
    project_to_run, test_type = retrieve_projects(args)

    # Get build variant configuration from TheRock
    build_variant_cmake_preset, build_variant_label, expect_failure = (
        retrieve_build_variant(args)
    )

    outputs = {
        f"{platform}_projects": json.dumps(project_to_run),
        "test_type": test_type,
        "build_variant_cmake_preset": build_variant_cmake_preset,
        "build_variant_label": build_variant_label,
        "expect_failure": json.dumps(expect_failure),
    }
    set_github_output(outputs)


if __name__ == "__main__":
    args = {}
    github_event_name = os.getenv("GITHUB_EVENT_NAME")
    platform = os.getenv("PLATFORM")
    args["platform"] = platform
    args["is_pull_request"] = github_event_name == "pull_request"
    args["is_push"] = github_event_name == "push"
    args["is_workflow_dispatch"] = github_event_name == "workflow_dispatch"
    args["is_nightly"] = github_event_name == "schedule"

    args["pr_labels"] = os.environ.get("PR_LABELS", '{"labels": []}')

    input_projects = os.getenv("PROJECTS", "")
    args["input_projects"] = input_projects

    args["base_ref"] = os.environ.get("BASE_REF", "HEAD^")

    # Build variant (e.g., "release", "asan", "tsan")
    args["build_variant"] = os.environ.get("BUILD_VARIANT", "release")

    logging.info(f"Retrieved arguments {args}")

    run(args)
