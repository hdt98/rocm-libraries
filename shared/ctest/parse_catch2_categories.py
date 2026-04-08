import yaml
import sys
import platform
import argparse
import contextlib


def load_yaml(yaml_file):
    """Load and parse a YAML file, exiting with a descriptive error on failure."""
    try:
        with open(yaml_file, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: YAML file not found: {yaml_file}", file=sys.stderr)
    except PermissionError:
        print(
            f"Error: Permission denied reading YAML file: {yaml_file}", file=sys.stderr
        )
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML syntax in {yaml_file}: {e}", file=sys.stderr)
    except Exception as e:
        print(
            f"Error: Unexpected failure loading {yaml_file}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
    sys.exit(1)


def build_catch2_tag_expression(test_tags, exclude_tags):
    """
    Build a Catch2 tag expression from include and exclude tag lists.

    Catch2 test spec grammar:
      - Comma ',' separates OR clauses
      - Space separates AND conditions within a clause
      - '~' negates the next condition

    Because comma binds looser than space, excludes must be duplicated
    across each include clause to get correct semantics:
      [a] ~[x],[b] ~[x]  means  ([a] AND NOT [x]) OR ([b] AND NOT [x])
    """
    include_tags = []
    if test_tags:
        for tag in test_tags:
            if tag != "[]":
                include_tags.append(tag)

    exclude_parts = []
    if exclude_tags:
        for tag in exclude_tags:
            exclude_parts.append(f"~{tag}")

    exclude_str = " ".join(exclude_parts) if exclude_parts else ""

    if not include_tags and not exclude_str:
        return ""

    if not include_tags:
        return exclude_str

    if not exclude_str:
        return ",".join(include_tags)

    # Duplicate excludes per include clause for correct precedence
    clauses = []
    for tag in include_tags:
        clauses.append(f"{tag} {exclude_str}")
    return ",".join(clauses)


def main():
    parser = argparse.ArgumentParser(
        description="Parse Catch2 test_categories YAML and generate CMake test definitions"
    )
    parser.add_argument("yaml_file", help="Path to the test_categories YAML file")
    parser.add_argument(
        "target_name", help="Name of the test target (e.g., rocroller-tests-catch)"
    )
    parser.add_argument("working_dir", help="Working directory for running tests")
    parser.add_argument(
        "install_test_file",
        nargs="?",
        default=None,
        help="Optional: Path to write install-time test definitions with relative paths",
    )

    args = parser.parse_args()

    yaml_file = args.yaml_file
    target_name = args.target_name
    working_dir = args.working_dir
    install_test_file = args.install_test_file

    config = load_yaml(yaml_file)

    try:
        install_cm = (
            open(install_test_file, "a", buffering=1)
            if install_test_file
            else contextlib.nullcontext()
        )
    except OSError as e:
        print(
            f"Warning: I/O error opening install test file {install_test_file}: {e}",
            file=sys.stderr,
        )
        install_cm = contextlib.nullcontext()
    except Exception as e:
        print(
            f"Warning: Unexpected error opening install test file {install_test_file}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        install_cm = contextlib.nullcontext()

    with install_cm as install_file_handle:
        categories = config.get("test_categories", {})
        execution_settings = config.get("execution_settings", {})
        timeouts = execution_settings.get("category_timeouts", {})
        timeout_multiplier = execution_settings.get("timeout_multiplier", 1)
        env_dict = execution_settings.get("environment", {}) or {}
        env_string = (
            ";".join(f"{k}={v}" for k, v in env_dict.items()) if env_dict else None
        )

        is_windows = platform.system() == "Windows"
        is_linux = platform.system() == "Linux"

        print("# Generated CMake code for Catch2 test categories (tag-based)")
        print(f"# Detected OS: {platform.system()}")
        print(f"# Timeout multiplier: {timeout_multiplier}")

        for category_name, category_info in categories.items():
            test_tags = category_info.get("test_tags", [])
            exclude_tags = category_info.get("exclude_tags", [])
            if exclude_tags is None:
                exclude_tags = []

            if is_windows:
                exclude_tags_win = category_info.get("exclude_tags_windows", [])
                if exclude_tags_win:
                    exclude_tags = exclude_tags + exclude_tags_win

            if is_linux:
                exclude_tags_linux = category_info.get("exclude_tags_linux", [])
                if exclude_tags_linux:
                    exclude_tags = exclude_tags + exclude_tags_linux

            labels = category_info.get("labels", [])
            base_timeout = timeouts.get(category_name, 300)
            timeout = int(base_timeout * timeout_multiplier)

            tag_expression = build_catch2_tag_expression(test_tags, exclude_tags)

            print(f"# Category: {category_name}")
            print(f'# Description: {category_info.get("description", "")}')
            print(
                f"# Tag expression: {tag_expression if tag_expression else '(all tests)'}"
            )

            if tag_expression:
                command_line = f'  COMMAND {target_name} "{tag_expression}"'
            else:
                command_line = f"  COMMAND {target_name}"

            label_string = '"' + ";".join(labels) + '"'

            print("add_test(")
            print(f"  NAME {target_name}_{category_name}_suite")
            print(command_line)
            print(f"  WORKING_DIRECTORY {working_dir}")
            print(")")

            print(
                f"set_tests_properties({target_name}_{category_name}_suite PROPERTIES"
            )
            print(f"  LABELS {label_string}")
            print(f"  TIMEOUT {timeout}")
            if env_string:
                print(f'  ENVIRONMENT "{env_string}"')
            print(")")
            print()

            if install_file_handle:
                try:
                    if tag_expression:
                        install_cmd = f'add_test({target_name}_{category_name}_suite "../{target_name}" "{tag_expression}")\n'
                    else:
                        install_cmd = f'add_test({target_name}_{category_name}_suite "../{target_name}")\n'
                    install_file_handle.write(install_cmd)
                    env_prop = f' ENVIRONMENT "{env_string}"' if env_string else ""
                    install_file_handle.write(
                        f"set_tests_properties({target_name}_{category_name}_suite PROPERTIES LABELS {label_string} TIMEOUT {timeout}{env_prop})\n\n"
                    )
                    install_file_handle.flush()
                except OSError as e:
                    print(
                        f"Warning: I/O error writing category {category_name} to install test file: {e}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"Warning: Unexpected error writing category {category_name} to install test file: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    main()
