import numpy as np
import pandas as pd
import sys
import os
import subprocess
import re
from typing import List, Dict, Any
from typing import List, Union, TextIO
from io import StringIO
import itertools
from tqdm import tqdm

DEFAULT_MIOPEN_DRIVER_ARGS = {
    "--in_w": 16,
    "--in_h": 16,
    "--in_d": 16,
    "--spatial_dim": 3,
    "--in_channels": 3,
    "--out_channels": 8,
    "--fil_d": 3,
    "--fil_h": 3,
    "--fil_w": 3,
    "--pad_h": 0,
    "--pad_w": 0,
    "--pad_d": 0,
    "--search": 1,
    "--time": 1,
    "--conv_stride_d": 1,
    "--conv_stride_h": 1,
    "--conv_stride_w": 1,
    "--dilation_d": 1,
    "--dilation_h": 1,
    "--dilation_w": 1,
}

# Regular expressions to match specific patterns in the log output

RUN_COUNTER_REGEX = re.compile(
    r"\(n_current, n_failed, n_runs_total\):\s+([0-9]+)/([0-9]+)/([0-9]+)"
)
CONFIG_REGEX = re.compile(
    r"config="
    r"(?P<config_name>[^<]+)"  # everything up to the '<'
    r"<(?P<param_list>[^>]+)>"  # everything between '<' and '>'
    r"(?:\+(?P<postfix_param>[0-9]+))?"  # optional +integer after '>'
)
AVG_TIME_REGEX = re.compile(r"avg_time=(?P<avg_time_ms>[0-9.]+) ms")
EVALUATE_INVOKER_REGEX = re.compile(
    r"Info \[EvaluateInvokers\] (?P<solver>[A-Za-z0-9_]+): (?P<config>(?:[A-Za-z\d_]+(?:/[A-Za-z\d_]+)*)?): (?P<time_ms>[0-9]*\.[0-9]+) [<>=+]"
)
CONTENT_INSERTED_REGEX = re.compile(
    r"Info2 \[SetValues\] (?P<fdb_key>[A-Za-z0-9-]+), content inserted: (?P<solver>[A-Za-z0-9_]+):(?P<kernel>[A-Za-z0-9_]+)<(?P<param_list>[^>]+)?>(?:\+(?P<postfix_param>[0-9]+))?"
)


def generate_parameter_combinations(**items):
    """Generate a dictionary for each combination of items in the input dictionary.
    Args:
        **items: A dictionary where keys are names and values are lists of items.
    Returns:
        An iterator of dictionaries, where each dictionary corresponds to a unique combination
        of items from the input lists.
    Example:
        >>> items = {'colors': ['red', 'green'], 'sizes': ['S', 'M']}
        >>> for product in generate_parameter_combinations(**items):
        ...     print(product)
        {'colors': 'red', 'sizes': 'S'}
        {'colors': 'red', 'sizes': 'M'}
        {'colors': 'green', 'sizes': 'S'}
        {'colors': 'green', 'sizes': 'M'}
    """
    keys = items.keys()
    for values in itertools.product(*items.values()):
        yield dict(zip(keys, values))


# set up debugging env vars for miopen
def set_miopen_env():
    """Set up MIOpen environment variables for debugging and configuration.
    This function creates a copy of the current environment and sets various MIOpen-specific
    environment variables for debugging and controlling convolution algorithms.
    Returns:
        dict: Modified environment dictionary with MIOpen-specific variables set as follows:
            - MIOPEN_LOG_LEVEL: Set to 6 for verbose logging
            - MIOPEN_FIND_MODE: Set to 1 to enable solution finding
            - MIOPEN_FIND_ENFORCE: Set to 4 to enforce specific solution search
            - MIOPEN_DEBUG_CONV_DIRECT: Disabled (0) - Direct convolution algorithm
            - MIOPEN_DEBUG_CONV_WINOGRAD: Disabled (0) - Winograd convolution algorithm
            - MIOPEN_DEBUG_CONV_GEMM: Enabled (1) - GEMM-based convolution algorithm
            - MIOPEN_DEBUG_CONV_IMPLICIT_GEMM: Enabled (1) - Implicit GEMM convolution
            - MIOPEN_DEBUG_GCN_ASM_KERNELS: Enabled (1) - GCN assembly kernels
            - MIOPEN_DEBUG_HIP_KERNELS: Enabled (1) - HIP kernel debugging
            - MIOPEN_DEBUG_OPENCL_CONVOLUTIONS: Enabled (1) - OpenCL convolution debugging
            - MIOPEN_USER_DB_path: Set to "./miopen_user_db" for solution database
    Note:
        The function also clears the contents of the "./miopen_user_db" directory if it exists,
        or creates it if it does not. This directory is used to store user-defined solutions
        for MIOpen convolution operations. This makes sure that the user_db directory is clean
        before running the benchmarks, allowing for fresh results without interference from previous runs.
    Raises:
        OSError: If there is an issue creating or clearing the user_db directory.
    """

    env = os.environ.copy()
    # set env variables for MIOpen debugging
    env["MIOPEN_LOG_LEVEL"] = "6"
    env["MIOPEN_FIND_MODE"] = "1"
    env["MIOPEN_FIND_ENFORCE"] = "4"
    env["MIOPEN_DEBUG_CONV_DIRECT"] = "0"
    env["MIOPEN_DEBUG_CONV_WINOGRAD"] = "0"
    env["MIOPEN_DEBUG_CONV_GEMM"] = "1"
    env["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"] = "1"
    env["MIOPEN_DEBUG_GCN_ASM_KERNELS"] = "1"
    env["MIOPEN_DEBUG_HIP_KERNELS"] = "1"
    env["MIOPEN_DEBUG_OPENCL_CONVOLUTIONS"] = "1"
    env["MIOPEN_USER_DB_path"] = "./miopen_user_db"

    # make sure the user_db directory exists
    if not os.path.exists("./miopen_user_db"):
        os.makedirs("./miopen_user_db")
    else:
        # clear the user_db directory by removing all files
        for f in os.listdir("./miopen_user_db"):
            file_path = os.path.join("./miopen_user_db", f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
    return env


def match_to_params(match: re.Match) -> List[Any]:
    """
    Convert a regex match object to a list of parameters.
    This is used to convert the parameters from the regex match to a list of integers or strings.
    """
    raw_params_str = match.group("param_list").strip()
    # convert the params to a list of integers or strings
    params = []
    for p in raw_params_str.split(","):
        p = p.strip()
        try:
            params.append(int(p))
        except ValueError:
            params.append(p)

    # if there was a "+N" after the '>', append that too
    postfix = match.group("postfix_param")
    if postfix is not None:
        params.append("+" + postfix)
    return params


def parse_benchmarks(source: Union[str, TextIO]) -> List[dict]:
    """
    Parse MIOpen benchmark output from a file or string and extract performance data.

    The function processes several types of log lines:
    1. Benchmark completion lines containing run counts and timing:
       "[GenericSearch] Finished benchmark (…), n/f/t: config=... avg_time=X ms"
    2. Kernel mapping information:
       "Info2 [SetValues] fdb_key, content inserted: solver:kernel<params>"
    3. Invoker evaluation lines:
       "Info [EvaluateInvokers] solver: config: time_ms"

    Args:
        source: File path as string or file-like object containing MIOpen log output

    Returns:
        pandas.DataFrame containing:
        - kernel_name: Name of the kernel/solver
        - kernel_params: Parameters for the kernel
        - avg_time_ms: Average execution time in milliseconds
        - Additional metadata like run counts where available

    Note:
        The function deduplicates kernel entries that appear both as solvers and
        kernels to avoid double-counting performance data.
    """
    results = []
    mapping_dicts = []
    close_when_done = False

    if isinstance(source, str):
        f = open(source, "r", encoding="utf-8")
        close_when_done = True
    else:
        f = source

    try:
        for line in f:
            if "miopendriver conv" in line.lower():
                # extract the command line arguments
                # for now we skip this, since we will catch the arguments outside this function
                continue
            if "Finished benchmark" in line:
                run_info = RUN_COUNTER_REGEX.search(line)

                if not run_info:
                    continue
                results_dict = {
                    "n_current": run_info[1],
                    "n_failed": run_info[2],
                    "n_total": run_info[3],
                }
                # extract the config and params
                config_match = CONFIG_REGEX.search(line)
                if not config_match:
                    continue
                results_dict["kernel_name"] = config_match.group("config_name").strip()
                results_dict["kernel_params"] = match_to_params(config_match)

                # extract the avg_time_ms
                avg_time_match = AVG_TIME_REGEX.search(line)
                if not avg_time_match:
                    continue
                results_dict["avg_time_ms"] = float(avg_time_match.group("avg_time_ms"))
                results.append(results_dict)
            if "Info2 [SetValues]" in line:
                # extract the fdb_key, solver, kernel and params
                # This will go into a mappin list to make sure we are not counting things double
                # Note this only matches lines that look like:
                # Info2 [SetValues] fdb_key, content inserted: solver_name:kernel_name<param1,param2,...>(+N)
                match = CONTENT_INSERTED_REGEX.search(line)
                if not match:
                    continue

                mapping_dict = {
                    "solver_name": match.group("solver"),
                    "kernel_name": match.group("kernel"),
                    "kernel_params": match_to_params(match),
                    "fdb_key": match.group("fdb_key"),
                }
                mapping_dicts.append(mapping_dict)
            if "Info [EvaluateInvokers]" in line:
                if "Selected" in line:
                    # skip lines that indicate a selected invoker
                    continue
                # extract the solver, config and time_ms
                match = EVALUATE_INVOKER_REGEX.search(line)
                if not match:
                    continue
                results_dict = {
                    "kernel_name": match.group("solver"),
                    "kernel_params": match.group("config"),
                    "avg_time_ms": float(match.group("time_ms")),
                    "n_current": 0,
                    "n_failed": 0,
                    "n_total": 0,
                }
                results.append(results_dict)
    finally:
        if close_when_done:
            f.close()

    # convert the results to a pandas DataFrame
    df = pd.DataFrame(results)
    df_map = pd.DataFrame(mapping_dicts)
    # Filter out duplicate kernel entries that appear both as solvers and kernels
    # 1. Find kernels that exist in both df and df_map (as kernel_name)
    # 2. Get the solver_names from df_map that correspond to these duplicate kernels
    # 3. Remove rows from df where kernel_name matches these solver_names
    # This prevents counting the same kernel twice under different names (i.e. as a solver and a kernel).
    duplicate_kernels = df_map[df_map["kernel_name"].isin(df["kernel_name"])][
        "solver_name"
    ]
    df = df[~df["kernel_name"].isin(duplicate_kernels)]
    return df


def extract_unique(run_data: pd.DataFrame, fname: str):
    """
    Extract unique kernel_name and kernel_params combinations from the run data
    and save them to a CSV file.
    Args:
        run_data (pd.DataFrame): DataFrame containing the run data with columns 'kernel_name' and 'kernel_params'.
        fname (str): The filename to save the unique combinations to.
    """
    unique_combinations = run_data[["kernel_name", "kernel_params"]]
    unique_combinations = unique_combinations.astype(str).drop_duplicates()
    # sort the unique combinations by kernel_name and kernel_params
    unique_combinations = unique_combinations.sort_values(
        by=["kernel_name", "kernel_params"]
    )
    unique_combinations.to_csv(fname, index=False)


def fill_miopendriver_args(
    args: Dict[str, Any], par_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Fill MIOpenDriver arguments with parameter values.

    Args:
        args: Base arguments dictionary with keys like '--param_h'
        par_dict: Parameter dictionary with keys like 'param_' or full parameter names like 'param_h'

    Returns:
        Dictionary with updated argument values
    """
    args_current = args.copy()

    for key in args_current:
        # Strip leading dashes from argument name
        param_name = key.lstrip("-")

        # Check if base name without h/w/d suffix exists in parameters
        base_name = param_name[:-1] if param_name[-1] in "hwd" else param_name

        # Update value if parameter exists
        if base_name in par_dict:
            args_current[key] = par_dict[base_name]
        if param_name in par_dict:
            args_current[key] = par_dict[param_name]

    return args_current


def __main__(
    miopendriver_path: str,
    grid_dict: Dict[str, List[Any]] = {},
    output_fname: str = "miopendriver_output",
):
    # set up debugging env vars for miopen
    env = set_miopen_env()

    # run /opt/rocm-6.3.2/bin/miopendriver
    if not os.path.exists(miopendriver_path):
        print(f"Error: {miopendriver_path} does not exist.")
        sys.exit(1)
    # check if miopendriver is executable
    if not os.access(miopendriver_path, os.X_OK):
        print(f"Error: {miopendriver_path} is not executable.")
        sys.exit(1)

    output_log = output_fname + ".log"
    # clear the output file if it exists
    if os.path.exists(output_log):
        os.remove(output_log)
    results_df = pd.DataFrame()

    iterator = generate_parameter_combinations(**grid_dict)
    # calculate the total number of runs for the progress bar
    total_runs = np.prod([len(g) for g in grid_dict.values()])
    for run_id, par_dict in enumerate(
        tqdm(
            iterator,
            total=total_runs,
            desc="Running MIOpenDriver benchmarks",
            unit="run",
        )
    ):
        args_current = fill_miopendriver_args(DEFAULT_MIOPEN_DRIVER_ARGS, par_dict)

        # construct the command line arguments
        cmd = f"{miopendriver_path} conv"
        for key, value in args_current.items():
            cmd += f" {key} {value}"

        try:
            result = subprocess.run(
                cmd, shell=True, check=True, capture_output=True, text=True, env=env
            )
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e.stderr}")
            sys.exit(1)

        # save full output including debug info to file
        with open(output_log, "a") as f:
            f.write(f"Command: {cmd}\n")
            f.write("Command output:\n")
            # write stdout to file
            f.write(result.stdout)
            # add stderr if available
            if result.stderr:
                f.write("\nError Output:\n")
                f.write(result.stderr)
            f.write("\nDebug Info:\n")
            f.write(e.stderr if "e" in locals() else "No debug info available.")

        # parse the output file for performance data
        results = parse_benchmarks(StringIO(result.stdout + result.stderr))
        # drop the n_current, n_failed, n_runs_total columns
        results = results.drop(
            columns=["n_current", "n_failed", "n_total"], errors="ignore"
        )
        # add the arguements to the results dataframe in their own columns
        for key, value in args_current.items():
            results[key] = value
        # add column for the run number/id
        results["run_id"] = run_id

        # sort columns of the results dataframe
        results = results[
            ["run_id", "kernel_name", "kernel_params", "avg_time_ms"]
            + list(args_current.keys())
        ]

        # add the results to the results dataframe
        results_df = pd.concat([results_df, results], ignore_index=True)

        # save the results to a csv file
        results_df.to_csv(output_fname + ".csv", index=False)

        extract_unique(results_df, output_fname + "_unique_configs.csv")


if __name__ == "__main__":

    miopendriver_path = "/home/bartgips/code/MIOpen/build/bin/MIOpenDriver"
    # miopendriver_path = '/opt/rocm-6.3.2/bin/MIOpenDriver'

    grid_dict = {
        "forw": [1, 2, 4],  # forward algorithms to test
        "fil_": [1, 3],  # [1, 3, 5],  # filter sizes to test
        "pad_": [0, 1],  # [0, 1, 2],  # padding sizes to test
        "conv_stride_": [1, 2],  # strides to test
        "in_": [8],  # [8, 16],  # input sizes to test
        "out_channels": [2],  # [2, 8],  # output channels to test
        "dilation_": [1, 2],  # dilation sizes to test
    }

    __main__(miopendriver_path, output_fname="miopendriver_output", grid_dict=grid_dict)
