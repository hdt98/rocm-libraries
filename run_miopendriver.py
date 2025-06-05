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


# set up debugging env vars for miopen
def set_miopen_env():
    # set MIOPEN_LOG_LEVEL to 6 for debug
    os.putenv("MIOPEN_LOG_LEVEL", "6")
    # set MIOPEN_FIND_MODE to 1 for debug
    os.putenv("MIOPEN_FIND_MODE", "1")
    # set MIOPEN_FIND_ENFORCE to 4 for debug
    os.putenv("MIOPEN_FIND_ENFORCE", "4")
    # disable naive kernels
    os.putenv("MIOPEN_DEBUG_CONV_DIRECT", "0")
    # disable winograd kernels
    os.putenv("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
    # enable GEMM and Implicit GEMM kernels
    os.putenv("MIOPEN_DEBUG_CONV_GEMM", "1")
    os.putenv("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM", "1")

    # make sure ASM, HIP, and OpenCL kernels are enabled
    os.putenv("MIOPEN_DEBUG_GCN_ASM_KERNELS", "1")
    os.putenv("MIOPEN_DEBUG_HIP_KERNELS", "1")
    os.putenv("MIOPEN_DEBUG_OPENCL_CONVOLUTIONS", "1")

    # set user_db path to a local directory
    os.putenv("MIOPEN_USER_DB_path", "./miopen_user_db")

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


# Precompile the regex for speed, capturing name and params separately
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


def parse_benchmarks(source: Union[str, TextIO]) -> List[dict]:
    """
    Parse a log file (or file-like object) and extract lines of the form:
      [GenericSearch] Finished benchmark (…),  1/0/3: config=… avg_time=0.123 ms
    Returns a list of dicts:
      [
        {
          'n_current':    1,
          'n_failed':     0,
          'n_runs_total': 3,
          'config':       'DeviceGroupedConvFwd…',
          'avg_time_ms':  0.124152
        },
        …
      ]
    """
    results = []
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
                # run_info = run_info.group(0)
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
                raw_params_str = config_match.group("param_list").strip()
                # convert the params to a list of integers or strings
                params = []
                for p in raw_params_str.split(","):
                    p = p.strip()
                    try:
                        params.append(int(p))
                    except ValueError:
                        params.append(p)

                # if there was a "+N" after the '>', append that too
                postfix = config_match.group("postfix_param")
                if postfix is not None:
                    params.append("+" + postfix)

                results_dict["kernel_params"] = params

                # extract the avg_time_ms
                avg_time_match = AVG_TIME_REGEX.search(line)
                if not avg_time_match:
                    continue
                results_dict["avg_time_ms"] = float(avg_time_match.group("avg_time_ms"))
                results.append(results_dict)
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

    return df


def extract_unique(run_data: pd.DataFrame, fname: str):
    unique_combinations = run_data[["kernel_name", "kernel_params"]]
    unique_combinations = unique_combinations.astype(str).drop_duplicates()
    # sort the unique combinations by kernel_name and kernel_params
    unique_combinations = unique_combinations.sort_values(
        by=["kernel_name", "kernel_params"]
    )
    unique_combinations.to_csv(fname, index=False)


def __main__(
    miopendirver_path: str,
    grid_dict: Dict[str, List[Any]] = None,
    output_fname: str = "miopendriver_output",
):
    # set up debugging env vars for miopen
    set_miopen_env()

    # run /opt/rocm-6.3.2/bin/miopendriver
    if not os.path.exists(miopendriver_path):
        print(f"Error: {miopendriver_path} does not exist.")
        sys.exit(1)
    # check if miopendriver is executable
    if not os.access(miopendriver_path, os.X_OK):
        print(f"Error: {miopendriver_path} is not executable.")
        sys.exit(1)

    args_base = {
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

    if grid_dict is None:
        grid_dict = dict(
            forwards=[1, 2, 4],  # forward algorithms to test
            filters=[1, 3, 5],  # filter sizes to test
            paddings=[0, 1, 2],  # padding sizes to test
            strides=[1, 2],  # strides to test
            inp_sizes=[8, 16],  # input sizes to test
            out_channels=[2, 8],  # output channels to test
            dilations=[1, 2],  # dilation sizes to test
        )

    output_log = output_fname + ".log"
    # clear the output file if it exists
    if os.path.exists(output_log):
        os.remove(output_log)
    results_df = pd.DataFrame()
    keyorder = ['filters', 'paddings', 'inp_sizes', 'out_channels', 'strides', 'dilations', 'forwards']
    grid = [grid_dict[key] for key in keyorder]
    iterator = itertools.product(*grid)
    # calculate the total number of runs for the progress bar
    total_runs = np.prod([len(g) for g in grid])
    for run_id, (
        filt_sz,
        pad_sz,
        inp_sz,
        out_chan,
        stride,
        dilation,
        forw,
    ) in enumerate(
        tqdm(
            iterator,
            total=total_runs,
            desc="Running MIOpenDriver benchmarks",
            unit="run",
        )
    ):
        args_current = args_base.copy()
        # set the iteration-specific arguments
        args_current["--fil_h"] = filt_sz
        args_current["--fil_w"] = filt_sz
        args_current["--fil_d"] = filt_sz
        args_current["--pad_h"] = pad_sz
        args_current["--pad_w"] = pad_sz
        args_current["--pad_d"] = pad_sz
        args_current["--forw"] = forw
        args_current["--in_w"] = inp_sz
        args_current["--in_h"] = inp_sz
        args_current["--in_d"] = inp_sz
        args_current["--out_channels"] = out_chan
        args_current["--conv_stride_h"] = stride
        args_current["--conv_stride_w"] = stride
        args_current["--conv_stride_d"] = stride
        args_current["--dilation_h"] = dilation
        args_current["--dilation_w"] = dilation
        args_current["--dilation_d"] = dilation

        # construct the command line arguments
        cmd = f"{miopendriver_path} conv"
        for key, value in args_current.items():
            cmd += f" {key} {value}"

        # print(f"Running command: {cmd}")
        # run command using subprocess

        try:
            result = subprocess.run(
                cmd, shell=True, check=True, capture_output=True, text=True
            )
            # print("Command output:")
            # print(result.stdout)
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
        # print the output file path
        # print(f"Output saved to {os.path.abspath( output_log)}")

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
        # print("Results saved to miopendriver_results.csv")

        extract_unique(results_df, output_fname + "_unique_configs.csv")


if __name__ == "__main__":

    miopendriver_path = "/home/bartgips/code/MIOpen/build/bin/MIOpenDriver"
    # miopendriver_path = '/opt/rocm-6.3.2/bin/MIOpenDriver'

    grid_dict = dict(
        forwards=[1, 2, 4],  # forward algorithms to test
        filters=[1, 3, 5],  # filter sizes to test
        paddings=[0, 1, 2],  # padding sizes to test
        strides=[1, 2],  # strides to test
        inp_sizes=[8, 16],  # input sizes to test
        out_channels=[2, 8],  # output channels to test
        dilations=[1, 2],  # dilation sizes to test
    )

    __main__(
        miopendriver_path, output_fname="miopendriver_output_v6", grid_dict=grid_dict
    )
