# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Convert miopen driver command to ck Profiler
# Example (single command mode):
#   python3 ../script/convert_miopen_driver_to_profiler.py
#   /opt/rocm/bin/MIOpenDriver conv -n 32 -c 64 -H 28 -W 28 -k 64 -y 3 -x 3
#   -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 32 -F 1 -t 1
#
# Example (batch mode):
#   python3 ../script/convert_miopen_driver_to_profiler.py
#   --input-file commands.txt --output-file results.txt

import argparse
import subprocess
import shlex
import sys
import re
from io import StringIO


def filter_to_best_config(output):
    """Filter output to only show the best configuration section.
    
    Args:
        output: Full profiler output string
        
    Returns:
        Filtered output containing only the best configuration section.
    """
    if not output:
        return output
    
    lines = output.split('\n')
    result_lines = []
    in_best_config = False
    
    # Valid prefixes for best config section
    valid_prefixes = ('Best configuration parameters:', 'name:', 'avg_time:', 'tflops:', 'GB/s:')
    
    for line in lines:
        if 'Best configuration parameters:' in line:
            in_best_config = True
        
        if in_best_config:
            stripped = line.strip()
            # Stop if we hit an error line, empty line after content, or other unexpected content
            if stripped.startswith('Error:') or stripped.startswith('max err:'):
                continue
            # Only include lines that are part of the best config or empty
            if stripped == '' or any(stripped.startswith(p) for p in valid_prefixes):
                result_lines.append(line)
            elif stripped and not any(stripped.startswith(p) for p in valid_prefixes):
                # Unknown line after best config section - stop
                continue
    
    # Remove trailing empty lines
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()
    
    return '\n'.join(result_lines) if result_lines else output


def init_const_args(args, profiler_path=None):
    args.ck_profiler_cmd = profiler_path if profiler_path else "../build/bin/ckProfiler"
    # use decimal values
    args.init_method = 2
    # don't print tensor values
    args.log_value = 0


def run_ck_profiler_cmd(cmd, capture_output=False):
    """Run the ckProfiler command.
    
    Args:
        cmd: List of command arguments
        capture_output: If True, capture and return stdout/stderr instead of printing
        
    Returns:
        If capture_output is True, returns a tuple (output_string, has_errors).
        Otherwise returns None.
    """
    cmd_concatenated_str = ""
    for arg in cmd:
        cmd_concatenated_str += arg + " "
    
    if capture_output:
        output = StringIO()
        output.write("ckProfiler command:\n")
        output.write(cmd_concatenated_str + "\n")
        has_errors = False
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output.write(result.stdout)
            if result.stderr:
                has_errors = True
                output.write(result.stderr)
        except Exception as e:
            has_errors = True
            output.write(f"Error running command: {e}\n")
        return (output.getvalue(), has_errors)
    else:
        print("ckProfiler command:")
        print(cmd_concatenated_str)
        subprocess.run(cmd)
        return None


def parse_layouts(args):
    if args.in_layout == "NCW" or args.in_layout == "NCHW" or args.in_layout == "NCDHW":
        if args.ck_profier_op == "grouped_conv_bwd_weight":
            args.layout = 4
        elif (
            args.ck_profier_op == "grouped_conv_fwd"
            or args.ck_profier_op == "grouped_conv_bwd_data"
        ):
            args.layout = 3
        else:
            print("Not supported layout for this op")
            exit(1)
    elif (
        args.in_layout == "NWC" or args.in_layout == "NHWC" or args.in_layout == "NDHWC"
    ):
        if args.ck_profier_op == "grouped_conv_bwd_weight":
            args.layout = 2
        elif (
            args.ck_profier_op == "grouped_conv_bwd_data"
            or args.ck_profier_op == "grouped_conv_fwd"
        ):
            args.layout = 1
    else:
        print("Not supported layout for this op")
        exit(1)


def parse_data_type(args):
    if args.data_type == "fp32":
        if (
            args.ck_profier_op == "grouped_conv_bwd_weight"
            or args.ck_profier_op == "grouped_conv_bwd_data"
            or args.ck_profier_op == "grouped_conv_fwd"
        ):
            args.data_type = 0
    if args.data_type == "fp16":
        if (
            args.ck_profier_op == "grouped_conv_bwd_weight"
            or args.ck_profier_op == "grouped_conv_bwd_data"
            or args.ck_profier_op == "grouped_conv_fwd"
        ):
            args.data_type = 1
    if args.data_type == "int8":
        if args.ck_profier_op == "grouped_conv_bwd_weight":
            args.data_type = 4
        if args.ck_profier_op == "grouped_conv_bwd_data":
            print("Not supported data type for grouped_conv_bwd_data")
            exit(1)
        if args.ck_profier_op == "grouped_conv_fwd":
            args.data_type = 3
    if args.data_type == "bfp16":
        if args.ck_profier_op == "grouped_conv_bwd_weight":
            args.data_type = 5
        if (
            args.ck_profier_op == "grouped_conv_bwd_data"
            or args.ck_profier_op == "grouped_conv_fwd"
        ):
            args.data_type = 2


def add_conv_params_to_cmd(args, cmd):
    if args.spatial_dim == 1:
        cmd += [str(args.fil_w), str(args.in_w)]
        cmd += [str(args.conv_stride_w), str(args.dilation_w)]
        cmd += [str(args.pad_w), str(args.pad_w)]
    elif args.spatial_dim == 2:
        cmd += [str(args.fil_h), str(args.fil_w)]
        cmd += [str(args.in_h), str(args.in_w)]
        cmd += [str(args.conv_stride_h), str(args.conv_stride_w)]
        cmd += [str(args.dilation_h), str(args.dilation_w)]
        cmd += [str(args.pad_h), str(args.pad_w)]
        cmd += [str(args.pad_h), str(args.pad_w)]
    elif args.spatial_dim == 3:
        cmd += [str(args.fil_d), str(args.fil_h), str(args.fil_w)]
        cmd += [str(args.in_d), str(args.in_h), str(args.in_w)]
        cmd += [str(args.conv_stride_d), str(args.conv_stride_h)]
        cmd += [str(args.conv_stride_w)]
        cmd += [str(args.dilation_d), str(args.dilation_h), str(args.dilation_w)]
        cmd += [str(args.pad_d), str(args.pad_h), str(args.pad_w)]
        cmd += [str(args.pad_d), str(args.pad_h), str(args.pad_w)]
    else:
        print("Not supported spatial dim (supported: 1, 2, 3)")
        exit(1)


def run_ck_grouped_conv_fwd(args, capture_output=False):
    args.ck_profier_op = "grouped_conv_fwd"
    parse_data_type(args)
    parse_layouts(args)
    # use int32 by default
    args.index_type = 0

    cmd = [str(args.ck_profiler_cmd), str(args.ck_profier_op)]
    cmd += [str(args.data_type), str(args.layout), str(args.index_type)]
    cmd += [str(args.verify), str(args.init_method)]
    cmd += [str(args.log_value), str(args.time)]
    cmd += [str(args.spatial_dim), str(args.group_count)]
    cmd += [str(args.batchsize), str(args.out_channels)]
    cmd += [str(args.in_channels)]
    add_conv_params_to_cmd(args, cmd)

    # Add optional named arguments
    if args.instance != -1:
        cmd += ["--instance", str(args.instance)]
    if args.list_instances:
        cmd += ["--list-instances"]

    result = run_ck_profiler_cmd(cmd, capture_output)
    return result


def run_ck_grouped_conv_bwd_data(args, capture_output=False):
    args.ck_profier_op = "grouped_conv_bwd_data"
    parse_data_type(args)
    parse_layouts(args)
    # Test all split K value from the list {1, 2, 4, 8, 32, 64, 128}
    args.split_k_value = -1

    cmd = [str(args.ck_profiler_cmd), str(args.ck_profier_op)]
    cmd += [str(args.data_type), str(args.layout)]
    cmd += [str(args.verify), str(args.init_method)]
    cmd += [str(args.log_value), str(args.time)]
    cmd += [str(args.spatial_dim), str(args.group_count)]
    cmd += [str(args.batchsize), str(args.out_channels)]
    cmd += [str(args.in_channels)]
    add_conv_params_to_cmd(args, cmd)

    cmd += [str(args.split_k_value)]
    
    # Add optional named arguments
    if args.instance != -1:
        cmd += ["--instance", str(args.instance)]
    if args.list_instances:
        cmd += ["--list-instances"]
    
    return run_ck_profiler_cmd(cmd, capture_output)


def run_ck_grouped_conv_bwd_weight(args, capture_output=False):
    args.ck_profier_op = "grouped_conv_bwd_weight"
    parse_data_type(args)
    parse_layouts(args)
    # Test all split K value from the list {1, 2, 4, 8, 32, 64, 128}
    args.split_k_value = "all"

    cmd = [str(args.ck_profiler_cmd), str(args.ck_profier_op)]
    cmd += [str(args.data_type), str(args.layout)]
    cmd += [str(args.verify), str(args.init_method)]
    cmd += [str(args.log_value), str(args.time)]
    cmd += [str(args.spatial_dim), str(args.group_count)]
    cmd += [str(args.batchsize), str(args.out_channels)]
    cmd += [str(args.in_channels)]
    add_conv_params_to_cmd(args, cmd)

    cmd += [str(args.split_k_value)]
    
    # Add optional named arguments
    if args.instance != -1:
        cmd += ["--instance", str(args.instance)]
    if args.list_instances:
        cmd += ["--list-instances"]
    
    return run_ck_profiler_cmd(cmd, capture_output)


# Get name of miopen driver, remove it from unknown
def process_miopen_driver_name(args, unknown):
    if "convint8" in unknown:
        args.data_type = "int8"
        unknown.remove("convint8")
    elif "convbfp16" in unknown:
        args.data_type = "bfp16"
        unknown.remove("convbfp16")
    elif "convfp16" in unknown:
        args.data_type = "fp16"
        unknown.remove("convfp16")
    elif "conv" in unknown:
        args.data_type = "fp32"
        unknown.remove("conv")
    else:
        print("Not supported driver (supported: conv, convfp16, convint8, convbfp16).")
        exit(1)


def run_ck_profiler(args, capture_output=False):
    """Run the CK profiler for the given args.
    
    Args:
        args: Parsed arguments
        capture_output: If True, capture and return output instead of printing
        
    Returns:
        If capture_output is True, returns a tuple (output_string, has_errors).
        Otherwise returns None.
    """
    # MIOpen get number of channel per all groups, CK profiler get number of
    # channel per group
    args.in_channels = int(args.in_channels / args.group_count)
    args.out_channels = int(args.out_channels / args.group_count)

    outputs = []
    any_errors = False
    
    if args.forw == 0 or args.forw == 1 or args.forw == 3 or args.forw == 5:
        result = run_ck_grouped_conv_fwd(args, capture_output)
        if result:
            if capture_output:
                outputs.append(result[0])
                any_errors = any_errors or result[1]
            else:
                outputs.append(result)
    if args.forw == 0 or args.forw == 2 or args.forw == 3 or args.forw == 6:
        result = run_ck_grouped_conv_bwd_data(args, capture_output)
        if result:
            if capture_output:
                outputs.append(result[0])
                any_errors = any_errors or result[1]
            else:
                outputs.append(result)
    if args.forw == 0 or args.forw == 4 or args.forw == 5 or args.forw == 6:
        result = run_ck_grouped_conv_bwd_weight(args, capture_output)
        if result:
            if capture_output:
                outputs.append(result[0])
                any_errors = any_errors or result[1]
            else:
                outputs.append(result)
    
    if capture_output:
        return ("\n".join(outputs), any_errors)
    return None


def create_conv_parser():
    """Create the argument parser for convolution parameters."""
    parser = argparse.ArgumentParser(
        prog="converter",
        description="Convert miopen driver command to ck Profiler"
        "\nExample (single command): python3 "
        "../script/convert_miopen_driver_to_profiler.py "
        "/opt/rocm/bin/MIOpenDriver conv -n 32 -c 64 -H 28 -W 28 "
        "-k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g "
        "32 -F 1 -t 1"
        "\nExample (batch mode): python3 "
        "../script/convert_miopen_driver_to_profiler.py "
        "--input-file commands.txt --output-file results.txt",
    )
    # Batch mode arguments
    parser.add_argument(
        "--input-file",
        type=str,
        required=False,
        default=None,
        help="Input file containing MIOpen driver commands (one per line). "
        "Enables batch mode.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default=None,
        help="Output file to store profiler results (required with --input-file).",
    )
    parser.add_argument(
        "--profiler-path",
        type=str,
        required=False,
        default=None,
        help="Path to ckProfiler executable (default: ../build/bin/ckProfiler).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show full profiler output. Default shows only best configuration.",
    )
    # Convolution arguments (for single command mode)
    parser.add_argument(
        "-in_layout",
        "-I",
        "--in_layout",
        "--I",
        default="NCHW",
        type=str,
        required=False,
        help="Input Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
    )
    parser.add_argument(
        "-forw",
        "-F",
        "--forw",
        "--F",
        default=0,
        type=int,
        required=False,
        help="Flag enables fwd, bwd, wrw convolutions"
        "\n0 fwd+bwd+wrw (default)"
        "\n1 fwd only"
        "\n2 bwd only"
        "\n4 wrw only"
        "\n3 fwd+bwd"
        "\n5 fwd+wrw"
        "\n6 bwd+wrw",
    )
    parser.add_argument(
        "-spatial_dim",
        "-_",
        "--spatial_dim",
        "--_",
        default=2,
        type=int,
        required=False,
        help="convolution spatial dimension (Default-2)",
    )
    parser.add_argument(
        "-batchsize",
        "-n",
        "--batchsize",
        "--n",
        default=100,
        type=int,
        required=False,
        help="Mini-batch size (Default=100)",
    )
    parser.add_argument(
        "-in_channels",
        "-c",
        "--in_channels",
        "--c",
        default=3,
        type=int,
        required=False,
        help="Number of Input Channels (Default=3)",
    )
    parser.add_argument(
        "-in_d",
        "-!",
        "--in_d",
        "--!",
        default=32,
        type=int,
        required=False,
        help="Input Depth (Default=32)",
    )
    parser.add_argument(
        "-in_h",
        "-H",
        "--in_h",
        "--H",
        default=32,
        type=int,
        required=False,
        help="Input Height (Default=32)",
    )
    parser.add_argument(
        "-in_w",
        "-W",
        "--in_w",
        "--W",
        default=32,
        type=int,
        required=False,
        help="Input Width (Default=32)",
    )
    parser.add_argument(
        "-out_channels",
        "-k",
        "--out_channels",
        "--k",
        default=32,
        type=int,
        required=False,
        help="Number of Output Channels (Default=32)",
    )
    parser.add_argument(
        "-fil_d",
        "-@",
        "--fil_d",
        "--@",
        default=3,
        type=int,
        required=False,
        help="Filter Depth (Default=3)",
    )
    parser.add_argument(
        "-fil_h",
        "-y",
        "--fil_h",
        "--y",
        default=3,
        type=int,
        required=False,
        help="Filter Height (Default=3)",
    )
    parser.add_argument(
        "-fil_w",
        "-x",
        "--fil_w",
        "--x",
        default=3,
        type=int,
        required=False,
        help="Filter Width (Default=3)",
    )
    parser.add_argument(
        "-conv_stride_d",
        "-#",
        "--conv_stride_d",
        "--#",
        default=1,
        type=int,
        required=False,
        help="Convolution Stride for Depth (Default=1)",
    )
    parser.add_argument(
        "-conv_stride_h",
        "-u",
        "--conv_stride_h",
        "--u",
        default=1,
        type=int,
        required=False,
        help="Convolution Stride for Height (Default=1)",
    )
    parser.add_argument(
        "-conv_stride_w",
        "-v",
        "--conv_stride_w",
        "--v",
        default=1,
        type=int,
        required=False,
        help="Convolution Stride for Width (Default=1)",
    )
    parser.add_argument(
        "-pad_d",
        "-$",
        "--pad_d",
        "--$",
        default=1,
        type=int,
        required=False,
        help="Zero Padding for Depth (Default=0)",
    )
    parser.add_argument(
        "-pad_h",
        "-p",
        "--pad_h",
        "--p",
        default=1,
        type=int,
        required=False,
        help="Zero Padding for Height (Default=0)",
    )
    parser.add_argument(
        "-pad_w",
        "-q",
        "--pad_w",
        "--q",
        default=1,
        type=int,
        required=False,
        help="Zero Padding for Width (Default=0)",
    )
    parser.add_argument(
        "-verify",
        "-V",
        "--verify",
        "--V",
        default=2, # By default, run verification on GPU.
        type=int,
        required=False,
        help="Verify Each Layer (Default=1)",
    )
    parser.add_argument(
        "-time",
        "-t",
        "--time",
        "--t",
        default=0,
        type=int,
        required=False,
        help="Time Each Layer (Default=0)",
    )
    parser.add_argument(
        "-dilation_d",
        "-^",
        "--dilation_d",
        "--^",
        default=1,
        type=int,
        required=False,
        help="Dilation of Filter Depth (Default=1)",
    )
    parser.add_argument(
        "-dilation_h",
        "-l",
        "--dilation_h",
        "--l",
        default=1,
        type=int,
        required=False,
        help="Dilation of Filter Height (Default=1)",
    )
    parser.add_argument(
        "-dilation_w",
        "-j",
        "--dilation_w",
        "--j",
        default=1,
        type=int,
        required=False,
        help="Dilation of Filter Width (Default=1)",
    )
    parser.add_argument(
        "-group_count",
        "-g",
        "--group_count",
        "--g",
        type=int,
        default=1,
        required=False,
        help="Number of Groups (Default=1)",
    )
    parser.add_argument(
        "-instance",
        "--instance",
        type=int,
        default=-1,
        required=False,
        help="Instance index (Default=-1)",
    )
    parser.add_argument(
        "-list-instances",
        "--list-instances",
        action="store_true",
        default=False,
        required=False,
        help="List valid instances without running",
    )
    return parser


def process_single_command(command_line, parser, capture_output=False, profiler_path=None, verbose=False):
    """Process a single MIOpen driver command line.
    
    Args:
        command_line: String containing the MIOpen driver command
        parser: The argument parser to use
        capture_output: If True, capture and return output instead of printing
        profiler_path: Optional path to ckProfiler executable
        verbose: If True, show full output; if False, show only best configuration
        
    Returns:
        If capture_output is True, returns a string with the output.
        Otherwise returns None.
    """
    # Parse the command line into arguments
    try:
        argv = shlex.split(command_line)
    except ValueError as e:
        error_msg = f"Error parsing command line: {e}\n"
        if capture_output:
            return error_msg
        print(error_msg)
        return None
    
    args, unknown = parser.parse_known_args(argv)
    init_const_args(args, profiler_path)
    process_miopen_driver_name(args, unknown)
    
    if not capture_output:
        print("Ignored args:")
        print(unknown)
    
    result = run_ck_profiler(args, capture_output)
    
    # Handle captured output
    if capture_output and result:
        output_str, has_errors = result
        if verbose:
            return output_str
        else:
            # Filter to best config and add warning if there were errors
            filtered = filter_to_best_config(output_str)
            if has_errors:
                filtered += "\nWARNING: Some kernels produced incorrect results"
            return filtered
    
    return result


def process_batch_file(input_file, output_file, parser, profiler_path=None, verbose=False):
    """Process a batch file of MIOpen driver commands.
    
    Args:
        input_file: Path to input file containing commands (one per line)
        output_file: Path to output file for results
        parser: The argument parser to use
        profiler_path: Optional path to ckProfiler executable
        verbose: If True, show full output; if False, show only best configuration
    """
    try:
        # Try UTF-8 first, fall back to UTF-16 (common on Windows)
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in:
                lines = f_in.readlines()
        except UnicodeDecodeError:
            with open(input_file, 'r', encoding='utf-16') as f_in:
                lines = f_in.readlines()
    except IOError as e:
        print(f"Error reading input file '{input_file}': {e}")
        sys.exit(1)
    
    total_lines = len(lines)
    
    # Open output file for incremental writing
    try:
        f_out = open(output_file, 'w')
    except IOError as e:
        print(f"Error opening output file '{output_file}': {e}")
        sys.exit(1)
    
    try:
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            print(f"Processing command {i}/{total_lines}: {line[:80]}...")
            
            # Write separator for readability
            f_out.write(f"{'='*80}\n")
            f_out.write(f"Input command: {line}\n")
            f_out.write(f"{'='*80}\n")
            
            # Process the command and capture output
            output = process_single_command(line, parser, capture_output=True, profiler_path=profiler_path, verbose=verbose)
            if output:
                f_out.write(output)
                f_out.write("\n")
            f_out.write("\n")  # Empty line between commands
            
            # Flush to ensure results are written immediately
            f_out.flush()
        
        print(f"\nResults written to '{output_file}'")
    finally:
        f_out.close()


if __name__ == "__main__":
    parser = create_conv_parser()
    
    # First, check if we're in batch mode
    # We need to do a preliminary parse to check for --input-file and --output-file
    preliminary_args, _ = parser.parse_known_args()
    
    if preliminary_args.input_file is not None:
        # Batch mode
        if preliminary_args.output_file is None:
            print("Error: --output-file is required when using --input-file")
            sys.exit(1)
        
        print(f"Batch mode: Reading commands from '{preliminary_args.input_file}'")
        profiler_path = preliminary_args.profiler_path
        verbose = preliminary_args.verbose
        if profiler_path:
            print(f"Using profiler: '{profiler_path}'")
        if not verbose:
            print("Output mode: best configuration only (use --verbose for full output)")
        process_batch_file(
            preliminary_args.input_file,
            preliminary_args.output_file,
            parser,
            profiler_path,
            verbose
        )
    else:
        # Single command mode (original behavior)
        args, unknown = parser.parse_known_args()
        init_const_args(args)
        process_miopen_driver_name(args, unknown)
        print("Ignored args:")
        print(unknown)
        run_ck_profiler(args)
