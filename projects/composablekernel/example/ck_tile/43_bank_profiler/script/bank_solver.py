#!/usr/bin/env python3
"""
Bank Solver: Determine the number of LDS banks by detecting wraparound.

Uses the CK lds_probe executable under rocprofv3 to measure bank conflicts.
"""

import subprocess
import pandas as pd
import os
import argparse
import sys


def find_binary(build_dir):
    """Locate the lds_probe binary."""
    candidates = [
        os.path.join(build_dir, "tile_bank_profiler_lds_probe"),
        os.path.join(build_dir, "bin", "tile_bank_profiler_lds_probe"),
        os.path.join(build_dir, "example", "ck_tile", "16_bank_profiler",
                     "tile_bank_profiler_lds_probe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fallback: search
    for root, dirs, files in os.walk(build_dir):
        if "tile_bank_profiler_lds_probe" in files:
            return os.path.join(root, "tile_bank_profiler_lds_probe")
    raise FileNotFoundError(
        f"Cannot find tile_bank_profiler_lds_probe in {build_dir}")


def run_profiling(binary, thread_a, thread_b, offset_a, offset_b, mode,
                  out_dir="out"):
    """Run the probe kernel under rocprofv3 and return conflict info."""
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "rocprofv3",
        "--pmc", "SQ_INSTS_LDS", "SQ_LDS_BANK_CONFLICT",
        "--output-format", "csv",
        "--output-file", "bank_test",
        "-d", out_dir,
        "--",
        binary,
        f"-thread_a={thread_a}",
        f"-thread_b={thread_b}",
        f"-offset_a={offset_a}",
        f"-offset_b={offset_b}",
        f"-mode={mode}",
        "-repeat=1",
    ]

    try:
        result = subprocess.run(cmd, timeout=30,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            print(f"\nrocprofv3 failed with return code {result.returncode}")
            return None
    except subprocess.TimeoutExpired:
        print("Timeout!")
        return None

    try:
        csv_path = os.path.join(out_dir, "bank_test_counter_collection.csv")
        df = pd.read_csv(csv_path)
        df = df[df["Counter_Name"].isin(
            ["SQ_INSTS_LDS", "SQ_LDS_BANK_CONFLICT"])]

        pivot = df.pivot_table(
            index=["Dispatch_Id", "Kernel_Name"],
            columns="Counter_Name",
            values="Counter_Value",
            aggfunc="first").reset_index()

        last_dispatch = pivot.iloc[-1]
        conflicts = last_dispatch["SQ_LDS_BANK_CONFLICT"]
        lds_insts = last_dispatch["SQ_INSTS_LDS"]
        conflict_ratio = conflicts / lds_insts if lds_insts > 0 else 0

        return lds_insts, conflicts, conflict_ratio
    except Exception as e:
        print(f"\nError reading profiling data: {e}")
        return None


def test_bank_conflict(binary, thread_0, thread_1, offset_0, offset_1, mode):
    """Test if two threads have a bank conflict at the given offsets."""
    result = run_profiling(binary, thread_0, thread_1, offset_0, offset_1,
                           mode)
    if result is None:
        return None
    lds_insts, conflicts, conflict_ratio = result
    has_conflict = conflicts > 0
    return has_conflict, conflicts, lds_insts


def solve_num_banks(binary, mode=2):
    """Solve for the number of LDS banks by testing wraparound."""
    mode_names = {0: "ds_read_b64", 1: "ds_read_b96",
                  2: "ds_read_b128", 3: "ds_write_b64"}
    mode_banks = {0: 2, 1: 3, 2: 4, 3: 2}  # banks per access
    instr_name = mode_names.get(mode, f"mode_{mode}")
    banks_per_access = mode_banks.get(mode, 2)

    print("=" * 70)
    print(f"Bank Solver ({instr_name}): Determining the number of LDS banks")
    print("=" * 70)
    print()

    thread_0 = 0
    thread_1 = 1
    print(f"Using threads {thread_0} and {thread_1}")
    print()

    offset_0 = 0
    banks_0 = list(range(banks_per_access))
    print(f"Thread {thread_0}: offset={offset_0} bytes -> banks {banks_0}")
    print()

    print("Testing thread 1 at different bank offsets...")
    print()
    print(f"{'Test':<6} {'Bank':<8} {'Offset':<10} {'Conflict':<10} {'Details'}")
    print("-" * 70)

    max_bank_to_test = 128
    test_banks = list(range(banks_per_access, max_bank_to_test, 1))

    first_conflict_bank = None
    conflict_banks = []

    for test_num, bank_1 in enumerate(test_banks, 1):
        offset_1 = bank_1 * 4
        banks_1 = [bank_1 + i for i in range(banks_per_access)]

        result = test_bank_conflict(binary, thread_0, thread_1,
                                    offset_0, offset_1, mode)

        if result is None:
            print(f"{test_num:<6} {bank_1:<8} {offset_1:<10} ERROR")
            continue

        has_conflict, conflicts, lds_insts = result
        conflict_str = "YES" if has_conflict else "no"
        detail_str = (f"banks {banks_1}, conflicts={conflicts}/{lds_insts}")
        print(f"{test_num:<6} {bank_1:<8} {offset_1:<10} "
              f"{conflict_str:<10} {detail_str}")

        if has_conflict:
            conflict_banks.append(bank_1)
            if first_conflict_bank is None:
                first_conflict_bank = bank_1
                num_banks = bank_1
                wrapped_banks = [b % num_banks for b in banks_1]
                overlap = [b for b in wrapped_banks if b in banks_0]

                print()
                print("!" * 70)
                print(f"FIRST CONFLICT DETECTED at bank {bank_1}!")
                print("!" * 70)
                print()
                print(f"Thread 1 banks {banks_1} wrap to {wrapped_banks} "
                      f"(mod {num_banks})")
                print(f"Overlaps with thread 0 banks {banks_0} at: {overlap}")
                print()
                print("Continuing to verify pattern...")
                print()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if first_conflict_bank is None:
        print(f"WARNING: No conflict detected up to bank {max_bank_to_test}")
        print("=" * 70)
        return None

    num_banks = first_conflict_bank
    print(f"Number of LDS banks: {num_banks}")
    print()
    print(f"Conflicts detected at banks: {conflict_banks}")
    print()

    print("Verification:")
    for bank in conflict_banks:
        wrapped = [(bank + i) % num_banks for i in range(banks_per_access)]
        overlaps = [b for b in wrapped if b in banks_0]
        if overlaps:
            print(f"  Bank {bank}: reads banks "
                  f"{[bank + i for i in range(banks_per_access)]} -> "
                  f"wraps to {wrapped} -> overlaps at {overlaps}")

    print("=" * 70)

    with open("bank_results.txt", "w") as f:
        f.write(f"LDS Bank Detection Results ({instr_name})\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Number of LDS banks: {num_banks}\n\n")
        f.write(f"Methodology:\n")
        f.write(f"  - Thread {thread_0} reads from banks {banks_0} "
                f"(offset {offset_0})\n")
        f.write(f"  - Thread {thread_1} tested progressively higher banks\n")
        f.write(f"  - First conflict at bank {first_conflict_bank}\n")
        f.write(f"  - All conflicts: {conflict_banks}\n")
        f.write(f"  - Confirms {num_banks} banks in LDS\n\n")
        f.write(f"Conflict Pattern:\n")
        for bank in conflict_banks[:10]:
            wrapped = [(bank + i) % num_banks
                       for i in range(banks_per_access)]
            overlaps = [b for b in wrapped if b in banks_0]
            f.write(f"  Bank {bank}: "
                    f"{[bank + i for i in range(banks_per_access)]} -> "
                    f"{wrapped} -> overlap {overlaps}\n")

    print()
    print("Results saved to bank_results.txt")
    return num_banks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDS Bank Solver")
    parser.add_argument("--build-dir", default=".",
                        help="Directory containing the compiled binary")
    parser.add_argument("--mode", type=int, default=2,
                        help="0=b64, 1=b96, 2=b128, 3=write_b64")
    args = parser.parse_args()

    binary = find_binary(args.build_dir)
    print(f"Using binary: {binary}")

    num_banks = solve_num_banks(binary, args.mode)

    if num_banks:
        print(f"\nSuccessfully determined: {num_banks} LDS banks")
    else:
        print("\nCould not determine number of banks")
