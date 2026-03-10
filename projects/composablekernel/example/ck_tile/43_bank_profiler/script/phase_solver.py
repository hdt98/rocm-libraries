#!/usr/bin/env python3
"""
Phase Solver: Systematically test which threads are in the same phase
by checking for bank conflicts when accessing the same bank.

Uses the CK lds_probe executable under rocprofv3.
"""

import subprocess
import pandas as pd
import os
import argparse


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
        "--output-file", "phase_test",
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
            return None
    except subprocess.TimeoutExpired:
        return None

    try:
        csv_path = os.path.join(out_dir, "phase_test_counter_collection.csv")
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
        return None


def test_phase(binary, thread_i, thread_j, mode):
    """Test if two threads are in the same phase."""
    # Both threads access the same bank (offset spaced by 64*4 bytes)
    offset_i = thread_i * 64 * 4
    offset_j = thread_j * 64 * 4

    result = run_profiling(binary, thread_i, thread_j,
                           offset_i, offset_j, mode)

    if result is None:
        return None

    _, conflicts, _ = result
    return conflicts > 0


def solve_phases(binary, mode=2):
    """Solve for the phase assignment of all threads."""
    mode_names = {0: "ds_read_b64", 1: "ds_read_b96",
                  2: "ds_read_b128", 3: "ds_write_b64"}
    instr_name = mode_names.get(mode, f"mode_{mode}")

    print("=" * 60)
    print(f"Phase Solver ({instr_name}): Testing all thread pairs")
    print("=" * 60)
    print()

    NUM_THREADS = 64
    conflict_matrix = [[False] * NUM_THREADS for _ in range(NUM_THREADS)]

    total_tests = (NUM_THREADS * (NUM_THREADS - 1)) // 2
    test_count = 0

    for thread_i in range(NUM_THREADS):
        for thread_j in range(thread_i + 1, NUM_THREADS):
            test_count += 1
            print(f"[{test_count}/{total_tests}] "
                  f"Testing thread {thread_i} vs thread {thread_j}... ",
                  end="", flush=True)

            has_conflict = test_phase(binary, thread_i, thread_j, mode)

            if has_conflict is None:
                print("ERROR")
                continue

            conflict_matrix[thread_i][thread_j] = has_conflict
            conflict_matrix[thread_j][thread_i] = has_conflict

            if has_conflict:
                print("CONFLICT")
            else:
                print("no conflict")

    print()
    print("=" * 60)
    print("Grouping threads by conflict patterns")
    print("=" * 60)
    print()

    phases = {}
    phase_groups = {}
    unassigned = list(range(NUM_THREADS))
    current_phase = 0

    while unassigned:
        representative = unassigned[0]
        phase_groups[current_phase] = [representative]
        phases[representative] = current_phase
        unassigned.remove(representative)

        threads_to_remove = []
        for thread_i in unassigned:
            if conflict_matrix[representative][thread_i]:
                phases[thread_i] = current_phase
                phase_groups[current_phase].append(thread_i)
                threads_to_remove.append(thread_i)

        for thread in threads_to_remove:
            unassigned.remove(thread)

        print(f"Phase {current_phase}: "
              f"{len(phase_groups[current_phase])} threads - "
              f"{phase_groups[current_phase]}")
        current_phase += 1

    print()
    print("=" * 60)
    print("Phase Assignment Results")
    print("=" * 60)
    print()

    for phase_id in sorted(phase_groups.keys()):
        threads = phase_groups[phase_id]
        print(f"Phase {phase_id}: {len(threads)} threads - {threads}")

    print()
    print(f"Total phases detected: {len(phase_groups)}")

    with open("phase_results.txt", "w") as f:
        f.write(f"Phase Assignment Results ({instr_name})\n")
        f.write("=" * 60 + "\n\n")
        for phase_id in sorted(phase_groups.keys()):
            threads = phase_groups[phase_id]
            f.write(f"Phase {phase_id}: {len(threads)} threads - {threads}\n")
        f.write(f"\nTotal phases: {len(phase_groups)}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Conflict Matrix (1 = conflict, 0 = no conflict)\n")
        f.write("=" * 60 + "\n\n")
        f.write("   ")
        for j in range(NUM_THREADS):
            f.write(f"{j:2d} ")
        f.write("\n")
        for i in range(NUM_THREADS):
            f.write(f"{i:2d} ")
            for j in range(NUM_THREADS):
                f.write(f" {'1' if conflict_matrix[i][j] else '0'} ")
            f.write("\n")

    print()
    print("Results saved to phase_results.txt")

    return phases, phase_groups, conflict_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDS Phase Solver")
    parser.add_argument("--build-dir", default=".",
                        help="Directory containing the compiled binary")
    parser.add_argument("--mode", type=int, default=2,
                        help="0=b64, 1=b96, 2=b128, 3=write_b64")
    args = parser.parse_args()

    binary = find_binary(args.build_dir)
    print(f"Using binary: {binary}")

    phases, phase_groups, conflict_matrix = solve_phases(binary, args.mode)
