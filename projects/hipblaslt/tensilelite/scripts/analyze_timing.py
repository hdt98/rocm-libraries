#!/usr/bin/env python3
"""
Analyze timing instrumentation output from tensilelite-client.

This script parses the TIMING: lines from stderr and generates a summary report
of where time is being spent during test execution.

Usage:
    # Run tests with timing instrumentation and capture stderr:
    pytest -s ... --global-parameters "TimingInstrumentation=True" 2>&1 | tee test_output.log

    # Analyze the output:
    python scripts/analyze_timing.py test_output.log

The timing output format is:
    TIMING:<category>:<duration_ms>
    TIMING_CONTEXT:M=...,N=...,K=...,batch=...,typeA=...,typeD=...

Categories tracked:
    Python phases:
    - python_benchmark_problems: Kernel generation and compilation
    - python_library_logic: Analyzing benchmark data
    - python_client_writer: Running the client (includes client execution)

    C++ client phases:
    - library_loading: Loading solution library
    - code_object_loading: Loading code objects
    - lazy_loading_init: Initializing lazy loading
    - data_init_setup: Setting up data initialization
    - cpu_data_init: Time to prepare CPU input data
    - cpu_reference_gemm: Time to compute reference GEMM on CPU
    - gpu_input_preparation: Preparing GPU inputs
    - rotating_buffer_preparation: Preparing rotating buffers
    - kernel_solving: Generating kernel invocations
    - warmup_runs: Warmup kernel executions
    - benchmark_runs: Benchmark kernel executions
    - gpu_kernel_execution: Time to execute the GPU kernel
"""

import sys
import re
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# Category groupings for display
CATEGORY_GROUPS = {
    "Python Phases (Total)": [
        "python_benchmark_problems",
        "python_library_logic",
        "python_client_writer",
    ],
    "Python Phases (Detail)": [
        "python_solution_generation",
        "python_kernel_compilation",
        "python_client_execution",
    ],
    "C++ Client Setup": [
        "library_loading",
        "code_object_loading",
        "lazy_loading_init",
        "data_init_setup",
        "listener_setup",
        "reporter_setup",
    ],
    "Data Preparation": [
        "pre_problem",
        "cpu_data_init",
        "gpu_input_preparation",
        "rotating_buffer_preparation",
    ],
    "Reference Computation": [
        "cpu_reference_gemm",
    ],
    "Kernel Execution": [
        "solution_selection",
        "pre_solution",
        "kernel_solving",
        "warmup_runs",
        "benchmark_runs",
        "gpu_kernel_execution",
        "post_solution",
        "post_problem",
        "finalize_report",
    ],
}

# All known categories (for detecting unknown ones)
ALL_KNOWN_CATEGORIES = set()
for cats in CATEGORY_GROUPS.values():
    ALL_KNOWN_CATEGORIES.update(cats)


@dataclass
class TimingRecord:
    category: str
    duration_ms: float
    context: Optional[Dict[str, str]] = None


@dataclass
class ProblemTiming:
    context: Dict[str, str]
    cpu_data_init_ms: float = 0.0
    cpu_reference_gemm_ms: float = 0.0
    gpu_kernel_execution_ms: float = 0.0


def parse_timing_line(line: str) -> Optional[TimingRecord]:
    """Parse a TIMING: line and return a TimingRecord."""
    match = re.match(r'TIMING:(\w+):([\d.]+)', line)
    if match:
        return TimingRecord(
            category=match.group(1),
            duration_ms=float(match.group(2))
        )
    return None


def parse_context_line(line: str) -> Optional[Dict[str, str]]:
    """Parse a TIMING_CONTEXT: line and return context dict."""
    match = re.match(r'TIMING_CONTEXT:(.*)', line)
    if match:
        context = {}
        for part in match.group(1).split(','):
            if '=' in part:
                key, value = part.split('=', 1)
                context[key] = value
        return context
    return None


def analyze_timing_file(filepath: str) -> Tuple[Dict[str, List[float]], List[ProblemTiming]]:
    """Analyze a file containing timing output and return categorized timings."""
    timings = defaultdict(list)
    current_context = None
    problem_timings = []
    current_problem = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Check for context line
            context = parse_context_line(line)
            if context:
                # Save previous problem if exists
                if current_problem:
                    problem_timings.append(current_problem)
                current_problem = ProblemTiming(context=context)
                current_context = context
                continue

            # Check for timing line
            record = parse_timing_line(line)
            if record:
                timings[record.category].append(record.duration_ms)

                # Update current problem timing
                if current_problem:
                    if record.category == 'cpu_data_init':
                        current_problem.cpu_data_init_ms = record.duration_ms
                    elif record.category == 'cpu_reference_gemm':
                        current_problem.cpu_reference_gemm_ms = record.duration_ms
                    elif record.category == 'gpu_kernel_execution':
                        current_problem.gpu_kernel_execution_ms = record.duration_ms

    # Don't forget last problem
    if current_problem:
        problem_timings.append(current_problem)

    return timings, problem_timings


def print_summary(timings: Dict[str, List[float]], problem_timings: List[ProblemTiming]):
    """Print a summary of timing data."""
    print("=" * 90)
    print("TIMING ANALYSIS SUMMARY")
    print("=" * 90)
    print()

    if not timings:
        print("No timing data found!")
        print()
        print("Make sure you ran with --global-parameters \"TimingInstrumentation=True\"")
        print("and captured stderr output (2>&1).")
        return

    # Calculate totals
    total_time_by_category = {}
    for category, values in timings.items():
        total_time_by_category[category] = sum(values)

    grand_total_ms = sum(total_time_by_category.values())
    grand_total_s = grand_total_ms / 1000.0

    # Print grouped summary
    print("TIMING BY PHASE")
    print("-" * 90)
    print(f"{'Category':<35} {'Count':>10} {'Total (s)':>12} {'Mean (ms)':>12} {'% of Total':>12}")
    print("-" * 90)

    group_totals = {}
    for group_name, categories in CATEGORY_GROUPS.items():
        group_total = 0
        group_items = []
        for category in categories:
            if category in timings:
                values = timings[category]
                count = len(values)
                total_ms = sum(values)
                total_s = total_ms / 1000.0
                mean_ms = total_ms / count if count > 0 else 0
                pct = (total_ms / grand_total_ms * 100) if grand_total_ms > 0 else 0
                group_total += total_ms
                group_items.append((category, count, total_s, mean_ms, pct))

        if group_items:
            group_pct = (group_total / grand_total_ms * 100) if grand_total_ms > 0 else 0
            group_totals[group_name] = group_total
            print(f"\n  {group_name} ({group_pct:.1f}% total)")
            print(f"  {'-' * 88}")
            for category, count, total_s, mean_ms, pct in group_items:
                print(f"    {category:<33} {count:>10} {total_s:>12.2f} {mean_ms:>12.2f} {pct:>11.1f}%")

    # Check for unknown categories
    unknown_cats = set(timings.keys()) - ALL_KNOWN_CATEGORIES
    if unknown_cats:
        print(f"\n  Other/Unknown Categories")
        print(f"  {'-' * 88}")
        for category in sorted(unknown_cats):
            values = timings[category]
            count = len(values)
            total_ms = sum(values)
            total_s = total_ms / 1000.0
            mean_ms = total_ms / count if count > 0 else 0
            pct = (total_ms / grand_total_ms * 100) if grand_total_ms > 0 else 0
            print(f"    {category:<33} {count:>10} {total_s:>12.2f} {mean_ms:>12.2f} {pct:>11.1f}%")

    print()
    print("-" * 90)
    print(f"{'TOTAL TRACKED TIME (with overlaps)':<45} {'':<10} {grand_total_s:>12.2f}")

    # Calculate non-overlapping time (avoid double counting)
    # Use the detail phases if available, otherwise use the total phases
    python_detail = sum(total_time_by_category.get(c, 0) for c in [
        "python_solution_generation", "python_kernel_compilation", "python_client_execution"
    ])
    python_total = total_time_by_category.get("python_benchmark_problems", 0)

    # If we have both, use the larger of: detail sum or total
    # (total includes all work, detail is more granular but might miss some)
    if python_detail > 0:
        python_time_for_calc = max(python_detail, python_total)
    else:
        python_time_for_calc = python_total

    # Add other non-python phases that aren't nested
    non_overlapping_total_ms = python_time_for_calc
    # Note: C++ phases are nested inside python_client_execution, so don't add them

    non_overlapping_s = non_overlapping_total_ms / 1000.0
    print(f"{'NON-OVERLAPPING TRACKED TIME':<45} {'':<10} {non_overlapping_s:>12.2f}")
    print()

    # Visual breakdown
    print("TIME BREAKDOWN (visual)")
    print("-" * 90)
    for group_name, categories in CATEGORY_GROUPS.items():
        group_total = sum(total_time_by_category.get(c, 0) for c in categories)
        if group_total > 0:
            pct = (group_total / grand_total_ms * 100) if grand_total_ms > 0 else 0
            bar_len = int(pct / 2)
            bar = '#' * bar_len
            print(f"{group_name:<30} {pct:>6.1f}% {bar}")
    print()

    # Key insights
    python_time = sum(total_time_by_category.get(c, 0) for c in CATEGORY_GROUPS.get("Python Phases", []))
    client_setup_time = sum(total_time_by_category.get(c, 0) for c in CATEGORY_GROUPS.get("C++ Client Setup", []))
    data_prep_time = sum(total_time_by_category.get(c, 0) for c in CATEGORY_GROUPS.get("Data Preparation", []))
    ref_time = sum(total_time_by_category.get(c, 0) for c in CATEGORY_GROUPS.get("Reference Computation", []))
    kernel_time = sum(total_time_by_category.get(c, 0) for c in CATEGORY_GROUPS.get("Kernel Execution", []))

    print("KEY INSIGHTS")
    print("-" * 90)
    print(f"Python phases (code gen, compilation):  {python_time/1000:.2f}s ({python_time/grand_total_ms*100:.1f}%)")
    print(f"C++ client setup (library loading):     {client_setup_time/1000:.2f}s ({client_setup_time/grand_total_ms*100:.1f}%)")
    print(f"Data preparation (CPU init, GPU prep):  {data_prep_time/1000:.2f}s ({data_prep_time/grand_total_ms*100:.1f}%)")
    print(f"Reference computation (CPU GEMM):       {ref_time/1000:.2f}s ({ref_time/grand_total_ms*100:.1f}%)")
    print(f"Kernel execution (warmup, benchmark):   {kernel_time/1000:.2f}s ({kernel_time/grand_total_ms*100:.1f}%)")
    print()

    cpu_ref_time = total_time_by_category.get('cpu_reference_gemm', 0)
    gpu_exec_time = total_time_by_category.get('gpu_kernel_execution', 0)
    if cpu_ref_time > 0 and gpu_exec_time > 0:
        ratio = cpu_ref_time / gpu_exec_time
        print(f"CPU reference / GPU execution ratio: {ratio:.1f}x")
        if ratio > 10:
            print(">>> CPU reference is a bottleneck - GPU-based reference would help significantly")
    print()

    # Top 10 slowest problems
    if problem_timings:
        print("TOP 10 SLOWEST PROBLEMS (by CPU reference time)")
        print("-" * 90)
        sorted_problems = sorted(problem_timings,
                                  key=lambda p: p.cpu_reference_gemm_ms,
                                  reverse=True)[:10]

        print(f"{'M':>8} {'N':>8} {'K':>8} {'Batch':>8} {'TypeA':>10} {'TypeD':>10} {'CPU Ref (ms)':>12} {'GPU (ms)':>10}")
        print("-" * 90)
        for p in sorted_problems:
            ctx = p.context
            print(f"{ctx.get('M', '?'):>8} {ctx.get('N', '?'):>8} {ctx.get('K', '?'):>8} "
                  f"{ctx.get('batch', '?'):>8} {ctx.get('typeA', '?'):>10} {ctx.get('typeD', '?'):>10} "
                  f"{p.cpu_reference_gemm_ms:>12.2f} {p.gpu_kernel_execution_ms:>10.2f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze timing instrumentation output from tensilelite-client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('logfile', help='Log file containing timing output')
    parser.add_argument('--csv', help='Output CSV file for detailed data')

    args = parser.parse_args()

    timings, problem_timings = analyze_timing_file(args.logfile)
    print_summary(timings, problem_timings)

    if args.csv:
        with open(args.csv, 'w') as f:
            f.write("M,N,K,batch,typeA,typeD,cpu_data_init_ms,cpu_reference_gemm_ms,gpu_kernel_execution_ms\n")
            for p in problem_timings:
                ctx = p.context
                f.write(f"{ctx.get('M', '')},{ctx.get('N', '')},{ctx.get('K', '')},"
                       f"{ctx.get('batch', '')},{ctx.get('typeA', '')},{ctx.get('typeD', '')},"
                       f"{p.cpu_data_init_ms},{p.cpu_reference_gemm_ms},{p.gpu_kernel_execution_ms}\n")
        print(f"Detailed data written to: {args.csv}")


if __name__ == '__main__':
    main()
