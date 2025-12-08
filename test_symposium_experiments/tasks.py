import os
import json
import time
import re
import datetime
from invoke import task
from invoke.exceptions import CommandTimedOut

# Paths
# This script is located in test_symposium_experiments/
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
PATCH_DIR = os.path.join(PROJECT_ROOT, "patches")
BUILD_DIR = os.path.join(REPO_ROOT, "projects", "hipblaslt", "build", "release")
TEST_BINARY = "./clients/hipblaslt-test"
LIBRARY_DIR = os.path.join(BUILD_DIR, "library")

# Environment
ENV = os.environ.copy()
# Ensure library can be found
ENV["LD_LIBRARY_PATH"] = f"{LIBRARY_DIR}:/opt/rocm/lib:{ENV.get('LD_LIBRARY_PATH', '')}"


def parse_patch_name(patch_name):
    """
    Parse patch filename into structured metadata.
    Expected format: cat_X_NN_description.patch
    Returns: {"category": "X", "number": NN, "description": "description", "full_name": "..."}
    """
    match = re.match(r"cat_([a-e])_(\d+)_(.+)\.patch", patch_name, re.IGNORECASE)
    if match:
        return {
            "category": match.group(1).upper(),
            "number": int(match.group(2)),
            "description": match.group(3),
            "full_name": patch_name,
        }
    return {
        "category": "unknown",
        "number": 0,
        "description": patch_name,
        "full_name": patch_name,
    }


def extract_log_summary(log_file):
    """
    Extract quick summary stats from a gtest log file.
    Returns dict with test counts, failure info, GPU errors, etc.
    """
    summary = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "disabled_tests": 0,
        "failed_test_names": [],
        "gpu_errors": [],
        "crashed": False,
        "completed": False,
        "first_failure": None,
    }

    if not os.path.exists(log_file):
        return summary

    try:
        with open(log_file, "r", errors="replace") as f:
            content = f.read()

        # Check for GPU errors
        gpu_errors = re.findall(r"(?:Hip error|Error code|hipError).*?(\d+)", content)
        summary["gpu_errors"] = list(set(gpu_errors))

        # Check completion
        if re.search(r"\[\s*PASSED\s*\]|\d+ tests from \d+ test suites ran", content):
            summary["completed"] = True
        else:
            summary["crashed"] = True

        # Extract test counts from final summary
        total_match = re.search(r"\[==========\] (\d+) tests? from", content)
        if total_match:
            summary["total_tests"] = int(total_match.group(1))

        passed_match = re.search(r"\[\s*PASSED\s*\]\s*(\d+)\s*tests?", content)
        if passed_match:
            summary["passed_tests"] = int(passed_match.group(1))

        failed_match = re.search(r"\[\s*FAILED\s*\]\s*(\d+)\s*tests?", content)
        if failed_match:
            summary["failed_tests"] = int(failed_match.group(1))
        
        # If total tests is still 0 (crash before summary), try to recover from start
        if summary["total_tests"] == 0:
             # Start line format: [==========] Running 40101 tests from 11 test suites.
             start_total_match = re.search(r"\[==========\] Running (\d+) tests? from", content)
             if start_total_match:
                 summary["total_tests"] = int(start_total_match.group(1))
             
             # Count [ OK ] and [ FAILED ] lines from the body
             # We use re.MULTILINE with ^ to match start of lines
             summary["passed_tests"] = len(re.findall(r"^\[\s*OK\s*\] ", content, re.MULTILINE))
             
             # For failures, look for the execution line which usually ends with time: (xx ms)
             # This helps distinguish from the summary list at the bottom if it partially exists.
             summary["failed_tests"] = len(re.findall(r"^\[\s*FAILED\s*\] .*\(", content, re.MULTILINE))

        # Extract failed test names
        failed_tests = re.findall(r"\[\s*FAILED\s*\]\s*([^\s,]+)", content)
        summary["failed_test_names"] = sorted(list(set(failed_tests)))
        
        # Sync failed_tests count if we have names but 0 count (fallback safety)
        if summary["failed_tests"] == 0 and len(summary["failed_test_names"]) > 0:
             summary["failed_tests"] = len(summary["failed_test_names"])

        # Find first failure (for FIDT analysis)
        first_fail_match = re.search(r"\[\s*FAILED\s*\]\s*([^\s,]+)", content)
        if first_fail_match:
            summary["first_failure"] = first_fail_match.group(1)

    except Exception as e:
        summary["parse_error"] = str(e)

    return summary


def run_test_command(c, args="", log_prefix="test", env=None, output_dir=None, timeout=None):
    """
    Helper to run the test binary, save logs, and measure time.
    """
    if env is None:
        env = ENV
    if output_dir is None:
        output_dir = PROJECT_ROOT

    timestamp = int(time.time())
    log_file = os.path.join(output_dir, f"{log_prefix}_{timestamp}.log")
    json_file = os.path.join(output_dir, f"{log_prefix}_{timestamp}.json")

    # We use --gtest_output=json to easily parse results later
    full_cmd = f"{TEST_BINARY} {args} --gtest_output=json:{json_file}"

    print(f"Running: {full_cmd}")
    start_time = time.time()
    
    timed_out = False
    exit_code = -1
    failed = True
    result = None

    # Run in build directory
    with c.cd(BUILD_DIR):
        try:
            # warn=True allows the script to continue even if tests fail (which they should)
            result = c.run(f"{full_cmd} > {log_file} 2>&1", env=env, warn=True, timeout=timeout)
            failed = result.failed
            exit_code = result.return_code
        except CommandTimedOut:
            print(f"TIMEOUT reached ({timeout}s)")
            timed_out = True
            # Log the timeout event to the log file so it's recorded
            with open(log_file, "a") as f:
                f.write(f"\n\n[FATAL] Test execution timed out after {timeout} seconds.\n")

    duration = time.time() - start_time

    return {
        "log_file": log_file,
        "json_file": json_file,
        "duration": duration,
        "command": full_cmd,
        "failed": failed or timed_out,
        "exit_code": exit_code,
        "timed_out": timed_out
    }


@task
def build(c):
    """Build the hipBLASLt project."""
    print("Building project...")
    with c.cd(BUILD_DIR):
        # -s: Silent mode (don't echo commands), only shows CMake progress for changed files
        c.run("make -j$(nproc) -s")


@task
def apply_patch(c, patch_name, check=False):
    """Apply a specific patch file."""
    patch_path = os.path.join(PATCH_DIR, patch_name)
    print(f"> Applying patch: {patch_name}")
    with c.cd(REPO_ROOT):
        if check:
            c.run(f"git apply --check --ignore-space-change {patch_path}")
        else:
            c.run(f"git apply --ignore-space-change {patch_path}")


@task
def revert_all(c):
    """Revert all tracked, uncommitted changes in the repo (git checkout .)."""
    print("> Reverting all changes...")
    with c.cd(REPO_ROOT):
        c.run("git checkout .")


@task
def test_patches(c):
    """Test all patches."""
    patches = sorted([f for f in os.listdir(PATCH_DIR) if f.endswith(".patch")])
    found_exception = 0

    # revert_all(c)
    for patch in patches:
        print(f"Testing patch: {patch}")
        try:
            apply_patch(c, patch, check=True)
            # revert_all(c)
        except Exception as e:
            print(f"Error applying {patch}: {e}")
            found_exception += 1

    print(
        "All patches applied successfully."
        if found_exception == 0
        else f"Some patches failed to apply. {found_exception} exceptions found."
    )


@task
def experiment_a(c):
    """
    Experiment A: Test Suite Minimization with Mutation Coverage.
    Runs all patches against the full suite to determine detection vectors.
    """
    print("=== Experiment A: Test Suite Minimization ===")

    # Find all .patch files
    patches = sorted([f for f in os.listdir(PATCH_DIR) if f.endswith(".patch")])

    results = {}

    for patch in patches:
        print(f"Testing patch: {patch}")
        try:
            revert_all(c)
            apply_patch(c, patch)
            build(c)

            # Run full suite
            res = run_test_command(
                c, args="--gtest_fail_fast=0", log_prefix=f"exp_a_{patch}"
            )
            results[patch] = res

        except Exception as e:
            print(f"Error processing {patch}: {e}")
        finally:
            revert_all(c)

    # Save results
    with open(os.path.join(PROJECT_ROOT, "experiment_a_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Experiment A completed. Results saved to experiment_a_results.json")


@task
def experiment_b(c):
    """
    Experiment B: F/T Ratio Stability.
    Runs a specific broad patch across different parameter partitions.
    """
    print("=== Experiment B: F/T Ratio Stability ===")

    # Use a broad bug (Category B) likely to cause failures but not crashes
    target_patch = "cat_b_04_rocblaslt_auxiliary_attr_mixup.patch"  # Or 05 if preferred
    if not os.path.exists(os.path.join(PROJECT_ROOT, target_patch)):
        print(f"Warning: {target_patch} not found, picking first available.")
        patches = [f for f in os.listdir(PROJECT_ROOT) if f.endswith(".patch")]
        if patches:
            target_patch = patches[0]

    revert_all(c)
    apply_patch(c, target_patch)
    build(c)

    # Define partitions using gtest filters
    partitions = {
        "precision_f32": "*f32*",
        "precision_f16": "*f16*",
        "transpose_NN": "*_NN_*",
        "transpose_TN": "*_TN_*",
        "size_large": "*large*",  # Assuming naming convention or use specific sizes
        "size_256": "*256*",
    }

    results = {}
    for name, filter_pat in partitions.items():
        print(f"Running partition: {name}")
        res = run_test_command(
            c, args=f"--gtest_filter={filter_pat}", log_prefix=f"exp_b_{name}"
        )
        results[name] = res

    with open(os.path.join(PROJECT_ROOT, "experiment_b_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    revert_all(c)
    print("Experiment B completed.")


@task
def experiment_c(c):
    """
    Experiment C: Fault Injection Detection Time.
    Runs tests in different orderings to measure time-to-first-failure.
    """
    print("=== Experiment C: FIDT ===")

    # Use a simple bug (Category A)
    target_patch = "cat_a_01_hipblaslt_swap_rows_cols.patch"

    revert_all(c)
    apply_patch(c, target_patch)
    build(c)

    results = {}

    # 1. Default Order
    print("Running Default Order...")
    results["default"] = run_test_command(
        c, args="--gtest_fail_fast", log_prefix="exp_c_default"
    )

    # 2. Randomized Order
    print("Running Randomized Order...")
    results["random"] = run_test_command(
        c, args="--gtest_shuffle --gtest_fail_fast", log_prefix="exp_c_random"
    )

    # 3. Prioritized (Simulated)
    # We prioritize 'quick' tests or specific known-failing patterns
    print("Running Prioritized Order (Simulated)...")
    results["prioritized"] = run_test_command(
        c, args="--gtest_filter=*quick* --gtest_fail_fast", log_prefix="exp_c_prio"
    )

    # 4. Coverage Guided (Simulated)
    # Placeholder
    print("Running Coverage Guided (Simulated)...")
    results["coverage"] = run_test_command(
        c, args="--gtest_filter=*matmul* --gtest_fail_fast", log_prefix="exp_c_cov"
    )

    with open(os.path.join(PROJECT_ROOT, "experiment_c_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    revert_all(c)
    print("Experiment C completed.")


@task
def experiment_d(c):
    """
    Experiment D: Naive Fractional Testing.
    Runs subsets of tests (shards) to see detection rate.
    """
    print("=== Experiment D: Fractional Testing ===")

    # Use a bug that affects specific tests (e.g. Patch 04)
    target_patch = "cat_a_04_hipblaslt_ext_stride_error.patch"

    revert_all(c)
    apply_patch(c, target_patch)
    build(c)

    # Shard counts corresponding to fractions: 1 (100%), 2 (50%), 4 (25%), 10 (10%), 20 (5%), 100 (1%)
    shards = [1, 2, 4, 10, 20, 100]
    results = {}

    for total_shards in shards:
        print(f"Running 1/{total_shards} shard...")

        # Modify env for sharding
        local_env = ENV.copy()
        local_env["GTEST_TOTAL_SHARDS"] = str(total_shards)
        local_env["GTEST_SHARD_INDEX"] = "0"  # Always run the first shard as sample

        res = run_test_command(
            c, args="", log_prefix=f"exp_d_1_{total_shards}", env=local_env
        )
        results[f"1/{total_shards}"] = res

    with open(os.path.join(PROJECT_ROOT, "experiment_d_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    revert_all(c)
    print("Experiment D completed.")


# =============================================================================
# COMPREHENSIVE DATA COLLECTION
# =============================================================================

@task
def collect_all(c, output_dir="results", category=None, patch=None, skip_build=False, timeout=1800, force=False):
    """
    Comprehensive data collection for all patches.
    
    Runs the FULL test suite for each patch and collects:
    - Complete test log (.log)
    - GTest JSON output (.json)  
    - Quick summary stats extracted from logs
    
    All data is organized in a single output directory with a master
    summary.json that serves as an index for later analysis.
    
    Resume behavior: If output_dir exists with a summary.json, patches with
    existing log files are skipped. Patches that failed to apply/build (no
    log file) are automatically retried.
    
    Options:
        --output-dir: Directory to store results (default: "results")
        --category: Only run patches from this category (a, b, c, d, e)
        --patch: Only run a specific patch file
        --skip-build: Skip the build step (use existing build)
        --timeout: Timeout in seconds for each test run (default: 1800 = 30 min)
        --force: Force re-run even if results exist (for --patch only)
    
    Example:
        invoke collect-all
        invoke collect-all --output-dir=run_2024_01_15
        invoke collect-all --category=e
        invoke collect-all --patch=cat_a_01_hipblaslt_swap_rows_cols.patch
        invoke collect-all --output-dir=results_2025-12-03_200832  # Resumes from existing
        invoke collect-all --output-dir=results --patch=cat_e_18... --force  # Force re-run
    """
    print("=" * 80)
    print("COMPREHENSIVE DATA COLLECTION")
    print("=" * 80)
    
    # Handle Resume Logic
    # If output_dir matches an existing directory, resume from it.
    # Otherwise treat it as a prefix for a new timestamped directory.
    
    target_path = os.path.join(PROJECT_ROOT, output_dir)
    is_resume = False
    
    if os.path.exists(target_path) and os.path.isdir(target_path):
        output_path = target_path
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") # Just for metadata update if needed
        is_resume = True
        print(f"RESUMING from existing directory: {output_path}")
    else:
        # Create timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = os.path.join(PROJECT_ROOT, f"{output_dir}_{timestamp}")
        os.makedirs(output_path, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    
    # Find patches to process
    all_patches = sorted([f for f in os.listdir(PATCH_DIR) if f.endswith(".patch")])
    
    # Filter by category if specified
    if category:
        cat_prefix = f"cat_{category.lower()}_"
        all_patches = [p for p in all_patches if p.startswith(cat_prefix)]
        print(f"Filtering to category {category.upper()}: {len(all_patches)} patches")
    
    # Filter by specific patch if specified
    if patch:
        if patch in all_patches:
            all_patches = [patch]
        else:
            print(f"ERROR: Patch '{patch}' not found in {PATCH_DIR}")
            return
    
    print(f"Processing {len(all_patches)} patches...")
    
    # Master summary structure
    summary_file = os.path.join(output_path, "summary.json")
    
    if is_resume and os.path.exists(summary_file):
        print(f"Loading existing summary from {summary_file}")
        with open(summary_file, "r") as f:
            master_summary = json.load(f)
    else:
        master_summary = {
            "metadata": {
                "created": timestamp,
                "output_dir": output_path,
                "total_patches": len(all_patches),
                "repo_root": REPO_ROOT,
                "build_dir": BUILD_DIR,
            },
            "patches": {},
            "categories": {
                "A": {"patches": [], "detected": 0, "escaped": 0},
                "B": {"patches": [], "detected": 0, "escaped": 0},
                "C": {"patches": [], "detected": 0, "escaped": 0},
                "D": {"patches": [], "detected": 0, "escaped": 0},
                "E": {"patches": [], "detected": 0, "escaped": 0},
            },
            "summary": {
                "total_detected": 0,
                "total_escaped": 0,
                "total_crashed": 0,
                "total_completed": 0,
                "total_timed_out": 0,
            }
        }
    
    for i, patch_name in enumerate(all_patches, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(all_patches)}] Processing: {patch_name}")
        print(f"{'='*60}")
        
        # Check if already processed (Resume Logic)
        if patch_name in master_summary["patches"] and not force:
            existing = master_summary["patches"][patch_name]
            # Consider done if we have a result status (detected/escaped) stored
            # or if log file exists.
            if existing.get("log_file") and os.path.exists(existing["log_file"]):
                print(f"  Skipping {patch_name} - Already processed. (use --force to re-run)")
                continue
        
        patch_meta = parse_patch_name(patch_name)
        patch_id = f"cat_{patch_meta['category'].lower()}_{patch_meta['number']:02d}"
        
        patch_result = {
            "patch_file": patch_name,
            "patch_id": patch_id,
            **patch_meta,
            "log_file": None,
            "json_file": None,
            "duration": 0,
            "exit_code": None,
            "build_success": False,
            "apply_success": False,
            "test_summary": {},
            "detected": False,
            "error": None,
            "timed_out": False
        }
        
        try:
            # 1. Revert any previous changes
            revert_all(c)
            
            # 2. Apply patch
            print(f"  Applying patch...")
            apply_patch(c, patch_name)
            patch_result["apply_success"] = True
            
            # 3. Build (unless skipped)
            if not skip_build:
                print(f"  Building...")
                build(c)
            patch_result["build_success"] = True
            
            # 4. Run full test suite (no fail-fast to get complete picture)
            print(f"  Running tests (timeout={timeout}s)...")
            test_res = run_test_command(
                c,
                args="",  # No filters, no fail-fast - run everything
                log_prefix=patch_id,
                output_dir=output_path,
                timeout=timeout
            )
            
            patch_result["log_file"] = test_res["log_file"]
            patch_result["json_file"] = test_res["json_file"]
            patch_result["duration"] = test_res["duration"]
            patch_result["exit_code"] = test_res.get("exit_code")
            patch_result["timed_out"] = test_res.get("timed_out", False)
            
            # 5. Extract summary from log
            print(f"  Extracting summary...")
            log_summary = extract_log_summary(test_res["log_file"])
            patch_result["test_summary"] = log_summary
            
            # 6. Determine detection status
            detected = (
                log_summary["failed_tests"] > 0 or 
                log_summary["crashed"] or
                test_res["failed"] or
                patch_result["timed_out"]
            )
            patch_result["detected"] = detected
            
            # Update category stats (handle re-runs by checking if already counted)
            cat = patch_meta["category"]
            was_previously_counted = (
                patch_name in master_summary["patches"] and 
                master_summary["patches"][patch_name].get("log_file") is not None
            )
            
            if cat in master_summary["categories"]:
                if patch_id not in master_summary["categories"][cat]["patches"]:
                    master_summary["categories"][cat]["patches"].append(patch_id)
                
                # Only update counts if not previously counted (or adjust for re-run)
                if was_previously_counted:
                    # Adjust: remove old counts before adding new
                    old_detected = master_summary["patches"][patch_name].get("detected", False)
                    if old_detected:
                        master_summary["categories"][cat]["detected"] -= 1
                        master_summary["summary"]["total_detected"] -= 1
                    else:
                        master_summary["categories"][cat]["escaped"] -= 1
                        master_summary["summary"]["total_escaped"] -= 1
                
                if detected:
                    master_summary["categories"][cat]["detected"] += 1
                else:
                    master_summary["categories"][cat]["escaped"] += 1
            
            # Update global stats
            if detected:
                master_summary["summary"]["total_detected"] += 1
            else:
                master_summary["summary"]["total_escaped"] += 1
            
            if log_summary["crashed"]:
                master_summary["summary"]["total_crashed"] += 1
            if log_summary["completed"]:
                master_summary["summary"]["total_completed"] += 1
            if patch_result["timed_out"]:
                master_summary["summary"]["total_timed_out"] = master_summary["summary"].get("total_timed_out", 0) + 1
            
            status = "DETECTED" if detected else "ESCAPED"
            if patch_result["timed_out"]:
                status += " (TIMEOUT)"
            print(f"  Result: {status} (duration: {test_res['duration']:.1f}s)")
            
        except Exception as e:
            patch_result["error"] = str(e)
            print(f"  ERROR: {e}")
        
        finally:
            # Always revert changes
            try:
                revert_all(c)
            except:
                pass
        
        # Store result
        master_summary["patches"][patch_name] = patch_result
        
        # Save intermediate summary (in case of crash)
        with open(summary_file, "w") as f:
            json.dump(master_summary, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 80)
    print("DATA COLLECTION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print(f"Summary file: {summary_file}")
    print(f"\nDetection Summary:")
    print(f"  Total patches: {len(all_patches)}")
    print(f"  Detected:      {master_summary['summary']['total_detected']}")
    print(f"  Escaped:       {master_summary['summary']['total_escaped']}")
    print(f"  Crashed:       {master_summary['summary']['total_crashed']}")
    print(f"  Completed:     {master_summary['summary']['total_completed']}")
    print(f"  Timed Out:     {master_summary['summary'].get('total_timed_out', 0)}")
    
    print("\nBy Category:")
    for cat, stats in master_summary["categories"].items():
        if stats["patches"]:
            total = len(stats["patches"])
            pct = 100 * stats["detected"] / total if total > 0 else 0
            print(f"  Category {cat}: {stats['detected']}/{total} detected ({pct:.0f}%)")
    
    return master_summary


@task
def collect_baseline(c, output_dir="baseline"):
    """
    Collect baseline test results with NO patches applied.
    This provides a reference for comparing mutated behavior.
    """
    print("=" * 80)
    print("BASELINE DATA COLLECTION (no patches)")
    print("=" * 80)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = os.path.join(PROJECT_ROOT, f"{output_dir}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    
    # Ensure clean state
    revert_all(c)
    
    # Build clean
    print("Building baseline (no patches)...")
    build(c)
    
    # Run full test suite
    print("Running baseline tests...")
    test_res = run_test_command(
        c,
        args="",
        log_prefix="baseline",
        output_dir=output_path,
    )
    
    # Extract summary
    log_summary = extract_log_summary(test_res["log_file"])
    
    # Save baseline info
    baseline_info = {
        "metadata": {
            "created": timestamp,
            "type": "baseline",
            "output_dir": output_path,
        },
        "results": {
            "log_file": test_res["log_file"],
            "json_file": test_res["json_file"],
            "duration": test_res["duration"],
            "exit_code": test_res.get("exit_code"),
            "summary": log_summary,
        }
    }
    
    summary_file = os.path.join(output_path, "baseline_summary.json")
    with open(summary_file, "w") as f:
        json.dump(baseline_info, f, indent=2)
    
    print(f"\nBaseline collection complete.")
    print(f"Results: {output_path}")
    print(f"Total tests: {log_summary['total_tests']}")
    print(f"Passed: {log_summary['passed_tests']}")
    print(f"Failed: {log_summary['failed_tests']}")
    
    return baseline_info
