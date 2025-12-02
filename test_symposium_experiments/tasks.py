import os
import json
import time
from invoke import task

# Paths
# This script is located in test_symposium_experiments/
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
BUILD_DIR = os.path.join(REPO_ROOT, "projects", "hipblaslt", "build", "release")
TEST_BINARY = "./clients/hipblaslt-test"
LIBRARY_DIR = os.path.join(BUILD_DIR, "library")

# Environment
ENV = os.environ.copy()
# Ensure library can be found
ENV["LD_LIBRARY_PATH"] = f"{LIBRARY_DIR}:/opt/rocm/lib:{ENV.get('LD_LIBRARY_PATH', '')}"

def run_test_command(c, args="", log_prefix="test", env=None):
    """
    Helper to run the test binary, save logs, and measure time.
    """
    if env is None:
        env = ENV
        
    timestamp = int(time.time())
    log_file = os.path.join(PROJECT_ROOT, f"{log_prefix}_{timestamp}.log")
    json_file = os.path.join(PROJECT_ROOT, f"{log_prefix}_{timestamp}.json")
    
    # We use --gtest_output=json to easily parse results later
    full_cmd = f"{TEST_BINARY} {args} --gtest_output=json:{json_file}"
    
    print(f"Running: {full_cmd}")
    start_time = time.time()
    
    # Run in build directory
    with c.cd(BUILD_DIR):
        # warn=True allows the script to continue even if tests fail (which they should)
        result = c.run(f"{full_cmd} > {log_file} 2>&1", env=env, warn=True)
        
    duration = time.time() - start_time
    
    return {
        "log_file": log_file,
        "json_file": json_file,
        "duration": duration,
        "command": full_cmd,
        "failed": result.failed
    }

@task
def build(c):
    """Build the hipBLASLt project."""
    print("Building project...")
    with c.cd(BUILD_DIR):
        c.run("make -j$(nproc)")

@task
def apply_patch(c, patch_name):
    """Apply a specific patch file."""
    patch_path = os.path.join(PROJECT_ROOT, patch_name)
    print(f"Applying patch: {patch_name}")
    with c.cd(REPO_ROOT):
        c.run(f"git apply {patch_path}")

@task
def revert_all(c):
    """Revert all changes in the repo (git checkout .)."""
    print("Reverting all changes...")
    with c.cd(REPO_ROOT):
        c.run("git checkout .")

@task
def experiment_a(c):
    """
    Experiment A: Test Suite Minimization with Mutation Coverage.
    Runs all patches against the full suite to determine detection vectors.
    """
    print("=== Experiment A: Test Suite Minimization ===")
    
    # Find all .patch files
    patches = sorted([f for f in os.listdir(PROJECT_ROOT) if f.endswith(".patch")])
    
    results = {}
    
    for patch in patches:
        print(f"Testing patch: {patch}")
        try:
            revert_all(c)
            apply_patch(c, patch)
            build(c)
            
            # Run full suite
            res = run_test_command(c, args="--gtest_fail_fast=0", log_prefix=f"exp_a_{patch}")
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
    target_patch = "09_cat_b_rocblaslt_auxiliary_attr_mixup.patch" # Or 05 if preferred
    if not os.path.exists(os.path.join(PROJECT_ROOT, target_patch)):
        print(f"Warning: {target_patch} not found, picking first available.")
        patches = [f for f in os.listdir(PROJECT_ROOT) if f.endswith(".patch")]
        if patches: target_patch = patches[0]
        
    revert_all(c)
    apply_patch(c, target_patch)
    build(c)
    
    # Define partitions using gtest filters
    partitions = {
        "precision_f32": "*f32*",
        "precision_f16": "*f16*",
        "transpose_NN": "*_NN_*",
        "transpose_TN": "*_TN_*",
        "size_large": "*large*", # Assuming naming convention or use specific sizes
        "size_256": "*256*"
    }
    
    results = {}
    for name, filter_pat in partitions.items():
        print(f"Running partition: {name}")
        res = run_test_command(c, args=f"--gtest_filter={filter_pat}", log_prefix=f"exp_b_{name}")
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
    target_patch = "01_cat_a_hipblaslt_swap_rows_cols.patch"
    
    revert_all(c)
    apply_patch(c, target_patch)
    build(c)
    
    results = {}
    
    # 1. Default Order
    print("Running Default Order...")
    results["default"] = run_test_command(c, args="--gtest_fail_fast", log_prefix="exp_c_default")
    
    # 2. Randomized Order
    print("Running Randomized Order...")
    results["random"] = run_test_command(c, args="--gtest_shuffle --gtest_fail_fast", log_prefix="exp_c_random")
    
    # 3. Prioritized (Simulated)
    # We prioritize 'quick' tests or specific known-failing patterns
    print("Running Prioritized Order (Simulated)...")
    results["prioritized"] = run_test_command(c, args="--gtest_filter=*quick* --gtest_fail_fast", log_prefix="exp_c_prio")
    
    # 4. Coverage Guided (Simulated)
    # Placeholder
    print("Running Coverage Guided (Simulated)...")
    results["coverage"] = run_test_command(c, args="--gtest_filter=*matmul* --gtest_fail_fast", log_prefix="exp_c_cov")

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
    target_patch = "04_cat_a_hipblaslt_ext_stride_error.patch"
    
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
        local_env["GTEST_SHARD_INDEX"] = "0" # Always run the first shard as sample
        
        res = run_test_command(c, args="", log_prefix=f"exp_d_1_{total_shards}", env=local_env)
        results[f"1/{total_shards}"] = res
        
    with open(os.path.join(PROJECT_ROOT, "experiment_d_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    revert_all(c)
    print("Experiment D completed.")

