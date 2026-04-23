#!/bin/bash
# Watchdog wrapper that monitors test execution and captures diagnostics on hang
# Usage: watchdog_wrapper.sh <timeout_seconds> <test_executable> [test_args...]

TIMEOUT=$1
shift
TEST_CMD="$@"

# Calculate when to capture diagnostics (30s before timeout)
CAPTURE_AT=$((TIMEOUT - 30))
if [ $CAPTURE_AT -lt 60 ]; then
    # For very short timeouts, capture halfway through
    CAPTURE_AT=$((TIMEOUT / 2))
fi

# Use rocwmma ptrace wrapper to enable gdb attachment without sudo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTRACE_WRAPPER="$SCRIPT_DIR/rocwmma_ptrace_wrapper"

# Run test in background
if [ -x "$PTRACE_WRAPPER" ]; then
    $PTRACE_WRAPPER $TEST_CMD &
    PID=$!
else
    echo "Warning: rocwmma_ptrace_wrapper not found, running without gdb support"
    $TEST_CMD &
    PID=$!
fi

# Monitor test execution
START_TIME=$(date +%s)
DIAGNOSTICS_CAPTURED=false

while kill -0 $PID 2>/dev/null; do
    ELAPSED=$(($(date +%s) - START_TIME))

    # Capture diagnostics once when approaching timeout
    if [ $ELAPSED -ge $CAPTURE_AT ] && [ "$DIAGNOSTICS_CAPTURED" = "false" ]; then
        echo ""
        echo "======================================================================"
        echo "WATCHDOG: Test approaching timeout ($ELAPSED/${TIMEOUT}s), capturing diagnostics"
        echo "======================================================================"

        # Userspace stack traces (gdb)
        echo ""
        echo "=== Userspace Stack Traces (gdb) ==="
        if command -v gdb >/dev/null 2>&1; then
            gdb -batch -ex "set pagination off" -ex "set debuginfod enabled off" \
                -ex "thread apply all bt" -p $PID 2>&1 | grep -v "^Debuginfod"
        else
            echo "gdb not installed"
        fi

        # Kernel-level thread info (always useful, shows what kernel function threads are blocked in)
        echo ""
        echo "=== Thread Kernel Status (wchan - what each thread is waiting on) ==="
        for task in /proc/$PID/task/*; do
            if [ -d "$task" ]; then
                tid=$(basename $task)
                wchan=$(cat $task/wchan 2>/dev/null || echo "unknown")
                status=$(grep "^State:" $task/status 2>/dev/null | awk '{print $2 $3}')
                name=$(grep "^Name:" $task/status 2>/dev/null | cut -f2)
                echo "Thread $tid ($name): state=$status, waiting_in=$wchan"
            fi
        done

        # Kernel messages (GPU errors, page faults, etc.)
        echo ""
        echo "=== Kernel Messages (dmesg, last 50 lines) ==="
        dmesg | tail -50 2>&1 || echo "dmesg unavailable or no permissions"

        # GPU state
        echo ""
        echo "=== AMD GPU Recovery State ==="
        cat /sys/kernel/debug/dri/*/amdgpu_gpu_recover 2>&1 || echo "GPU state unavailable"

        # Process info
        echo ""
        echo "=== Process State ==="
        ps -p $PID -o pid,ppid,state,wchan:20,comm 2>&1 || true

        # Open files (useful for seeing what GPU devices are open)
        echo ""
        echo "=== Open Files and Devices (lsof) ==="
        if command -v lsof >/dev/null 2>&1; then
            lsof -p $PID 2>&1 | grep -E "COMMAND|REG|CHR|/dev|kfd|render" | head -20 || echo "lsof failed"
        else
            echo "lsof not installed - showing /proc/$PID/fd"
            ls -l /proc/$PID/fd 2>&1 | head -10 || echo "(no permission)"
        fi

        echo ""
        echo "======================================================================"
        echo "WATCHDOG: Diagnostics captured, waiting for test to complete or timeout"
        echo "======================================================================"
        echo ""

        DIAGNOSTICS_CAPTURED=true
    fi

    sleep 1
done

# Wait for test to finish and get exit code
wait $PID
exit $?
