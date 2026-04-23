/*
 * rocWMMA ptrace wrapper - enables ptrace for child processes using PR_SET_PTRACER
 * This allows gdb to attach without changing global ptrace_scope
 *
 * Compile: gcc -o rocwmma_ptrace_wrapper rocwmma_ptrace_wrapper.c
 * Usage: rocwmma_ptrace_wrapper <command> [args...]
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <command> [args...]\n", argv[0]);
        fprintf(stderr, "Runs command with PR_SET_PTRACER enabled (allows gdb attach)\n");
        return 1;
    }

    // Enable ptrace for any process
    if (prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0) == -1) {
        fprintf(stderr, "Warning: PR_SET_PTRACER failed: %s\n", strerror(errno));
        fprintf(stderr, "gdb may not be able to attach\n");
    }

    // Execute the command
    execvp(argv[1], &argv[1]);

    // If execvp returns, it failed
    fprintf(stderr, "Failed to execute %s: %s\n", argv[1], strerror(errno));
    return 1;
}
