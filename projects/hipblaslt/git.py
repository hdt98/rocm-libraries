#!/usr/bin/env python3
"""
run_install_git.py

Usage:
  python3 run_install_git.py

Environment:
  SKIP_APT=1     # set to skip apt update/install step (useful if not on Debian/Ubuntu or you already installed deps)

Notes:
  - This script uses sudo for apt and make install. You will be prompted for your password.
  - It stops on the first failing command by default.
  - Logs are written to run_install_git.log in the current directory.
"""

import os
import subprocess
import sys
from datetime import datetime

LOGFILE = "run_install_git.log"
GIT_VERSION = "2.42.0"
GIT_TARBALL = f"git-{GIT_VERSION}.tar.gz"
GIT_URL = f"https://www.kernel.org/pub/software/scm/git/{GIT_TARBALL}"
SRC_DIR = "/usr/src"
BUILD_DIR = os.path.join(SRC_DIR, f"git-{GIT_VERSION}")

def log(msg):
    timestamp = datetime.now().isoformat()
    line = f"{timestamp}  {msg}"
    print(line)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")

def run(cmd, check=True, shell=False, env=None):
    log(f"Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=shell, env=env)
    log(result.stdout.strip() if result.stdout else "<no output>")
    return result.returncode

def ensure_sudo_cached():
    # Ask for sudo up front so subsequent sudo calls won't repeatedly prompt
    try:
        log("Caching sudo credentials (you may be prompted for your password)...")
        subprocess.run(["sudo", "-v"], check=True)
    except subprocess.CalledProcessError:
        log("Failed to obtain sudo credentials. Exiting.")
        sys.exit(1)

def main():
    open(LOGFILE, "w").close()
    log("Starting install script")

    skip_apt = os.environ.get("SKIP_APT", "") == "1"

    if not skip_apt:
        ensure_sudo_cached()
        # update and install build deps (Debian/Ubuntu example)
        try:
            run(["sudo", "apt", "update"])
            run(["sudo", "apt", "install", "-y", "build-essential", "libssl-dev", "libcurl4-gnutls-dev", "libexpat1-dev", "gettext"])
        except subprocess.CalledProcessError as e:
            log(f"APT step failed with exit code {e.returncode}. Exiting.")
            sys.exit(e.returncode)
    else:
        log("Skipping apt update/install step (SKIP_APT=1)")

    # Ensure source dir exists
    try:
        if not os.path.isdir(SRC_DIR):
            log(f"{SRC_DIR} does not exist, creating...")
            run(["sudo", "mkdir", "-p", SRC_DIR])
            run(["sudo", "chown", f"{os.getlogin()}:{os.getlogin()}", SRC_DIR])
    except Exception as e:
        log(f"Failed preparing {SRC_DIR}: {e}")
        sys.exit(1)

    # Change to /usr/src
    try:
        os.chdir(SRC_DIR)
        log(f"Changed working dir to {SRC_DIR}")
    except Exception as e:
        log(f"Cannot change to {SRC_DIR}: {e}")
        sys.exit(1)

    # Download tarball (use curl -LO)
    try:
        if not os.path.exists(GIT_TARBALL):
            run(["curl", "-LO", GIT_URL])
        else:
            log(f"{GIT_TARBALL} already exists, skipping download")
    except subprocess.CalledProcessError as e:
        log(f"Download failed with exit code {e.returncode}. Exiting.")
        sys.exit(e.returncode)

    # Extract tarball
    try:
        if not os.path.isdir(BUILD_DIR):
            run(["sudo", "tar", "-xzf", GIT_TARBALL])
            # If tar created directory owned by root, change ownership to current user to build without sudo
            run(["sudo", "chown", "-R", f"{os.getlogin()}:{os.getlogin()}", BUILD_DIR])
        else:
            log(f"{BUILD_DIR} already exists, skipping extract")
    except subprocess.CalledProcessError as e:
        log(f"Extraction failed with exit code {e.returncode}. Exiting.")
        sys.exit(e.returncode)

    # Build and install
    try:
        os.chdir(BUILD_DIR)
        log(f"Changed working dir to {BUILD_DIR}")
        # Configure & build using make prefix=/usr/local all
        run(["make", "prefix=/usr/local", "all"])
        # Install (requires sudo)
        run(["sudo", "make", "prefix=/usr/local", "install"])
    except subprocess.CalledProcessError as e:
        log(f"Build/install failed with exit code {e.returncode}. Exiting.")
        sys.exit(e.returncode)
    except Exception as e:
        log(f"Unexpected error during build/install: {e}")
        sys.exit(1)

    # Verify installation
    try:
        # Use full path /usr/local/bin/git if necessary or rely on PATH
        rc = run(["/usr/local/bin/git", "--version"])
        if rc != 0:
            log("Verification failed. You may need to ensure /usr/local/bin is before /usr/bin in your PATH.")
        else:
            log("Git installed successfully. Check /usr/local/bin/git --version")
    except Exception as e:
        log(f"Verification step failed: {e}")

    log("Script finished.")

if __name__ == "__main__":
    main()

