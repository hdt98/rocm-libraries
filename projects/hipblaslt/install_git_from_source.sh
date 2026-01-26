#!/bin/bash
set -e  # Exit on error

# Script to install Git from source
# This installs a specific version of Git (2.42.0) from kernel.org

echo "=== Installing Git from Source ==="

# Install build dependencies (Ubuntu/Debian)
echo "Step 1: Installing build dependencies..."
sudo apt update
sudo apt install -y build-essential libssl-dev libcurl4-gnutls-dev libexpat1-dev gettext

# Pick a stable release, e.g., 2.42.0 (change to latest if needed)
GIT_VERSION="2.42.0"
echo "Step 2: Downloading Git version ${GIT_VERSION}..."
cd /usr/src
sudo curl -LO https://www.kernel.org/pub/software/scm/git/git-${GIT_VERSION}.tar.gz
sudo tar -xzf git-${GIT_VERSION}.tar.gz
cd git-${GIT_VERSION}

# Build and install
echo "Step 3: Building Git (this may take a few minutes)..."
sudo make prefix=/usr/local all

echo "Step 4: Installing Git..."
sudo make prefix=/usr/local install

# Verify installation
echo "Step 5: Verifying installation..."
echo "Git has been installed to /usr/local/bin"
/usr/local/bin/git --version

echo ""
echo "=== Installation Complete ==="
echo "Make sure /usr/local/bin is before /usr/bin in your PATH"
echo "Current PATH: $PATH"
echo ""
echo "To use the new Git version, run: /usr/local/bin/git --version"
echo "Or ensure /usr/local/bin is at the beginning of your PATH variable"
