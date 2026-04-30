#!/bin/bash

# This script automates the setup and build process for the KDInfer_test project.

# Check and install required packages
REQUIRED_PKGS=(g++ cmake make libgmp-dev libssl-dev)
MISSING_PKGS=()

# Check g++ version
if command -v g++ >/dev/null 2>&1; then
    GPP_VER=$(g++ -dumpversion | cut -d. -f1)
    if [ "$GPP_VER" -lt 8 ]; then
        MISSING_PKGS+=("g++")
    fi
else
    MISSING_PKGS+=("g++")
fi

# Check other packages
for pkg in cmake make; do
    command -v "$pkg" >/dev/null 2>&1 || MISSING_PKGS+=("$pkg")
done
for pkg in libgmp-dev libssl-dev; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING_PKGS+=("$pkg")
done

if [ ${#MISSING_PKGS[@]} -ne 0 ]; then
    echo "Installing missing packages: ${MISSING_PKGS[*]}"
    sudo apt-get update
    sudo apt-get install -y "${MISSING_PKGS[@]}"
else
    echo "-- All required packages are already installed."
fi


# Check and install SEAL 3.3.2
if [ ! -d "SEAL" ]; then
    echo "Installing Microsoft SEAL 3.3.2..."
    git clone https://github.com/microsoft/SEAL.git
    cd SEAL || exit
    git checkout 3.3.2
    cd native/src
    cmake .
    make
    sudo make install
    cd ../../..
fi

# Check and install Eigen 3.3
if [ ! -d "eigen" ]; then
    echo "Installing Eigen 3.3..."
    wget https://gitlab.com/libeigen/eigen/-/archive/3.3/eigen-3.3.tar.gz
    tar -xzf eigen-3.3.tar.gz
    mv eigen-3.3 eigen
    rm eigen-3.3.tar.gz
else
    echo "-- Eigen 3.3 already detected."
fi

# Create a build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

# Navigate to the build directory
cd build || exit

# Run cmake and make
cmake ..
make
