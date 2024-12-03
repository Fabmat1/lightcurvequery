#!/bin/bash

# Define source file and output shared library
SOURCE_FILE="extern_gls_fast.cpp"
OUTPUT_LIB="libgls_shared.so"

# Compilation flags
CXXFLAGS="-O3 -march=native -fopenmp -funroll-loops -ftree-vectorize -fopenmp-simd -fopt-info-vec-optimized -floop-interchange -floop-block -floop-strip-mine"

# Compile the shared library
gcc $SOURCE_FILE -shared -o $OUTPUT_LIB -fPIC $CXXFLAGS

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Installation successful."
else
    echo "Failed to create shared library."
fi
