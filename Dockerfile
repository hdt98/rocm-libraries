# ============================================================================
# Base Image and Build Arguments
# ============================================================================
FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.10.0 AS ck-builder

# Build configuration
ARG MAX_JOBS=128
ENV MAX_JOBS=${MAX_JOBS}

# GPU architecture target
ARG GPU_ARCHS=gfx950
ENV GPU_ARCHS=${GPU_ARCHS}

# ============================================================================
# Basic Dependencies
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        python3-pip \
        numactl \
        gfortran \
        cmake \
        build-essential \
        libsqlite3-dev \
        libbz2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir joblib pyyaml msgpack amdsmi

# Install MLPerf automation tools (MLCFlow + CLI) globally
RUN pip install --break-system-packages mlc-scripts mlcflow

# Pull the MLPerf automation repository via MLCFlow (recommended)
# NOTE: uses the 'mlc' CLI from mlcflow to fetch the correct repo/branch
RUN mlc pull repo --url=mlcommons@mlperf-automations --branch=dev

# ============================================================================
# Build and Install Apex
# ============================================================================

ARG APEX_BRANCH="a571543e240a5d91a9745cff505e888acd427135"
ENV APEX_BRANCH=${APEX_BRANCH}

# Clone Apex repository at specified commit
RUN echo "Building Apex from branch/commit: $APEX_BRANCH" && \
    git clone https://github.com/rocm/apex /apex && \
    cd /apex && \
    git checkout ${APEX_BRANCH}

# Build and install Apex with C++ and CUDA extensions
RUN cd /apex && \
    pip uninstall -y apex || true && \
    PYTORCH_ROCM_ARCH=${GPU_ARCHS} \
    APEX_BUILD_AMP_C=1 \
    APEX_BUILD_APEX_C=1 \
    APEX_BUILD_DISTRIBUTED_ADAM=1 \
    APEX_BUILD_FOCAL_LOSS=1 \
    APEX_BUILD_FUSED_ADAM=1 \
    APEX_BUILD_FUSED_CONV_BIAS_RELU=1 \
    pip install -v \
        --no-cache-dir --no-build-isolation \
        ./ && \
    cd / && rm -rf /apex

# ============================================================================
# Install package dependencies for rocJPEG and rocAL
# ============================================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libva-dev \
        libva2 \
        libva-drm2 \
        libva-x11-2 \
        mesa-va-drivers \
        mivisionx-dev \
        half \
        liblmdb-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libturbojpeg0-dev \
        libopencv-dev \
        python3.12-dev \
        python3-wheel \
        libstdc++-12-dev \
        libomp-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build and install hipcc from source (required by rocJPEG and MIOpen)
RUN cd / && \
    git clone --depth 1 --branch amd-staging https://github.com/ROCm/llvm-project.git hipcc-src && \
    cd hipcc-src/amd/hipcc && \
    mkdir build && cd build && \
    cmake .. && \
    make -j ${MAX_JOBS} && \
    ls -la && \
    cp hipcc hipconfig /opt/rocm/bin/ && \
    chmod +x /opt/rocm/bin/hipcc /opt/rocm/bin/hipconfig && \
    ls -la /opt/rocm/bin/hipcc /opt/rocm/bin/hipconfig && \
    cd / && rm -rf hipcc-src

# Build and Install rocJPEG from source
RUN echo "Verifying hipcc installation:" && \
    ls -la /opt/rocm/bin/hipcc /opt/rocm/bin/hipconfig && \
    /opt/rocm/bin/hipcc --version && \
    cd /opt && \
    git clone https://github.com/ROCm/rocJPEG.git && \
    cd rocJPEG && \
    mkdir build && cd build && \
    cmake -DCMAKE_PREFIX_PATH=/opt/rocm .. && \
    make -j ${MAX_JOBS} && \
    make install && \
    cd / && rm -rf /opt/rocJPEG

# Install PyBind11 from source (required version: v2.11.1)
RUN cd /tmp && \
    git clone https://github.com/pybind/pybind11.git && \
    cd pybind11 && \
    git checkout v2.11.1 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j ${MAX_JOBS} install && \
    cd / && rm -rf /tmp/pybind11

# Install RapidJSON from source (required: master branch)
RUN cd /tmp && \
    git clone https://github.com/Tencent/rapidjson.git && \
    cd rapidjson && \
    git checkout 24b5e7a8b27f42fa16b96fc70aade9106cf7102f && \
    mkdir build && cd build && \
    cmake .. && \
    make -j ${MAX_JOBS} install && \
    cd / && rm -rf /tmp/rapidjson

# Clone and build rocAL
RUN cd /opt && \
    git clone https://github.com/ROCm/rocAL.git rocAL-src && \
    cd rocAL-src && \
    git checkout release/rocm-rel-7.2 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/rocal \
        -DPYTHON_VERSION_SUGGESTED=3.12 && \
    make -j ${MAX_JOBS} && \
    make install && \
    make PyPackageInstall && \
    cd / && rm -rf /opt/rocAL-src

# Set environment variables for rocAL and OpenMP
ENV PYTHONPATH=/opt/rocal/lib
ENV LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH

# ============================================================================
# Clone ROCm Libraries Source
# ============================================================================

ARG ROCM_LIBS_BRANCH="2b85c2e40499db92b63aa427f3ae3a6cbeed59f0"
ENV ROCM_LIBS_BRANCH=${ROCM_LIBS_BRANCH}

WORKDIR /
RUN echo "ROCM_LIBS_BRANCH=$ROCM_LIBS_BRANCH, GPU_ARCH=$GPU_ARCHS" && \
    git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git rocm-libraries && \
    cd rocm-libraries && \
    git sparse-checkout init --cone && \
    git sparse-checkout set projects/miopen projects/composablekernel && \
    git checkout $ROCM_LIBS_BRANCH

# ============================================================================
# Build and Install ComposableKernel (CK)
# ============================================================================
RUN echo "Building CK from branch/commit: $ROCM_LIBS_BRANCH" \
    && mkdir -p /build_ck && cd /build_ck \
    && export ROCM_PATH=/opt/rocm \
    && CXX=/opt/rocm/llvm/bin/amdclang++ \
       CC=/opt/rocm/llvm/bin/amdclang \
       cmake \
      -D CMAKE_PREFIX_PATH=/opt/rocm \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/opt/composable_kernel \
      -D GPU_TARGETS=${GPU_ARCHS} \
      -D MIOPEN_REQ_LIBS_ONLY=ON \
      -D CMAKE_CXX_FLAGS=" -O3 " \
      /rocm-libraries/projects/composablekernel \
    && make -j ${MAX_JOBS} install \
    && cd / && rm -rf /build_ck /rocm-libraries/projects/composablekernel

# ============================================================================
# Stage 2: Build MIOpen using ROCm MIOpen CI image
# ============================================================================
FROM rocm/miopen:ci_281db1_gfx942_gfx950 AS miopen-builder

ARG MAX_JOBS=128
ARG GPU_ARCHS=gfx950
ARG ROCM_LIBS_BRANCH="2b85c2e40499db92b63aa427f3ae3a6cbeed59f0"

# Copy CK artifacts and MIOpen source from ck-builder
COPY --from=ck-builder /opt/composable_kernel /opt/composable_kernel
COPY --from=ck-builder /rocm-libraries/projects/miopen /rocm-libraries/projects/miopen

# Install MIOpen dependencies using install_deps.cmake
RUN cd /rocm-libraries/projects/miopen && \
    sed -i 's|^\(.*composablekernel.*\)|#\1|g' requirements.txt && \
    export CXX=/opt/rocm/llvm/bin/amdclang++ && \
    export CC=/opt/rocm/llvm/bin/amdclang && \
    mkdir -p install_dir && \
    cmake -P install_deps.cmake --minimum --prefix ./install_dir

# Build and install MIOpen to dedicated prefix
RUN echo "Building MIOpen from branch/commit: $ROCM_LIBS_BRANCH" && \
    mkdir /build_miopen && cd /build_miopen && \
    CXX=/opt/rocm/llvm/bin/amdclang++ \
    CC=/opt/rocm/llvm/bin/amdclang \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_DEV=Off \
        -DBUILD_TESTING=Off \
        -DCMAKE_INSTALL_PREFIX=/opt/miopen \
        -Dcomposable_kernel_DIR=/opt/composable_kernel/lib/cmake/composable_kernel \
        -DCMAKE_CXX_FLAGS="-I/opt/composable_kernel/include" \
        -DCMAKE_PREFIX_PATH="/opt/composable_kernel;/rocm-libraries/projects/miopen/install_dir;/opt/rocm" \
        -DGPU_TARGETS=${GPU_ARCHS} \
        -DMIOPEN_USE_MLIR=OFF \
        -DMIOPEN_USE_COMPOSABLEKERNEL=ON \
        /rocm-libraries/projects/miopen && \
    make -j ${MAX_JOBS} install

# ============================================================================
# Stage 3: Final image — ck-builder base with MIOpen copied in
# ============================================================================
FROM ck-builder

# Copy MIOpen from miopen-builder
COPY --from=miopen-builder /opt/miopen/ /opt/rocm/

# Clean up MIOpen source left over from ck-builder
RUN rm -rf /rocm-libraries

# ============================================================================
# Install HIP extensions for RetinaNet 
# ============================================================================

# Copy retinanet code (specifically csrc/hip_extensions)
WORKDIR /retinanet-training
COPY retinanet/csrc/hip_extensions /retinanet-training/retinanet/csrc/hip_extensions
 
# Set PYTHONPATH and build HIP extensions
ENV PYTHONPATH=/retinanet-training/retinanet:/opt/rocal/lib
RUN cd /retinanet-training/retinanet/csrc/hip_extensions && \
    chmod +x build.sh && \
    ./build.sh && \
    # Clean up build artifacts to reduce image size
    rm -rf build/ dist/ *.egg-info/ *.pyc __pycache__

# ============================================================================
# Final Cleanup
# ============================================================================

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /
