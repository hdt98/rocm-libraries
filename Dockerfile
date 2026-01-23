FROM ubuntu:22.04 as miopen

ARG PROJ="miopen"
ARG DEBIAN_FRONTEND=noninteractive

# Set build argument for parallel build jobs
ARG MAX_JOBS=128
ENV MAX_JOBS=${MAX_JOBS}
ARG GPU_ARCHS=gfx942
ENV GPU_ARCHS=${GPU_ARCHS}
ARG ROCM_BRANCH="ebb11d75f1ac355735942d7f843c3810d9489cd4"
ENV ROCM_BRANCH=${ROCM_BRANCH}

# Set up the basics, before taking $ROCM_BRANCH into account.
RUN apt update \
    && apt -y --fix-broken install \
    && apt install -y git gfortran cmake build-essential \
    && apt clean

RUN git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git rocm-libraries \
    && cd rocm-libraries \
    && git sparse-checkout init --cone \
    && git sparse-checkout set projects/composablekernel projects/miopen cmake \
    && git checkout $ROCM_BRANCH 

# Copy Composable Kernel (MIOpen expects this path)
COPY projects/composablekernel projects/composable_kernel

# Support multiarch
RUN dpkg --add-architecture i386 && \
# Install preliminary dependencies and add rocm gpg key
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
        apt-utils ca-certificates curl libnuma-dev gnupg2 wget  && \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm-keyring.gpg && \
# Get and install amdgpu-install.
    wget https://repo.radeon.com/amdgpu-install/7.1/ubuntu/jammy/amdgpu-install_7.1.70100-1_all.deb --no-check-certificate && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
       ./amdgpu-install_7.1.70100-1_all.deb

# Add rocm repository
RUN export ROCM_APT_VER=7.1; \
    echo $ROCM_APT_VER &&\
    sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/amdgpu/30.20/ubuntu jammy main > /etc/apt/sources.list.d/amdgpu.list' &&\
    sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/rocm/apt/$ROCM_APT_VER jammy main > /etc/apt/sources.list.d/rocm.list' && \
# Install rocm components
    sh -c "echo deb http://mirrors.kernel.org/ubuntu jammy main universe | tee -a /etc/apt/sources.list" && \
    amdgpu-install -y --usecase=rocm --no-dkms

# Install the ROCM libraries
RUN echo "ROCM_BRANCH=$ROCM_BRANCH, GPU_ARCH=$GPU_ARCHS"
RUN mkdir build && cd build \
    && export ROCM_PATH=/opt/rocm \
    && CXX=/opt/rocm/bin/amdclang++ cmake \
      -D CMAKE_PREFIX_PATH=/opt/rocm \
      -D CMAKE_BUILD_TYPE=Release \
      -D GPU_ARCHS=${GPU_ARCHS} \
      -D MIOPEN_REQ_LIBS_ONLY=ON \
      -D CMAKE_CXX_FLAGS=" -O3 " \
      ../projects/composable_kernel \
    && make -j ${MAX_JOBS} install \   
    && cd .. && rm -rf build && mkdir build && cd build \
    && CXX=/opt/rocm/llvm/bin/clang++ CXXFLAGS='-Werror' cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_DEV=Off \
      -DCMAKE_INSTALL_PREFIX=/opt/rocm \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      ../projects/miopen \
    && make -j ${MAX_JOBS} install \
    # Remove CK source once built and installed
    && rm -rf /composable_kernel
    