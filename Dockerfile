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

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

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

# Install DVC + other dependencies
RUN mkdir -p /etc/apt/keyrings && \
    wget -qO - https://dvc.org/deb/iterative.asc | sudo gpg --dearmor -o /etc/apt/keyrings/packages.iterative.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.iterative.gpg] https://dvc.org/deb/ stable main" | sudo tee /etc/apt/sources.list.d/dvc.list && \
    chmod 644 /etc/apt/keyrings/packages.iterative.gpg /etc/apt/sources.list.d/dvc.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    build-essential \
    cmake \
    doxygen \
    git \
    git-lfs \
    half \
    lbzip2 \
    lcov \
    libncurses5-dev \
    stunnel \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    redis \
    rocblas-dev \
    rocm-developer-tools \
    rocm-llvm-dev \
    rocrand-dev \
    rpm \
    software-properties-common \
    dvc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf amdgpu-install* && \
# Remove unnecessary rocm components that take a lot of space
    apt-get remove -y miopen-hip

RUN git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git rocm-libraries \
    && cd rocm-libraries \
    && git sparse-checkout init --cone \
    && git sparse-checkout set projects/composablekernel projects/miopen cmake \
    && git checkout $ROCM_BRANCH 

# Add requirements files
ARG PROJ="projects/miopen"
ADD ${PROJ}/rbuild.ini /rbuild.ini
ADD ${PROJ}/requirements.txt /requirements.txt
ADD ${PROJ}/docs/sphinx/requirements.txt /doc-requirements.txt

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && rm dumb-init_*.deb && \
# Install cget && rbuild
    pip3 install https://github.com/pfultz2/cget/archive/a426e4e5147d87ea421a3101e6a3beca541c8df8.tar.gz && \
    pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/6d78a0553babdaea8d2da5de15cbda7e869594b8.tar.gz && \
# Add symlink to /opt/rocm
    [ -d /opt/rocm ] || ln -sd $(realpath /opt/rocm-*) /opt/rocm && \
# Install doc requirements
    pip3 install -r /doc-requirements.txt && \
# Composable Kernel requires this version cmake
    pip3 install --upgrade cmake==3.27.5

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
      ../rocm-libraries/projects/composablekernel \
    && make -j ${MAX_JOBS} install 

# Install MIOpen dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    libsqlite3-dev sqlite3 bzip2 libbz2-dev nlohmann-json3-dev libboost-all-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 
#RUN cget install -f /requirements.txt

RUN rm -rf build && mkdir build && cd build \
    && CXX=/opt/rocm/bin/amdclang++ cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_DEV=Off \
      -DCMAKE_INSTALL_PREFIX=/opt/rocm \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DMIOPEN_USE_MLIR=Off \
      ../rocm-libraries/projects/miopen \
    && make -j ${MAX_JOBS} install
