# 第一阶段：构建阶段
FROM nvidia/cuda:12.5.0-base-ubuntu22.04 AS build

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.9.0-0
ARG PYTORCH_VERSION=*cuda*

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

# 安装基本依赖并安装 Miniforge
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        tini \
        > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

COPY . .

# 安装必要的 Python 包
RUN mamba install pytorch=*=${PYTORCH_VERSION} -c nvidia && \
    mamba install dpdata ase numpy scipy && \
    mamba install pymatgen dargs cp2kdata && \
    mamba install cython setuptools setuptools_scm wheel boost && \
    mamba clean -ai -y && \
    mamba init

# 安装gcc编译器
RUN apt-get update && apt-get install build-essential -y
# 构建 wheel 文件
RUN python setup.py bdist_wheel

# 第二阶段：最终镜像
FROM nvidia/cuda:12.5.0-base-ubuntu22.04

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.9.0-0

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

# 安装基本依赖并安装 Miniforge
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        tini \
        > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

# 安装必要的 Python 包
RUN mamba install pytorch=*=${PYTORCH_VERSION} -c nvidia && \
    mamba install dpdata ase numpy scipy && \
    mamba install pymatgen dargs cp2kdata -c conda-forge && \
    mamba install deprecation pynauty networkx tenacity pymace -c conda-forge && \
    mamba clean -ai -y && \
    mamba init
RUN python -c "from mace.calculators import mace_anicc, mace_mp, mace_off; mace_mp('small');mace_mp('medium');mace_mp('large')"

# 复制构建阶段生成的 wheel 文件并安装
COPY --from=build dist/*.whl ./
RUN pip install ./*.whl

ENTRYPOINT ["tini", "--"]
CMD ["/bin/bash"]
