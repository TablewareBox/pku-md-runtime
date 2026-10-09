# Production sampling plus DMFF training.
#
# This file does not COPY a build context, so it can be pasted into a Bohrium
# Dockerfile build. It clones this repository at RUNTIME_REV and places the
# files on the absolute paths the production scripts use.
#
#   docker build --platform linux/amd64 -t pku-md:20261010 .
#
# Bump RUNTIME_REV when the plugins, force field, boxes, or bridge change.
# OpenMM stays on the conda-forge 8.4 release. Do not use 8.6.1, and do not
# set JAX_PLATFORMS=gpu. Do not swap in another plugin binary.

# Docker Hub's CloudFront endpoint times out from the Bohrium builder.
FROM docker.m.daocloud.io/library/ubuntu:24.04

ARG RUNTIME_REPO=https://github.com/TablewareBox/pku-md-runtime.git
ARG RUNTIME_REV=deedb81ed5b74f1db1ffa82aa81d16f57a5d7fe5
ARG DMFF_REV=809f656153018498e3260e07162197139ba103d6

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    CONDA_DIR=/home/changjh/miniconda3 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN sed -i \
        -e 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' \
        -e 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' \
        /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git bzip2 build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=${CONDA_DIR}/envs/md/bin:${CONDA_DIR}/bin:${PATH}

RUN curl -fL --retry 5 --retry-all-errors -o /tmp/miniforge.sh \
        https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p "$CONDA_DIR" \
    && rm /tmp/miniforge.sh \
    && ("$CONDA_DIR/bin/conda" config --system --remove channels defaults || true) \
    && "$CONDA_DIR/bin/conda" config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
    && "$CONDA_DIR/bin/conda" config --system --set channel_priority strict

RUN conda create -y -n md \
        python=3.11.17 \
        openmm=8.4 \
        cuda-version=12.6 \
        numpy=2.4.6 \
    && conda clean -afy

# RUNTIME_REV holds the inputs. DMFF upstream is checked out separately and
# then covered by dmff-overlay, because wangxy/v1.0.0-devel still imports jax.config.
RUN set -eu; \
    git init /tmp/pku-md-runtime; \
    git -C /tmp/pku-md-runtime remote add origin "$RUNTIME_REPO"; \
    git -C /tmp/pku-md-runtime fetch --depth 1 origin "$RUNTIME_REV"; \
    git -C /tmp/pku-md-runtime -c advice.detachedHead=false checkout --detach FETCH_HEAD; \
    (cd /tmp/pku-md-runtime && sha256sum -c SHA256SUMS); \
    git clone --filter=blob:none https://github.com/deepmodeling/DMFF.git /home/changjh/MD_projects/DMFF; \
    git -C /home/changjh/MD_projects/DMFF checkout "$DMFF_REV"; \
    cp -a /tmp/pku-md-runtime/dmff-overlay/dmff/. /home/changjh/MD_projects/DMFF/dmff/; \
    mkdir -p \
        /home/changjh/MD_projects/PKUGraduateThesis/data \
        /home/changjh/MD_projects/PhyNEO-Electrolyte/examples/md_simulation \
        /home/changjh/MD_projects/DeePDih/artifacts/exp_systems \
        /home/changjh/MD_projects/plugin_snapshots/dmff_match_20261008/cuda_mode3 \
        /home/changjh/MD_projects/ADMPPmeOpenMMPlugin \
        /home/changjh/MD_projects/ADMPPmeOpenMMPlugin-native-slater/build_native \
        /home/changjh/miniconda3/envs/md/lib/plugins \
        /home/changjh/miniconda3/envs/md/lib/python3.11/site-packages; \
    cp -a /tmp/pku-md-runtime/bridge /home/changjh/MD_projects/PKUGraduateThesis/bridge; \
    cp -a /tmp/pku-md-runtime/data/production_ready_experiments.csv /home/changjh/MD_projects/PKUGraduateThesis/data/; \
    cp -a /tmp/pku-md-runtime/forcefield/phyneo_hmtff_admp.xml \
          /tmp/pku-md-runtime/forcefield/phyneo_ecl.xml \
          /tmp/pku-md-runtime/forcefield/init.pdb \
          /home/changjh/MD_projects/PhyNEO-Electrolyte/examples/md_simulation/; \
    cp -a /tmp/pku-md-runtime/exp_systems/. /home/changjh/MD_projects/DeePDih/artifacts/exp_systems/; \
    cp -a /tmp/pku-md-runtime/plugins/libADMPPmePlugin.so /home/changjh/miniconda3/envs/md/lib/libADMPPmePlugin.so; \
    cp -a /tmp/pku-md-runtime/plugins/libADMPPmePlugin.so \
          /home/changjh/MD_projects/ADMPPmeOpenMMPlugin-native-slater/build_native/libADMPPmePlugin.so; \
    cp -a /tmp/pku-md-runtime/plugins/libADMPPmePluginCUDA.so \
          /home/changjh/miniconda3/envs/md/lib/plugins/libADMPPmePluginCUDA.so; \
    cp -a /tmp/pku-md-runtime/plugins/libADMPPmePluginCUDA.so \
          /home/changjh/MD_projects/plugin_snapshots/dmff_match_20261008/cuda_mode3/libADMPPmePluginCUDA.so; \
    cp -a /tmp/pku-md-runtime/python/mpidplugin.py \
          /tmp/pku-md-runtime/python/_mpidplugin.cpython-311-x86_64-linux-gnu.so \
          /home/changjh/miniconda3/envs/md/lib/python3.11/site-packages/; \
    cp -a /tmp/pku-md-runtime/python/mpidplugin.py /home/changjh/MD_projects/ADMPPmeOpenMMPlugin/mpidplugin.py; \
    printf '%s\n' \
      '15a5265f1afe596fd3ba5f8e9661fbf7e2bf9c64af1b36e270c7e4b898881bef  /home/changjh/MD_projects/DMFF/dmff/admp/pme.py' \
      '19d316563e444071bdd97ed97ae485bd470db529fa4e3e17684ba3a95d808f01  /home/changjh/miniconda3/envs/md/lib/libADMPPmePlugin.so' \
      '6e43000c41a354928f210cd4a90ec08dcddf9340657268a5fb368d1587216491  /home/changjh/miniconda3/envs/md/lib/plugins/libADMPPmePluginCUDA.so' \
      | sha256sum -c -; \
    rm -rf /tmp/pku-md-runtime

RUN bash -lc "set -euo pipefail; \
    source /home/changjh/miniconda3/etc/profile.d/conda.sh; conda activate md; \
    pip install --no-cache-dir setuptools setuptools_scm wheel \
        freud-analysis networkx optax jaxopt pymbar tqdm mdtraj parmed pandas \
        jax==0.4.35 jaxlib==0.4.35 \"jax-cuda12-plugin[with_cuda]==0.4.35\" jax-cuda12-pjrt==0.4.35; \
    pip install --no-cache-dir --no-build-isolation --no-deps -e /home/changjh/MD_projects/DMFF; \
    init=\"\$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvcc/__init__.py\"; \
    mkdir -p \"\$(dirname \"\$init\")\"; : > \"\$init\"; \
    python -c 'import dmff, jax, jaxlib, importlib.metadata as m; print(dmff.__file__); print(jax.__version__, jaxlib.__version__, m.version(\"jax-cuda12-plugin\"))'"

RUN conda run -n md python -c "import dmff, mpidplugin, openmm, numpy; print(openmm.version.version, numpy.__version__, dmff.__file__)"

WORKDIR /home/changjh/MD_projects/PKUGraduateThesis
CMD ["bash"]
