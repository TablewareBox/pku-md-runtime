# pku-md-runtime

公开输入，供 PKU 电解液生产模拟镜像在别处构建。`Dockerfile` 不使用构建上下文：它把本仓库的 `RUNTIME_REV` 克隆进镜像，再放到生产脚本使用的绝对路径上。玻尔把 `Dockerfile` 的内容贴进「基于 Dockerfile」即可。

```bash
docker build --platform linux/amd64 -t pku-md:20261010 https://github.com/TablewareBox/pku-md-runtime.git
```

当前 `RUNTIME_REV` 是 `9f927eb`。插件、力场、盒子或 `bridge/` 有改动时要一起改这个参数。镜像里的脚本要包含这次的路径改动之后，玻尔任务才能把结果写回工作目录。

生产入口是 `bridge/run_merged_production.sh`。它和下面这些 Python 文件是服务器上正在用的副本，导入链到此为止：`production_merged_sr.py`、`merged_aligned_sr.py`、`production_reporters.py`、`stability_hypotheses.py`、`exchange_hardcore.py`、`dmff_dispersion.py`、`dmff_qqtt.py`、`add_slater_custom.py`。三条队列脚本 `submit_ready_{nvt,2fs,4fs}.sh` 也在 `bridge/`。

这些脚本里的路径仍是服务器上的绝对路径。镜像里要把本仓库的文件放到同样的位置：

| 仓库路径 | 镜像内路径 |
|---|---|
| `bridge/` | `/home/changjh/MD_projects/PKUGraduateThesis/bridge/` |
| `forcefield/phyneo_hmtff_admp.xml`、`phyneo_ecl.xml`、`init.pdb` | `/home/changjh/MD_projects/PhyNEO-Electrolyte/examples/md_simulation/` |
| `exp_systems/` | `/home/changjh/MD_projects/DeePDih/artifacts/exp_systems/` |
| `plugins/libADMPPmePluginCUDA.so` | `/home/changjh/MD_projects/plugin_snapshots/dmff_match_20261008/cuda_mode3/libADMPPmePluginCUDA.so` |
| `plugins/libADMPPmePlugin.so` | `$CONDA_PREFIX/lib/libADMPPmePlugin.so`，以及 `ADMPPmeOpenMMPlugin-native-slater/build_native/` 下的同名文件 |
| `python/mpidplugin.py` 与 `_mpidplugin*.so` | `site-packages/`，`mpidplugin.py` 再放一份到 `/home/changjh/MD_projects/ADMPPmeOpenMMPlugin/` |
| `data/production_ready_experiments.csv` | `/home/changjh/MD_projects/PKUGraduateThesis/data/production_ready_experiments.csv` |

## 内容

| 路径 | 用途 |
|---|---|
| `plugins/libADMPPmePlugin.so` | ADMP 核心库，SHA256 `19d31656…` |
| `plugins/libADMPPmePluginCUDA.so` | 生产 CUDA 插件，SHA256 `6e4300…` |
| `python/mpidplugin.py` 与 `_mpidplugin*.so` | OpenMM Python 绑定 |
| `forcefield/phyneo_hmtff_admp.xml` | 服务器上改过 Thole 与 mScale 的力场。上游没有这个文件 |
| `forcefield/phyneo_ecl.xml`、`forcefield/init.pdb` | Slater / QqTt / 色散参数，以及默认 PDB。与 `junminchen/phyneo-electrolyte` 中的文件哈希相同，这里放的是服务器副本 |
| `exp_systems/` | 72 个实验盒子 PDB 及对应 JSON |
| `data/production_ready_experiments.csv` | 生产实验表 |
| `dmff-overlay/` | 训练用。生产模拟不 `import dmff`。盖在 DMFF `809f656` 上的 4 个文件 |

`forcefield/LICENSE` 是 PhyNEO 的 MIT 许可证。不要换成快照里另外两份 CUDA 二进制（`f236c130`、`b8e6c6ea`）。OpenMM 用 conda-forge 的 8.4，不要用 8.6.1。

## DMFF

公开分支 `wangxy/v1.0.0-devel` 仍有 `from jax.config import config`，在 JAX 0.4.35 上无法导入。这里的 4 个文件来自 `deepmodeling/DMFF` 的 `809f656`，并包含本地修改：`settings.py` 改为 `from jax import config`，`spatial.py` 对齐 OpenMM 的坐标轴，`pme.py` 与 `generators/admp.py` 增加 `implicit_pol`。DMFF 本体是 LGPL-3.0，许可证在 `dmff-overlay/LICENSE`。

```bash
git clone https://github.com/deepmodeling/DMFF.git
git -C DMFF checkout 809f656153018498e3260e07162197139ba103d6
cp -a dmff-overlay/dmff/. DMFF/dmff/
```

覆盖之后 `dmff/admp/pme.py` 的 SHA256 应为 `15a5265f1afe596fd3ba5f8e9661fbf7e2bf9c64af1b36e270c7e4b898881bef`。

## 玻尔 Batch Job

`bohr batchjob submit` 把 `--input` 打成 zip，在容器里解压后作为工作目录，再执行 `--command`。命令用 `bash run.sh`。结果只回收 `--out-file` 点名的路径，写到镜像里的 `runs/` 不会出现在下载包里。

`bohr/input/run.sh` 把日志、检查点和轨迹写到工作目录的 `results/`，结束时打成 `results.tar`。`job.env` 决定跑法：默认 `MODE=smoke` 是 200 步（0.1 ps）。队列把 `MODE` 改成 `nvt`、`2fs` 或 `4fs`，并在同一目录放 `list.tsv`。

盒子默认用镜像里的 `exp_systems`。输入目录里如果有 `exp_systems/`，或者 `job.env` 设置了 `EXP_SYSTEMS`，pdb 列和 `--pdb` 改从那里找。绝对路径保持不变；`artifacts/exp_systems/名称.pdb` 会去掉这个前缀。

```bash
bohr batchjob machine list --choose-type gpu
IMAGE=<镜像中心的完整地址> MACHINE_TYPE=<上面列出的机型> bash bohr/submit.sh
IMAGE=<镜像地址> MACHINE_TYPE=<机型> SUBMIT=1 bash bohr/submit.sh
bohr batchjob download <job_id> --dest ./result
```

`submit.sh` 默认 `--dry-run`，不创建任务。`download` 的目标目录必须是还不存在的新目录。解压后读取 `results.tar`。机型名称以本机 `bohr batchjob submit --help` 为准。

## 校验

```bash
shasum -a 256 -c SHA256SUMS
```
