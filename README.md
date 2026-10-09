# pku-md-runtime

公开输入，供 PKU 电解液生产模拟镜像在别处构建。玻尔的 Dockerfile 构建不能 `COPY` 本地文件，所以构建机用 `git clone` 取这一份。

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

## 校验

```bash
shasum -a 256 -c SHA256SUMS
```
