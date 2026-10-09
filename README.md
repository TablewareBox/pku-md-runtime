# pku-md-runtime

公开输入，供 PKU 电解液生产模拟镜像在别处构建。玻尔的 Dockerfile 构建不能 `COPY` 本地文件，所以构建机用 `git clone` 取这一份。

镜像里还需要本仓库没有的两样东西：论文仓库里的 `bridge/`，以及 `junminchen/phyneo-electrolyte` 里已经公开的 `phyneo_ecl.xml` 和 `init.pdb`。这两份和服务器上的文件哈希相同，不必再放一份。

## 内容

| 路径 | 用途 |
|---|---|
| `plugins/libADMPPmePlugin.so` | ADMP 核心库，SHA256 `19d31656…` |
| `plugins/libADMPPmePluginCUDA.so` | 生产 CUDA 插件，SHA256 `6e4300…` |
| `python/mpidplugin.py` 与 `_mpidplugin*.so` | OpenMM Python 绑定 |
| `forcefield/phyneo_hmtff_admp.xml` | 服务器上改过 Thole 与 mScale 的力场。上游 `junminchen/phyneo-electrolyte` 没有这个文件，许可证是 MIT，见 `forcefield/LICENSE` |
| `exp_systems/` | 72 个实验盒子 PDB 及对应 JSON |
| `dmff-overlay/` | 盖在 DMFF `809f656` 上的 4 个文件 |

不要换成快照里另外两份 CUDA 二进制（`f236c130`、`b8e6c6ea`）。OpenMM 用 conda-forge 的 8.4，不要用 8.6.1。

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
