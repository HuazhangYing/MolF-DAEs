# MolF-DAEs Benchmark Prepare

这个目录只负责准备后续 benchmark 会消费的中间产物，不做下游质量评估、聚类、绘图或 summary benchmark。

当前阶段重点改成：优先复用已经生成好的 AE latent 表征，再做后续 3D 降维比较。

其中 `pubchemfp` 已按 `/data/yinghuazhang/MolF-DAEs/code/PCA.ipynb` 的方式接到 `/data/yinghuazhang/MolF-DAEs/result/comparison/PCA/1-dae`，直接复用分块保存的 `128D` latent；不会重新生成 128 维输入。

当前阶段已实现：

- 复用预计算 latent 作为下游投影输入
- `pacmap3d` 参数网格投影
- `phate3d` 参数网格投影
- `gtm3d` 保留目录骨架
- `full / 100k / 300k / 1m` 子集管理
- `mol2vec` 300D embedding 接入 benchmark 体系，可继续复用 `pacmap3d/phate3d` 投影
- `coords_3d.npy`、`ids.npy`、`params.json`、`reconstruction_summary.json` 输出

当前阶段未实现：

- trustworthiness / continuity
- clustering quality
- plotting summary
- GTM 真正训练或投影
- 重新训练或重新导出 AE latent

## 目录结构

输出按“方法 -> 特征组 -> 子集 -> 参数哈希”组织：

```text
3.benchmark/
  benchmark_config.yaml
  run_benchmark_prepare.py
  utils/
  outputs/
    _subsets/
    _reconstruction/
    pacmap3d/
    phate3d/
    gtm3d/
```

每个实际运行目录包含：

- `coords_3d.npy`: 当前参数配置下生成的 3D 坐标
- `ids.npy`: 当前 run 使用的样本 id
- `params.json`: 方法、子集、参数、来源 latent、时间戳等元数据
- `reconstruction_summary.json`: 当前阶段主要保存 latent 来源摘要；不强制重新跑 decoder reconstruction
- `run_log.txt`: 运行日志

## 运行方式

```bash
python run_benchmark_prepare.py --feature-group pubchemfp --method pacmap3d --subset full
python run_benchmark_prepare.py --feature-group mol2vec --method phate3d --subset 100k
python run_benchmark_prepare.py --feature-group pubchemfp --method phate3d --subset 100k
python run_benchmark_prepare.py --feature-group all --method pacmap3d --subset 300k
```

强制覆盖已有结果：

```bash
python run_benchmark_prepare.py --feature-group pubchemfp --method pacmap3d --subset full --force
```

## 关于 latent 来源

- `pubchemfp`: 已切到 `PCA.ipynb` 使用的 `/data/yinghuazhang/MolF-DAEs/result/comparison/1-pubchem` 分块 128D latent
- `mol2vec`: 由 `run_mol2vec_3d.py` 生成 `300D` embedding，并通过 `valid_row_ids.npy` 映射回原始 CSV 行号
- `maccsfp` / `pharmacopfp`: 当前工作区里还没找到对应的 comparison 版预计算 128D latent，所以配置里暂时保留旧 latent 来源；如果你后面给出对应路径，可以直接在 `benchmark_config.yaml` 里切换

## 关于 reconstruction_summary

当前阶段的目标是比较降维效果，不是重新生成 AE 输入或重新评估 decoder 重建。因此：

- `reconstruction_rate`、`mean_mse`、`mean_mae` 目前默认留空
- `reconstruction_summary.json` 主要记录 latent 来源、latent 维度、参考 notebook 和可用的训练 summary 元数据
