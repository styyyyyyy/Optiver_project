# Optiver 已实现波动率研究项目

[英文主页](README.md) · [实验结果](docs/experiment_results.md) · [复现说明](docs/reproducing.md) · [代码架构](docs/architecture.md)

这是整理后的 Optiver 短期已实现波动率预测项目。新版将原来集中在根目录的
实验脚本拆分为可复用模块、命令行入口、测试、实验协议和结果表。

## 最终开发集结果

| 模型 | Pooled OOF RMSPE | Conditional 95% CI |
|---|---:|---:|
| Strict nested LightGBM + MLP | **0.217123** | [0.212922, 0.222404] |
| LightGBM | 0.218334 | [0.213820, 0.224134] |
| MLP | 0.227346 | [0.223732, 0.232059] |

Strict nested ensemble 相对 LightGBM 改善约 `0.555%`，paired difference 为
`-0.001211`，conditional paired 95% CI 为
`[-0.001941, -0.000558]`。

这个结果仍是同一 development dataset 上的 grouped-CV estimate，不是独立测试集
成绩。模型、特征与 ensemble 路径均受到该数据集的研究反馈。

## 从哪里开始

- 想看结论：[`docs/experiment_results.md`](docs/experiment_results.md)
- 想跑实验：[`docs/reproducing.md`](docs/reproducing.md)
- 想看每个文件的职责：[`docs/architecture.md`](docs/architecture.md)
- 想准备数据：[`data/README.md`](data/README.md)
- 想看可直接检查的结果表：[`results/`](results/)
- 原始脚本与旧图：[`legacy/`](legacy/)，仅作历史参考

## 快速安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mlp,plots]'
python -m unittest discover -s tests
```

数据集受 Kaggle 竞赛规则约束，因此没有重新发布在公开 GitHub 仓库。获得访问权限后，
请按 [`data/README.md`](data/README.md) 放置数据和特征表。
