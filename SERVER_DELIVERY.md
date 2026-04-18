# CARIS PlantSeg Server Delivery

本仓库已按单张 RTX 4090 Linux 服务器交付面收敛为以下固定流程：

## 1. 数据放置

- 数据目录固定为仓库根目录同级的 `../plantseg`
- 目录中必须包含：
  - `main.json`
  - `images/`
  - `ann/`

## 2. 服务器环境

- 推荐使用仓库内的 `envs/server_4090.yml`
- `scripts/setup_server_env.sh` 负责：
  - 创建或更新 conda 环境
  - 安装 `mmcv-full>=1.3.17,<1.5.0`
  - 安装服务器补充依赖

## 3. 官方预训练资产

- BERT 使用官方 `bert-base-uncased`
- Swin 使用官方 `swin_base_patch4_window7_224_22k.pth`
- 默认缓存目录为仓库内 `pretrained_assets/`
- 训练和测试入口会自动下载缺失资产

## 4. 训练

- 入口：`scripts/train_plantseg.sh`
- 固定设置：
  - 数据集：`plantseg`
  - 文本：`caption[3]`
  - epoch：50
  - 单卡 batch size：4
  - best checkpoint 判据：验证集 `mIoU`

## 5. 测试

- 入口：`scripts/test_plantseg.sh`
- 测试 split 固定为 `test`
- 输出内容固定落在 `output_dir`
  - 聚合指标：`test_metrics.txt`
  - 预测 mask：`test_masks/ann/*.png`

## 6. 最终指标

- `IoU`：前景 IoU
- `Dice`：前景 Dice
- `Recall`：前景 Recall
- `mIoU`：前景与背景 IoU 平均
- `mACC`：前景与背景 Recall 平均

## 7. 当前不纳入主流程的字段

- `false_healthy_ann` 已保持可见，但当前训练和测试都不接入
- `is_refined` 当前仅保留为元信息，不改变监督来源
