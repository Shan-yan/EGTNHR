# DyPH-RAG 服务器部署

本仓库保留原 KARE、GraphCare-like/PyHealth 基线路径，并将 DyPH-RAG 放在
`src/dyphrag/` 旁路中。服务器包不包含原始 EHR、历史结果、密钥、缓存或 Git
历史。

## 1. 本地生成上传包

```bash
bash scripts/package_server.sh
```

输出：

```text
dist/KARE-DyPH-RAG-server.tar.gz
dist/KARE-DyPH-RAG-server.tar.gz.sha256
```

## 2. 服务器安装

```bash
tar -xzf KARE-DyPH-RAG-server.tar.gz
cd KARE-DyPH-RAG-server
KARE_INSTALL_DYPHRAG=1 KARE_TORCH_BACKEND=cu128 bash scripts/setup_env.sh
conda activate kare
```

请按服务器 CUDA 版本修改 `KARE_TORCH_BACKEND`。如果不确定，先在服务器上运行
`nvidia-smi`，再按 PyTorch 官方支持的 wheel 选择。
需要 W&B 时增加 `KARE_INSTALL_WANDB=1`，并在运行配置中启用
`tracking.wandb=true`。

## 3. 唯一运行配置文件

编辑 `scripts/dyphrag_env.sh`，至少设置真实数据路径和外部医学语料路径：

```bash
export DYPHRAG_DATASET=mimiciv
export DYPHRAG_MIMIC4_JSONL=/secure/data/dyphrag_mimiciv.jsonl
export DYPHRAG_MEDICAL_CORPUS=/secure/data/versioned_medical_evidence.jsonl
export CUDA_VISIBLE_DEVICES=0,1
export MAX_PARALLEL=2
```

API 密钥也只在这个文件对应的环境变量中配置。核心训练器不会把患者文本发送给
在线 API；外部知识必须先保存为带日期、版本、来源、人群、证据等级和 provenance
的 JSONL。

直接运行全部基线、模型和消融：

```bash
bash scripts/dyphrag_env.sh
```

实时矩阵显示在终端；单任务 stdout/stderr、状态、GPU 监控与汇总位于
`results/matrix/`。中断后重复同一命令会跳过成功任务并恢复其余 checkpoint。

## 4. 单实验

```bash
python -m src.train \
  experiment=full_dyphrag \
  dataset=mimiciv \
  task=disease_pair_binary \
  seed=42 \
  output_dir=results/full_dyphrag_seed42
```

## 5. 上传前边界

打包脚本只纳入源码、实验 YAML、文档、测试、示例和原基线目录。以下内容必须在
服务器上通过安全通道单独提供：

- MIMIC 或其他患者数据；
- 预处理 JSONL 和 pseudonym salt；
- 外部医学知识库；
- API、W&B 或云服务密钥；
- 历史 checkpoint、MLflow、TensorBoard 和结果目录。
