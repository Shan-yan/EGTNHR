# KARE 本地复现指南

这份指南针对当前机器（Linux、RTX 5070 Laptop 8 GB）整理。原论文代码默认使用作者服务器的 `/shared/eng/...` 路径和 6–7 张 GPU；本地适配版把路径改为参数或环境变量，并提供离线检查、MIMIC 预处理、单卡训练和推理入口。

## 1. 复现边界

目前可以在本地直接验证：代码语法、发布的 45 MB 原始知识图谱、10,000 条社区摘要、两组各 2,000 个检索示例、评估逻辑，以及 0.5B 模型的单卡训练链路。

要复现论文表格中的完整数值，还必须自行提供：

- 经 PhysioNet 授权下载的 MIMIC-III 1.4 或 MIMIC-IV；仓库不能分发这些受限数据。
- OpenAI 或 Amazon Bedrock 凭据，用于知识摘要、嵌入和推理链生成；这些步骤会产生费用。
- 论文规模的 7B 全参训练算力。当前 8 GB 单卡只适合小模型功能验证或低比特 LoRA，不能等价替代论文训练。
- 官方仓库没有发布全部中间文件，例如完整社区索引、实体聚类映射、患者上下文、训练集和模型检查点。仓库中的 `data_examples` 是示例，不是完整论文数据。

此外，上游 issue #17 指出死亡/再入院标签构造可能存在就诊排序及时间差定义争议。为了与发布代码保持一致，本地适配没有静默改变标签定义；做严格科学复现时应同时报告“原始定义”和“修正定义”的敏感性分析。

## 2. 环境

创建核心环境（默认安装体积较小的 CPU PyTorch，足以完成预处理、检索和评估）：

```bash
chmod +x scripts/setup_env.sh finetune/sft_mort.sh finetune/sft_readmit.sh
bash scripts/setup_env.sh
conda activate kare
```

已验证环境的 127 个直接与传递依赖版本记录在 `requirements-lock.txt`；常规安装仍建议使用脚本，因为 CPU/CUDA PyTorch 需要不同的官方 wheel 索引。

需要在此环境中使用 RTX 5070 训练/推理时，再显式安装 CUDA 12.8 wheel（下载数 GB，网络较慢时耗时很长）：

```bash
KARE_TORCH_BACKEND=cu128 bash scripts/setup_env.sh
```

需要微调依赖时：

```bash
KARE_INSTALL_FINETUNE=1 KARE_ENV_NAME=kare bash scripts/setup_env.sh
```

如果 `kare` 环境已经创建，不要再次执行 `conda env create`，直接运行：

```bash
python -m pip install -r requirements-finetune.txt
```

GPU 模式选择 CUDA 12.8 是因为 RTX 5070 属于 Blackwell 架构。不要用项目发布时期常见的旧 CUDA 11.x PyTorch wheel。

## 3. 离线验证

不需要 MIMIC、API 或 GPU：

```bash
python scripts/preflight.py
python scripts/smoke_test.py
python -m compileall -q .
```

预期离线 smoke test 输出四行 `PASS`。`preflight.py` 中 EHR、LLM 或训练阶段显示 `BLOCKED` 并不代表代码损坏，而是对应外部资源尚未配置。

## 4. 配置数据和凭据

```bash
cp .env.example .env
# 编辑 .env，至少填写 MIMIC3_ROOT 或 MIMIC4_ROOT
set -a
source .env
set +a
```

API 密钥只从环境变量/AWS 标准凭据链读取，不再写入源码，也不需要复制或重命名 `apis_example`。

## 5. MIMIC-IV 预处理示例

死亡预测：

```bash
python -m ehr_prepare.ehr_data_prepare \
  --dataset mimic4 \
  --task mortality \
  --mimic-root "$MIMIC4_ROOT" \
  --output-dir "$KARE_DATA_DIR/ehr_data" \
  --max-samples 10000 \
  --seed 42

python -m ehr_prepare.sample_prepare \
  --dataset mimic4 \
  --task mortality \
  --data-dir "$KARE_DATA_DIR/ehr_data" \
  --seed 528
```

再入院任务把两个命令中的 `mortality` 改为 `readmission`。首次运行 PyHealth 会建立缓存并可能下载编码映射，因此耗时较长。

生成不调用 LLM 的基础患者文本：

```bash
python -m patient_context.base_context \
  --input "$KARE_DATA_DIR/ehr_data/pateint_mimic4_mortality.json" \
  --output "$KARE_DATA_DIR/patient_context/base_context/patient_contexts_mimic4_mortality.json"
```

配置 `OPENAI_API_KEY` 后生成患者嵌入并检索相似患者：

```bash
python -m patient_context.get_emb \
  --input "$KARE_DATA_DIR/patient_context/base_context/patient_contexts_mimic4_mortality.json" \
  --output "$KARE_DATA_DIR/patient_context/base_context/patient_embeddings_mimic4_mortality.pkl"

python -m patient_context.sim_patient_ret_faiss \
  --contexts "$KARE_DATA_DIR/patient_context/base_context/patient_contexts_mimic4_mortality.json" \
  --patient-data "$KARE_DATA_DIR/ehr_data/pateint_mimic4_mortality.json" \
  --embeddings "$KARE_DATA_DIR/patient_context/base_context/patient_embeddings_mimic4_mortality.pkl" \
  --output "$KARE_DATA_DIR/patient_context/similar_patient/patient_to_top_1_patient_contexts_mimic4_mortality.json"
```

文件名中的 `pateint` 是上游代码遗留拼写，为兼容其他脚本而保留。

## 6. 单卡训练功能验证

安装 `requirements-finetune.txt` 后，下面命令默认使用仓库内 8 条合成训练样本、2 条测试样本和 `Qwen/Qwen2-0.5B-Instruct`，只用于验证训练链路：

```bash
bash finetune/sft_mort.sh
```

训练真实生成数据时覆盖路径：

```bash
export KARE_TRAIN_FILE="$KARE_DATA_DIR/llm_finetune_data/mimic4_mortality_train.jsonl"
export KARE_TEST_FILE="$KARE_DATA_DIR/llm_finetune_data/mimic4_mortality_test.jsonl"
bash finetune/sft_mort.sh
```

原论文高算力配置仍在 `finetune/recipes/config_full_*.yaml`，但其中的显存需求不适合本机。

## 7. 本地推理与评估

```bash
python -m prediction.llm_inference.generate \
  --model data/outputs/local-smoke \
  --test-file data_examples/finetune_smoke/test.jsonl \
  --output data/outputs/local-smoke-predictions.json \
  --max-new-tokens 128

python -m prediction.eval data/outputs/local-smoke-predictions.json \
  --output data/outputs/local-smoke-metrics.json
```

新的评估器不会像原脚本那样在解析失败时悄悄翻转真实标签；默认遇到无法解析的预测会报错。需要兼容不完整旧结果时可显式使用 `--missing-policy skip` 或 `--missing-policy zero`。

## 8. 严格复现实验建议

记录并固定以下内容：MIMIC 版本、PhysioNet 文件校验和、PyHealth 版本与缓存、数据划分 seed、LLM 的完整模型 ID、提示词、API 日期、所有生成中间件的哈希、训练模型 revision、CUDA/PyTorch 版本及评估解析失败数量。由于 LLM API 后端会更新，仅固定温度和 seed 不足以保证逐字符一致。
