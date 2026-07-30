# DyPH-RAG 本地研究框架

DyPH-RAG 是在 KARE 旁路新增的疾病条件患者–疾病–时间分类框架。现有 KARE、微调和 baseline 文件未被删除，原入口保持不变。本实现提供可审计、可恢复、离线可运行的研究参考实现；它不声称在缺少原始预处理、划分和外部知识版本时精确复现 GraphCare 或 KARE 的论文结果。

## 1. 安装与 CPU 验收

    cd /home/shanyan/cqdmk/xinghuo/KARE-main
    KARE_INSTALL_DYPHRAG=1 bash scripts/setup_env.sh
    conda activate kare
    ruff check src tests
    pytest -q
    bash scripts/smoke_dyphrag.sh

冒烟测试只使用合成患者，不访问网络，不读取 MIMIC，也不会调用 LLM API。

## 2. 统一运行入口

    python -m src.train \
      experiment=full_dyphrag \
      dataset=synthetic \
      task=disease_pair_binary \
      seed=42 \
      output_dir=results/full_dyphrag_seed42

参数采用 Hydra 风格的 group=value 和 nested.key=value 语法，由离线 YAML 组合器解析。每个成功运行都会产生：

    results/<run_id>/
    ├── resolved_config.yaml
    ├── run_metadata.json
    ├── metrics.jsonl
    ├── summary.json
    ├── checkpoints/{last,best,calibrated}.ckpt
    ├── indices/cohort_manifest.json
    ├── retrieval_traces/
    ├── evidence_exports/
    ├── tensorboard/
    └── mlflow_run_id.txt

MLflow 使用本地 file store；W&B 默认关闭，只有显式配置时才启用。trace 和 evidence export 只写运行内匿名引用、证据哈希、来源、极性、分数和 provenance 完整度，不写患者 ID、临床文本或 prompt。

## 3. 架构边界

- self：只检索当前患者 timestamp <= cutoff 的 visit/event/episode/trend/state 单元。
- peer：构造函数只接受训练集患者；先做患者级候选，再做事件级重排；默认屏蔽直接目标疾病文本。
- medical：支持 vector、graph 和 hybrid；JSONL 必须提供来源、日期、版本、人群、证据等级、相对目标疾病、极性、provenance、许可证以及可选的 supersedes/contradicts。
- prior：无检索患者–疾病先验，只参与 mixture，不进入 citation/evidence export。
- Router：支持 learned soft、sparse Gumbel、uniform 和 heuristic，输出 3+1 权重、总检索门、逐源门、每源预算、证据充分性和停止概率，并可形成真实 prior-only 路径。查询显式为 `[h_patient; h_disease; interaction; delta_t; context]`。
- Hypergraph：包含 patient、disease、diagnosis、medication、procedure、lab/vital、time、visit、context、external evidence 节点及七类超边。数值 value 节点保留原值/归一化值/单位/标准单位/参考区间/异常方向/测量条件/时间/斜率/基线偏移/缺失与观测掩码。原始 EHR 超边不可删除；动态算子只能停用 derived/retrieved 超边。
- Evidence：support、refute、differential、neutral 独立聚合，并提供可靠性/适用性、矛盾率、充分性与 abstention。

默认图编码器是轻量 typed heterogeneous Hypergraph Transformer。hgnn、hgat、clique、star 作为配置基线。数值编码支持 continuous MLP、Fourier、spline/basis、monotonic 和 bucketing 消融。

## 4. MIMIC-III/MIMIC-IV

真实数据必须先在有授权的环境中转换为 cutoff-safe JSONL。路径只通过环境变量传入：

当前工作区已验证的 MIMIC-III 转换命令：

    python scripts/prepare_dyphrag_mimic.py \
      --dataset mimiciii \
      --root data/mimiciii \
      --output data/processed/dyphrag_mimiciii_full.jsonl \
      --salt-file data/.dyphrag_id_salt \
      --split-strategy temporal \
      --seed 42 \
      --top-k-diseases 50 \
      --min-train-positive 25

转换器在落盘前用本地 keyed HMAC 替换患者、住院和事件 ID；salt 文件权限为 0600，且 data 目录不进入版本控制。任务是在 index admission 入院时预测该住院最终记录的 ICD category。诊断表只提供标签，不进入输入或检索索引。阴性仅表示该住院未编码目标类别，不表示临床上无病。

该命令已生成 7,517 名患者、47,870 个疾病对样本；训练/验证/测试患者数为 6,013/752/752，样本数为 38,416/4,666/4,788。

本地真实数据 CPU miniature 验收命令：

    python -m src.train \
      experiment=mimic_core_no_medical \
      dataset=mimiciii \
      seed=42 \
      data.prepared_jsonl=data/processed/dyphrag_mimiciii_full.jsonl \
      data.max_train_examples=64 \
      data.max_validation_examples=32 \
      data.max_test_examples=32 \
      training.epochs=1 \
      runtime.device=cpu \
      output_dir=results/mimiciii_full_mini_final_seed42

该目录未包含 LABEVENTS/CHARTEVENTS，也没有提供带许可证和版本 provenance 的外部医学 corpus，因此此配置关闭 numeric-state 和 medical expert，名称明确标为 mimic_core_no_medical。

miniature 运行只用于验证真实数据的数据流、泄漏边界、恢复和指标计算；其 AUROC/AUPRC 不应作为论文结果或模型比较结论。

    export DYPHRAG_MIMIC3_JSONL=/secure/path/mimiciii_disease_pairs.jsonl
    # 或
    export DYPHRAG_MIMIC4_JSONL=/secure/path/mimiciv_disease_pairs.jsonl
    export DYPHRAG_MEDICAL_CORPUS=/secure/path/versioned_medical_evidence.jsonl

    python -m src.train \
      experiment=full_dyphrag \
      dataset=mimiciv \
      seed=42 \
      output_dir=results/mimiciv_full_seed42

每条患者样本必须包含 patient_id、target_disease、cutoff_time、label、label_rule、split、context 和 events。label_rule 不可为空。加载器在构建任何索引前检查患者划分互斥、事件时间、出院诊断、post-cutoff outcome 和目标标签字段。label=2 表示 unknown，并训练 abstention；二分类指标只使用已知标签并报告 known-label coverage。

外部医学知识 JSONL 的必填字段：

    source_id, text, concepts, publication_date, source_type, version,
    population, evidence_grade, polarity, provenance_span, license,
    target_diseases

可选字段为 reliability、supersedes、contradicts 和 graph_neighbors。真实 MIMIC 运行启用 medical retriever 时若未提供该 corpus，会明确失败，不会用合成知识冒充真实知识。
字段格式示例见 `data_examples/dyphrag_medical_corpus.schema.jsonl`；该记录明确不是医学证据，不能用于真实实验。

## 5. 中断与精确恢复

runtime.resume=true 时自动读取 checkpoints/last.ckpt。检查点包含模型、优化器、epoch 内 position、Python/NumPy/PyTorch/CUDA RNG 状态。

    python -m src.train \
      experiment=full_dyphrag dataset=synthetic seed=43 \
      runtime.interrupt_after_steps=3 \
      output_dir=results/resume_test

    python -m src.train \
      experiment=full_dyphrag dataset=synthetic seed=43 \
      runtime.interrupt_after_steps=0 \
      output_dir=results/resume_test

第二条命令从下一条训练样本继续；同一 MLflow run 追加 metrics。

## 6. 实验矩阵

全部 baseline、DyPH-RAG 和消融默认运行三个 seed：

    bash scripts/launch_all.sh

所有路径、API 环境变量、GPU、seed 和并发参数集中在唯一配置文件
`scripts/dyphrag_env.sh`；编辑后也可直接执行：

    bash scripts/dyphrag_env.sh

常用环境变量：

- EXPERIMENT_GROUPS=principal|all|smoke|配置名
- SEEDS=42,43,44
- MAX_PARALLEL=1
- RETRY_COUNT=1
- FAIL_FAST=0|1
- DYPHRAG_DATASET=synthetic|mimiciii|mimiciv
- DYPHRAG_MATRIX_DIR=results/matrix
- DYPHRAG_MATRIX_OVERRIDES 用于批量追加配置覆盖。

启动器保存 pending/running/succeeded/failed/interrupted 状态，跳过已成功任务，保留 stdout/stderr，轮询 GPU 到 CSV，并生成 aggregate CSV/JSON/Markdown、paired seed-wise comparisons 和不含患者证据的 reproducibility bundle。
同一 GPU 由进程锁保证同一时刻只运行一个任务；Rich 表实时显示 step、split、最近指标和耗时。另生成 `aggregate_stats.csv/json`，报告每个实验的多 seed 均值和标准差。

服务器上传包：

    bash scripts/package_server.sh

详见 `SERVER_DEPLOYMENT.md`。归档明确排除患者数据、结果、密钥、缓存和 Git 历史。
三项创新的代码级精细定义见 `DYPHRAG_INNOVATIONS.md`。

## 7. 配置目录

- configs/experiment/full_dyphrag.yaml
- 统一 baseline registry 包含 EHR GRU/Transformer、静态异构图、visit hypergraph、GraphCare/KARE adaptation、static vector RAG、disease retrieval、dynamic no-polarity 和 full DyPH-RAG。
- 检索、超图和 evidence/RAG 的所有指定消融均位于 `configs/experiment/ablation_*.yaml`；同时保留早期配置名作为兼容别名。
- 主结果至少运行 3 个 seed；最终报告可配置 5 个 seed。

## 8. 当前外部阻塞

以下内容不会伪造结果：

1. 已完成自定义 admission-time ICD coding 任务的 MIMIC-III demo；它不是论文作者的原始标签规则或 split，因此不能视为原论文指标复现。
2. 未提供经过许可且带版本/日期/provenance 的 guideline/paper/ontology corpus，因此真实 medical retrieval 尚未运行。
3. 没有临床专家标注的 evidence relevance/polarity qrels；当前 retrieval/evidence 指标主要验证计算和数据流。
4. GraphCare/KARE 配置是统一数据契约上的 adaptation，不是论文代码和预处理的精确复制。
5. 当前提供的核心表不包含 lab/vital 数值事件，数值状态超边仅在合成测试中完成工程验收。
