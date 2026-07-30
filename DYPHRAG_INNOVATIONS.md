# DyPH-RAG 三项创新的精细实现定义

本文描述仓库中的可执行定义，而不是尚未完成的设想。所有预测样本统一为

\[
x=(p,d,t,y),\qquad y\in\{0,1,\mathrm{unknown}\}.
\]

模型输入、粗排索引、细排证据、超图节点及 peer 表示都必须满足可用时间不晚于
\(t\)。参数化先验不属于可引用证据。

## 创新一：疾病条件化的 3+1 动态检索

每一轮查询严格构造为

\[
q_{p,d,t}^{(r)}
=[h_p^{(r)};h_d;h_p^{(r)}\odot h_d;\Delta t;h_{\mathrm{context}}].
\]

Router 同时输出：

\[
(w_s,w_p,w_m,w_0),\ (g_s,g_p,g_m),\ g_R,\
(k_s,k_p,k_m),\ P_{\mathrm{sufficient}},\ P_{\mathrm{stop}}.
\]

- \(w_s,w_p,w_m,w_0\)：self、peer、medical、parametric prior 的混合权重；
- \(g_s,g_p,g_m\)：三个可追溯来源的独立门控；
- \(g_R\)：总检索门控，低于阈值时预算全部为零，形成真实 prior-only 路径；
- \(k_s,k_p,k_m\)：受 `max_total_k` 约束的整数预算；
- 充分性与停止概率在检索、图更新和不确定性重估后再次计算。

learned-soft、sparse-Gumbel、uniform 和 uncertainty heuristic 共用同一输出契约。
learned router 采用检索正向初始化，避免训练初期塌缩到 prior-only；训练损失监督
总检索门、来源门、充分性和停止决定，并加入检索成本正则。

Self memory 将截止时间前 EHR 组织为 event、visit、episode、numeric trend window
和 patient-state hyperedge。每条返回项带事件哈希、时间、来源粒度和概念集合。

Peer memory 只接受 `split=train` 的患者构建索引，并使用两阶段检索：

1. 对每名训练患者的每个历史 cutoff 物化时间切片快照；
2. 查询时，每名患者只允许使用不晚于查询 \(t\) 的最新快照做粗排；
3. 在入选快照内部重排事件或状态；
4. 显式排除当前患者，并再次审计 peer split、patient ID 和时间。

因此，未来事件既不能进入细排结果，也不能通过完整患者向量影响粗排名次。快照
cutoff、索引 hash 和去标识化 peer 表示进入审计元数据。

Medical memory 的 JSONL 文档必须提供来源、版本、发布日期、人群、证据等级、
provenance、许可证、概念和 `target_diseases`。证据极性相对于查询疾病重新解释：
支持另一疾病的文献属于 differential，而不是当前疾病的 support。发布日期晚于
\(t\) 的文档不可检索。

Parametric prior 由 \(h_p,h_d,h_p\odot h_d\) 产生，只参与预测和路由训练，不创建
EvidenceItem，不进入 citation、trace evidence 或 evidence export。

## 创新二：检索驱动的动态时序患者超图

图包含 patient、disease、diagnosis、medication、procedure、lab/vital、time、
visit、context、patient-state、external evidence 和内部 value 节点。原始 EHR
事实形成 `immutable_raw=true` 的 visit、numeric_state、treatment、
chronic_context 和 temporal_transition 超边；`deactivate()` 对其会直接报错。

每个 visit 生成显式 patient-state 节点。相邻 visit 的状态由 temporal-transition
连接；treatment 超边包含 patient、disease、pre-state、治疗实体、post-state 和
time，并保存前后状态引用及治疗时间。动态更新只可作用于 cohort/evidence 等派生
超边。

每个 value 节点保留原始值、归一化值、原/标准单位、参考范围、异常方向、测量
条件和时间、距上次测量时间、变化率、患者基线偏移及 missing/observed 掩码。
支持 continuous MLP、Fourier、spline/basis、monotonic、bucketing 和默认的
disease-conditioned 编码。疾病条件模式使用 \(h_d\) 对连续表示做 FiLM 变换。

每轮执行：

1. 编码当前 patient–disease 超图；
2. 路由并按预算检索；
3. 依据规范化 concept 集合将证据对齐到原 EHR 实体；
4. 创建或重加权 evidence/cohort 超边；
5. 只对低综合置信派生边执行停用；
6. 用 typed temporal hyperedge self-attention 和 patient cross-attention 重编码；
7. 重新估计预测熵和证据充分性；
8. 生成下一轮疾病条件查询，由 post-retrieval Router 决定停止。

证据边权由 relevance、source reliability 和 population applicability 共同决定，
并记录 alignment confidence。`reweight_only` 不会修改无关派生边。Cohort 节点
使用训练患者的 cutoff-safe 快照向量，不复制当前患者表示。

默认编码器为 heterogeneous temporal Hypergraph Transformer；HGNN、HGAT、
clique/star ordinary-graph expansion 和静态超图均使用同一输入契约。

## 创新三：支持—反证—鉴别证据一致性 RAG

每条证据相对于目标疾病判定为

\[
z_e\in\{\mathrm{support},\mathrm{refute},
\mathrm{differential},\mathrm{neutral}\}.
\]

极性分类器输入证据表示、疾病表示、二者交互和来源嵌入。真实 EHR 的启发式极性
不会被伪装成监督标签；只有合成受控标签或医学语料明确标签进入 polarity loss。
Differential evidence 可携带去标识化的鉴别疾病引用。

四类证据分别池化，最终预测显式使用 \(h^+,h^-,h^\Delta\)，而非拼接全部原始
证据。池化权重联合考虑 reranker、检索分数、来源可靠性和人群适用性。矛盾检测
同时使用显式 `contradicts` 关系及对齐概念上的 support/refute 冲突。

训练目标为

\[
\mathcal L =
\lambda_c\mathcal L_{\mathrm{cls}}+
\lambda_r\mathcal L_{\mathrm{retrieval}}+
\lambda_g\mathcal L_{\mathrm{routing}}+
\lambda_p\mathcal L_{\mathrm{polarity}}+
\lambda_m\mathcal L_{\mathrm{support/refute}}+
\lambda_e\mathcal L_{\mathrm{consistency}}+
\lambda_b\mathcal L_{\mathrm{Brier}}+
\lambda_a\mathcal L_{\mathrm{selective}}+
\lambda_k\mathcal L_{\mathrm{retrieval\ cost}}.
\]

温度参数不与分类器联合训练。选出最佳 checkpoint 后冻结模型，只在验证集拟合
temperature；abstention threshold 也只在验证集、满足最小 coverage 约束时选择，
随后冻结用于测试。最终输出 yes、no 或 insufficient_evidence，同时返回校准概率、
abstention probability 和可追溯证据。

## 实验执行与边界

`python -m src.train experiment=<name> ...` 是唯一训练入口。`all` 矩阵包含 10 个
主模型/基线、29 个指定消融，以及 Router、数值编码和图编码控制实验。完整清单由
`src/orchestrate.py` 的 `ALL_EXPERIMENTS` 唯一生成。

`bash scripts/dyphrag_env.sh` 先做 fail-fast preflight，再运行全部 seed：检查依赖、
配置、真实数据和医学语料；多 GPU 调度；Rich 实时状态；独立日志与 checkpoint；
成功跳过和中断恢复；最终生成 CSV、JSON、Markdown、多 seed 统计及可复现归档。

代码不能替代真实数据许可、临床标签定义、带许可证的医学语料、人工 relevance/
polarity qrels、充分样本量的统计结论和原论文预处理。这些条件不满足时，结果只能
解释为工程 smoke 或方法学验证，不能作为医学性能结论。
