# DyPH-RAG implementation summary

## 2026-07-27 rigor pass

- Added explicit total/source retrieval gates, genuine prior-only routing,
  post-retrieval sufficiency/stop decisions and retrieval-cost supervision.
- Replaced full-trajectory peer coarse vectors with cutoff-safe train-patient
  snapshots; cohort nodes now use the retrieved snapshot representation.
- Added disease-conditioned numeric FiLM encoding, patient-state nodes,
  pre/post treatment state edges and state-to-state temporal transitions.
- Added concept-level evidence alignment, confidence-aware derived-edge
  updates and corrected `reweight_only` semantics.
- Made polarity classification disease- and source-conditioned; external
  medical documents now declare `target_diseases`, so evidence for another
  disease becomes differential rather than support.
- Replaced jointly learned temperature with validation-only post-hoc
  temperature scaling and validation-only selective threshold fitting.
- Added a complete-matrix preflight, a refined innovation specification and a
  disease-conditioned numeric control configuration.
- Current validation: 29 unit/integration tests pass; the full-model CPU smoke
  completes with nonzero three-source budgets and a saved calibrated
  checkpoint. No medical performance result is claimed.

## 2026-07-27 completion pass

- Made the Router query explicitly `[h_patient; h_disease; interaction; delta_t; context]`, added per-source continuation flags, exact bounded budget allocation and trainable stopping supervision.
- Rebuilt the peer corpus as one profile per unique training patient and retained coarse patient retrieval followed by fine event/hyperedge reranking.
- Added actual visit/event/episode/numeric-trend/patient-state self candidates and meaningful random/shuffled negative controls.
- Completed the required node schema, visit nodes, grouped numeric-state hyperedges and all numeric audit attributes.
- Ensured update operators never reweight or deactivate immutable raw EHR hyperedges.
- Added a uniform baseline capability registry and canonical full matrix containing every requested baseline and ablation.
- Added label-masked model queries, content-level train-corpus/cohort/medical hashes and stricter availability/discharge/future audits.
- Added resolvable external medical citation IDs/locators while keeping EHR patient and event references anonymous.
- Added live matrix step/metric display, per-GPU locks, graceful interruption, multi-seed mean/std tables and efficiency aggregation.
- Added a single editable run file (`scripts/dyphrag_env.sh`) and a data/key-free server archive builder.
- Current validation: 24 tests pass, including exact interrupted-vs-uninterrupted checkpoint equality; synthetic CPU full-model smoke and three-job scheduler smoke succeeded.
- TensorBoard, MLflow and Ruff could not be exercised in the current Python environment because those optional/runtime packages are not installed here; pinned installation inputs and fail-loud checks are provided.

## Architectural decisions

- Added src/dyphrag as an isolated framework so existing KARE baselines retain their original behavior.
- Made the patient, target disease, cutoff time and label tuple the only task contract and audited it before index construction.
- Kept the parametric prior outside the evidence type system, preventing accidental citation.
- Used deterministic local hashing for the offline reference encoder; production encoders can replace this interface without changing leakage rules.
- Represented raw EHR hyperedges as immutable and limited deactivation to derived evidence edges.
- Stored only salted or hashed references in traces and exports.
- Used YAML configuration composition with Hydra-compatible command-line semantics to avoid a runtime internet dependency.
- Checkpointed optimizer, cursor and all RNG states for exact continuation.

## Main additions

- src/train.py: uniform training, evaluation, checkpoint/resume and artifact contract.
- src/orchestrate.py: parallel, retryable, resumable experiment matrix and aggregation.
- contracts.py, data.py and leakage.py: typed inputs, MIMIC prepared-data adapter and fail-loud audits.
- retrieval.py and router.py: 3+1 retrieval and dynamic routing.
- hypergraph.py, evidence.py and model.py: typed temporal hypergraph, update loop and evidence-consistent prediction.
- metrics.py, runtime.py, tracking.py and privacy.py: metrics, provenance, MLflow/TensorBoard and privacy.
- configs: datasets, baselines, full model and required ablations.
- tests and scripts: acceptance and automation.

## Validation completed

- Python bytecode compilation and shell syntax checks.
- Twenty-four unit, leakage, schema, orchestration and exact-resume tests.
- CPU synthetic end-to-end full-model smoke.
- Three-job scheduler smoke with state, per-run logs, aggregation and reproducibility bundle.
- Forced interruption and exact tensor equality against an uninterrupted reference checkpoint.
- Upload archive inspection confirming that data/results/keys/cache/Git history are excluded.
