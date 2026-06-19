# Execution Policy

**Tier 1 — Local (4060 + 8845HS)**
Run locally if estimated runtime is:
* < 30 minutes
* < 12GB VRAM
* < 32GB RAM
* dataset < 5GB
* single-model experiments
* feature engineering
* EDA
* CatBoost
* LightGBM
* XGBoost
* H3 generation
* SHAP analysis
* TF-IDF
* FAISS/RAG prototypes

Examples:
✅ Leakage detection
✅ CatBoost benchmark
✅ LightGBM benchmark
✅ H3 feature generation
✅ TF-IDF experiment
✅ OSMnx graph generation
✅ RAG prototype

---

**Tier 2 — Kaggle T4**
Push to Kaggle if:
* 30 min – 6 hour runtime
* embedding generation
* multiple CV folds
* AutoGluon
* Optuna sweeps
* large NLP inference

Examples:
✅ multilingual-e5 embeddings
✅ BGE-M3 embeddings
✅ AutoGluon best_quality
✅ 50-model ensemble search
✅ hyperparameter tuning

---

**Tier 3 — Azure / AMD Credits**
Only if Kaggle becomes limiting.

Examples:
✅ serving demo
✅ vector database
✅ final deployment
✅ large parallel experiments

---

**Tier 4 — Never Run**
Regardless of available GPUs.
❌ Train custom foundation model on 8k rows
❌ STGCN
❌ DCRNN
❌ Graph Transformer
❌ HexConvLSTM
❌ Train LLM from scratch
*(These are dataset mismatches, not compute limitations.)*

---

### AI Assistant Instruction (Gemini)
Before generating code, estimate runtime on:
* RTX 4060 Laptop GPU
* Ryzen 8845HS
* 32GB RAM

1. If estimated runtime is under 30 minutes, generate a LOCAL version.
2. If estimated runtime exceeds 30 minutes, automatically:
   * Create a Kaggle-compatible notebook.
   * Generate requirements.
   * Generate a GitHub-ready project structure.
   * Add Kaggle GPU instructions.
   * Optimize for a T4 GPU.
3. Never default to cloud unless local runtime exceeds 30 minutes.

### GridSight AI Project Specifics
**Run Local:** Leakage audit, Dataset profiling, CatBoost/LightGBM baselines, H3 generation, OSMnx experiments, Retrieval prototype.
**Push to Kaggle:** multilingual-e5, BGE-M3, AutoGluon, large ablation studies, ensemble generation.
