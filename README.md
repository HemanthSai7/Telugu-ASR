# Telugu-ASR

**Telugu Automatic Speech Recognition (ASR)** system built **from scratch** under a strict parameter budget.

This project was developed as part of a research pilot task to design a compact ASR model for Telugu within **15M parameters**.  
The final model contains **3.26M parameters** and achieves:

- **Word Error Rate (WER): 0.52**  
- **Character Error Rate (CER): 0.21**

---

## 📄 Project Overview

- **Architecture**: Whisper-style encoder–decoder Transformer (**Moonshine** variant)  
- **Subsampling**: FastConformer subsampling with 3× depthwise separable Conv1D (stride=2, kernel size=9)  
- **Embedding dimension**: 192 
- **Positional encoding**: Rotary (RoPE)  
- **Training objective**: Cross-entropy loss  
- **Hyperparameter tuning**: [Optuna](https://optuna.org/) (21 trials)  

This work explores tokenization strategies, efficient modeling, and hyperparameter optimization for building an ASR system in a morphologically rich, low-resource language like **Telugu**.

---
## 📊 Results

| Model Size | WER  | CER  |
|------------|------|------|
| 3.26M      | 0.52 | 0.21 |  
---

## 📂 Repository Structure


├── src/ # Core source code \
│ ├── dataset/ `# Data loading & preprocessing` \
│ ├── model/ `# Model architecture` \
│ ├── utils/ `# Helper functions` \
│ └── ... \
├── configs/ `# Config files for experiments` \
├── train.py \
├── train_optuna.py `# Hyperparameter optimization` \
├── test.py `# Evaluation script` \
├── requirements.txt \
└── README.md


Read the full project report with background, experiments, and analysis:
- Report for this project: [Telugu ASR report](https://api.wandb.ai/links/ondevicevoice/4cbmfkpm) 
- The Model inference output file is available at: [Model inference](https://github.com/HemanthSai7/Telugu-ASR/blob/main/data/logs/model/2025-09-04/output.tsv)



