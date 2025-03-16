# Project Overview

This repository contains essential files and directories structured for efficient project organization.

## Environment

### Training

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv envs/eval --python 3.11 && source envs/eval/bin/activate  && uv pip install pip
uv pip install transformers datasets torch trl
```

### Evaluation

```bash
git clone git@github.com:huggingface/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout main
uv pip install -e .
bash eval.sh
```

## 📁 Directory Structure

/ (Root Directory) 
│── scripts/ # Contains executable scripts (in Python) 
│── Tables/ # Stores tables, datasets, or other structured data 
│── Figures/ # Holds figures generated from our code


### 📜 Directory Details

#### 1️⃣ `scripts/`
- This folder contains scripts used for reproducing the main results of our paper
- Files:
  - `main.ipynb` – Contains the main part of our analysis, specifically the PCA/ICA for comparing the principle component spaces and LiNGCReL for our causal analysis.
  - `matrix_completion.ipynb` – Compare our approach with existing ones for imputing missing benchmark accuracy data.
  - `find_proximal_benchmark.ipynb` - Find out a benchmark that is best-aligned with the latent causal factors discovered by LiNGCReL.
  - `plot_causal_graph.ipynb` - Plot the exact v.s. inexact structural causal models (SCM).
  - `Qwen_illustrate.ipynb` - Compare the performance of models that use Qwen2.5 pretrained models with different sizes as base model.
  - `utils.py` - Contains the implemention of various algorithms/functions that are used in the `.ipynb` files.

#### 2️⃣ `Tables/`
- Stores tables that are used in our analysis
- Files:
  - `open_llm_leaderboard.csv` – Detailed information of the new open LLM leaderboard. Can be directly loaded from an url link in `main.ipynb`.
  - `open_llm_leaderboard_with_token_size.csv` – Add information about pretraining token size into the leaderboard data and remove the rows where such  data is unavailable. Can be obtained by running `main.ipynb`.
  - `MMLU-by-task-Leaderboard.csv` contains the accuracy of models on each category of the MMLU benchmark.

#### 3️⃣ `Figures/`
- The figures in this folder can all be reproduced by running the scripts.

## 🚀 Usage Instructions
- Simply run all the scripts in the `scripts/` folder to reproduce the results.
- Check the `Tables/` folder for processed data.
- Store all figures in the `Figures/` directory.