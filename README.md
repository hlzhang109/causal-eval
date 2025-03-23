# Project Overview

This repository contains essential files and directories structured for efficient project organization.

## Environment

### Training

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv $SCRATCH/envs/eval --python 3.11 && source $SCRATCH/envs/eval/bin/activate  && uv pip install pip
uv pip install transformers datasets torch trl
```

### Evaluation

```bash
git clone git@github.com:huggingface/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout main
uv pip install -e .
```

Double check https://github.com/EleutherAI/lm-evaluation-harness/pull/2772/commits/7207e4ea33df5bd6a10c872f2fac4993ea727067
envs/eval/lib/python3.11/site-packages/lm_eval/tasks/leaderboard/math/utils.py

Alternatively,

```bash
uv pip install lm_eval\[vllm\]
uv pip install langdetect immutabledict antlr4-python3-runtime==4.11
uv pip install math-verify==0.7.0
```

Then,
```bash
bash eval.sh
```

Maybe hack the vllm code to allow prompt length = max_model_len.
```python
# vllm/v1/engine/processor.py
if len(prompt_ids) > self.model_config.max_model_len:
    raise ValueError(
        f"Prompt length of {len(prompt_ids)} is longer than the "
        f"maximum model length of {self.model_config.max_model_len}.")
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