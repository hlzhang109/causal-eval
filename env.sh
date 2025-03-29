uv venv $SCRATCH/envs/eval2 --python 3.10 && source $SCRATCH/envs/eval2/bin/activate  && uv pip install pip

source $SCRATCH/envs/eval2/bin/activate

# uv pip install transformers datasets torch trl lm_eval[vllm] langdetect immutabledict antlr4-python3-runtime==4.11 math-verify==0.7.0

#  uv pip install datasets==2.16.0?
# lm-eval==0.4.8? torch==2.4.0?
# uv pip install numpy==2.0?
# uv pip install -U flash-attn --no-build-isolation