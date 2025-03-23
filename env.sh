uv venv $SCRATCH/envs/eval --python 3.11 && source $SCRATCH/envs/eval/bin/activate  && uv pip install pip

source $SCRATCH/envs/eval/bin/activate

uv pip install transformers datasets torch trl lm_eval[vllm] langdetect immutabledict antlr4-python3-runtime==4.11 math-verify==0.7.0

#  uv pip install datasets==2.16.0?
# lm-eval==0.4.8?
# uv pip install numpy==2.0