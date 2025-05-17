#!/bin/bash

# Define the models to run
MODELS=(
    # "Qwen/Qwen2.5-14B"
    # "meta-llama/Meta-Llama-3-8B"
    # "google/gemma-2-9b"
    "$SCRATCH/causal-eval/models/Qwen/Qwen2.5-14B/1/checkpoint-483"
    "$SCRATCH/causal-eval/models/Qwen/Qwen2.5-14B/2/checkpoint-84"
    "$SCRATCH/causal-eval/models/Qwen/Qwen2.5-14B/3/checkpoint-1527"
    "$SCRATCH/causal-eval/models/Qwen/Qwen2.5-14B/21/checkpoint-570"
    "$SCRATCH/causal-eval/models/Qwen/Qwen2.5-14B/123/checkpoint-2100"
    "$SCRATCH/causal-eval/models/meta-llama/Meta-Llama-3-8B/1/checkpoint-474"
    "$SCRATCH/causal-eval/models/meta-llama/Meta-Llama-3-8B/2/checkpoint-84"
    "$SCRATCH/causal-eval/models/meta-llama/Meta-Llama-3-8B/3/checkpoint-1467"
    "$SCRATCH/causal-eval/models/meta-llama/Meta-Llama-3-8B/21/checkpoint-561"
    "$SCRATCH/causal-eval/models/meta-llama/Meta-Llama-3-8B/123/checkpoint-2031"
    "$SCRATCH/causal-eval/models/google/gemma-2-9b/1/checkpoint-501"
    "$SCRATCH/causal-eval/models/google/gemma-2-9b/2/checkpoint-81"
    "$SCRATCH/causal-eval/models/google/gemma-2-9b/3/checkpoint-1530"
    "$SCRATCH/causal-eval/models/google/gemma-2-9b/21/checkpoint-585"
    "$SCRATCH/causal-eval/models/google/gemma-2-9b/123/checkpoint-2115"
)

# Submit a separate job for each model
for model in "${MODELS[@]}"; do
    # model_name=$(basename "$model")
    # model_name=$model
    # get the last three parts like getting Qwen/Qwen2.5-7B/1
    model_name=$(echo "$model" | rev | cut -d'/' -f1,2,3,4 | rev)
    task_name="leaderboard"
    out_dir="outs_math/${model_name}/${task_name}/"

    GPUs_per_model=4
    gpu_memory_utilization=0.5

    echo "Submitting job for $model_name"
    echo "Output directory: $out_dir"
    
    # Create a temporary script for this specific model
    temp_script=$(mktemp)
    
    cat > "$temp_script" << EOF
#!/bin/bash
#SBATCH --job-name=eval_${model_name}
#SBATCH --output=logs/eval/eval_${model_name}_%j.out
#SBATCH --error=logs/eval/eval_${model_name}_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=240G

#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --partition=kempner_h100 # _requeue
#SBATCH --constraint=h100
#SBATCH --account=kempner_sham_lab

source $SCRATCH/envs/eval2/bin/activate
echo "Running Eval for $model"
nvidia-smi
lm_eval --model vllm \
    --model_args "pretrained=${model},tensor_parallel_size=${GPUs_per_model},dtype=auto,gpu_memory_utilization=${gpu_memory_utilization}" \
    --tasks $task_name \
    --batch_size auto \
    --output_path "${out_dir}"
EOF
    
    # Submit the job
    sbatch "$temp_script"
    
    # Clean up the temporary script
    rm "$temp_script"
done

echo "All jobs submitted" 