#!/bin/bash

# Slurm script to submit separate jobs for each model

# Common Slurm parameters
#SBATCH --job-name=sft_models
#SBATCH --output=logs/slurm/sft_%A_%a.out
#SBATCH --error=logs/slurm/sft_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Create necessary directories
mkdir -p logs/slurm
mkdir -p logs/sft

# Define the models to run
MODELS=(
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"
    "meta-llama/Meta-Llama-3-8B"
    "google/gemma-2-9b"
)

# Submit a separate job for each model
for model in "${MODELS[@]}"; do
    model_name=$(basename "$model")
    echo "Submitting job for $model_name"
    
    # Create a temporary script for this specific model
    temp_script=$(mktemp)
    
    cat > "$temp_script" << EOF
#!/bin/bash
#SBATCH --job-name=sft_${model_name}
#SBATCH --output=logs/slurm/sft_${model_name}_%j.out
#SBATCH --error=logs/slurm/sft_${model_name}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner
#SBATCH --account=kempner_sham_lab

source $SCRATCH/envs/rl/bin/activate
echo "Running SFT for $model_name"
python sft.py --model_name_or_path "$model" > logs/sft/${model_name}.log 2>&1
EOF
    
    # Submit the job
    sbatch "$temp_script"
    
    # Clean up the temporary script
    rm "$temp_script"
done

echo "All jobs submitted" 