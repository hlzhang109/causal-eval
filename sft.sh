source $SCRATCH/envs/rl/bin/activate

for model in "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B" "meta-llama/Meta-Llama-3-8B" "google/gemma-2-9b" 
do
    model_name=${model##*/}
    echo $model_name
    python train/full_sft.py --model_name_or_path $model > logs/sft/${model_name}.log 2>&1
done