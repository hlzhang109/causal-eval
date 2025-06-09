
source $SCRATCH/envs/eval2/bin/activate

task_name=leaderboard
GPUs_per_model=4
model_replicas=1

for model_name in google/gemma-2-9b Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B meta-llama/Meta-Llama-3-8B
do
    # Attention: For instruction models add the --apply_chat_template and fewshot_as_multiturn option.
    gpu_memory_utilization=0.6 # 0.5 for Qwen2.5-7B on 40GiB GPUs
    echo "model_name: $model_name"
    echo "task_name: $task_name"
    
    lm_eval --model vllm \
        --model_args "pretrained=${model_name},tensor_parallel_size=${GPUs_per_model},dtype=auto,gpu_memory_utilization=${gpu_memory_utilization}" \
        --tasks $task_name \
        --batch_size auto \
        --output_path "outs/${model_name}_${task_name}/"
done
