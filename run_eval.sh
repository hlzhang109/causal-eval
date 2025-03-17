
source envs/eval/bin/activate

task_name=leaderboard
GPUs_per_model=8
model_replicas=1

for model_name in meta-llama/Meta-Llama-3-8B # google/gemma-2-9b Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B
do
    # Attention: For instruction models add the --apply_chat_template and fewshot_as_multiturn option.
    # data_parallel_size=${model_replicas}
    gpu_memory_utilization=0.6
    if [ "$model_name" == "google/gemma-2-9b" ]; then
        gpu_memory_utilization=0.2
    elif [ "$model_name" == "Qwen/Qwen2.5-14B" ]; then
        gpu_memory_utilization=0.4
    fi
    if [ "$model_name" == "Qwen/Qwen2.5-7B" ]; then
        GPUs_per_model=4
    fi

    lm_eval --model vllm \
        --model_args "pretrained=${model_name},tensor_parallel_size=${GPUs_per_model},dtype=auto,gpu_memory_utilization=${gpu_memory_utilization}" \
        --tasks $task_name \
        --batch_size auto \
        --output_path "outs_math/${model_name}_${task_name}/"
done
