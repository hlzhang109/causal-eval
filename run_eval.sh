
source $SCRATCH/envs/eval/bin/activate

task_name=leaderboard # ifeval #
GPUs_per_model=4
model_replicas=1

for model_name in Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B # meta-llama/Meta-Llama-3-8B google/gemma-2-9b # 
do
    # Attention: For instruction models add the --apply_chat_template and fewshot_as_multiturn option.
    # data_parallel_size=${model_replicas}
    gpu_memory_utilization=0.4
    # if [ "$model_name" == "google/gemma-2-9b" ]; then
    #     gpu_memory_utilization=0.2
    # elif [ "$model_name" == "Qwen/Qwen2.5-14B" ]; then
    #     gpu_memory_utilization=0.4
    # fi
    # if [ "$model_name" == "Qwen/Qwen2.5-7B" ]; then
    #     GPUs_per_model=4
    # fi
    echo "model_name: $model_name"
    echo "task_name: $task_name"
    # max_model_len=8192,
    lm_eval --model vllm \
        --model_args "pretrained=${model_name},tensor_parallel_size=${GPUs_per_model},dtype=auto,gpu_memory_utilization=${gpu_memory_utilization}" \
        --tasks $task_name \
        --batch_size auto \
        --output_path "outs_math/${model_name}_${task_name}/"
done
