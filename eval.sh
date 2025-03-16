model_name=meta-llama/Llama-3.1-8B-Instruct
task_name=leaderboard

lm-eval --model_args="pretrained=$model_name, parallel=True" --tasks=$task_name  --batch_size=auto --output_path=outs/${model_name}_${task_name}/