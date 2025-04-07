import os
import glob
import wandb
import json
import click
import torch
from accelerate import PartialState
from accelerate import Accelerator

from transformers import AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer
from datasets import get_dataset_config_names
# z1 - BBH, z2 - IFEval, z3 - MATH.
# IFEval BBH MATH LVl 5 GPQA MUSR MMLU-PRO

# 1.6k - strategyqa
# 15k - dolly: open_qa 3.6k, closed_qa 1.8k, general_qa 2.2k, classification 2.1k, 
#              brainstorming 1.8k, information_extraction 1.5k, summarization 1.3k, creative_writing 0.7k
# datasets = [["ChilleD/StrategyQA"], ["llm-wizard/dolly-15k-instruction-alpaca-format"]]
datasets = [["brainstorming", "creative_writing"]] # ["open_qa"], 

input_output_map = {
    "open_qa": {"input": "instruction", "output": "output"},
    "brainstorming": {"input": "instruction", "output": "output"},
    "creative_writing": {"input": "instruction", "output": "output"}
}
labels = {"open_qa": 4, "brainstorming": 5, "creative_writing": 6}

max_examples = 2450
print(labels)
scratch_dir = os.environ["SCRATCH"]

def process_dataset(dataset, dataset_name):
    input_key = input_output_map[dataset_name]["input"]
    output_key = input_output_map[dataset_name]["output"]
    dataset = dataset.map(lambda example: {"prompt": example[input_key], "completion": example[output_key]})
    return dataset

def fetch_dataset(dataset):
    dataset_label = ""
    all_datasets = []
    for each_dataset in dataset:
        all_configs = get_dataset_config_names("llm-wizard/dolly-15k-instruction-alpaca-format")
        print(all_configs)
        
        each_dataset_label = labels[each_dataset]
        dataset_label += f"{each_dataset_label}"
        for each_config in all_configs:
            # filter by split "each_dataset"
            dataset = load_dataset("llm-wizard/dolly-15k-instruction-alpaca-format", each_config, trust_remote_code=True).filter(lambda example: example["category"] == each_dataset)
            available_splits = list(dataset.keys())
            print(available_splits)
            dataset = process_dataset(dataset, each_dataset)
            # Choose a consistent split (prefer 'train' if available, otherwise use first available)
            target_split = 'train' if 'train' in available_splits else available_splits[0]
            print(f"Using split '{target_split}' for {each_dataset}/{each_config}")
            dataset = dataset[target_split]
            all_datasets.append(dataset)
            
    # Properly concatenate datasets
    if len(all_datasets) > 1:
        combined_dataset = concatenate_datasets(all_datasets)
    else:
        combined_dataset = all_datasets[0]
    return combined_dataset, dataset_label

@click.command()
@click.option("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
def main(model_name_or_path):
    wandb.init(project="causal-eval", name=model_name_or_path, entity="hanlin-ml")
    
    for dataset in datasets:
        train_dataset, dataset_label = fetch_dataset(dataset)
        # NOTE control for the number of fine-tuning examples
        train_dataset = train_dataset.select(range(max_examples))
        # find the latest checkpoint
        checkpoint_dir = f"{scratch_dir}/causal-eval/models/{model_name_or_path}/{dataset_label}/checkpoint-*"
        if not os.path.exists(checkpoint_dir):
            latest_checkpoint = None
        else:
            latest_checkpoint = max(glob.glob(checkpoint_dir), key=os.path.getctime)
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        
        sft_config = SFTConfig(max_seq_length=512, packing=True, 
                               per_device_train_batch_size=1, per_device_eval_batch_size=1,
                               gradient_accumulation_steps=8,
                               save_steps=200,
                               resume_from_checkpoint=latest_checkpoint,
                               output_dir=f"{scratch_dir}/causal-eval/models/{model_name_or_path}/{dataset_label}/")

        # device_string = PartialState().process_index
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto",
                                                     attn_implementation="flash_attention_2") # device_map="auto", device_map={'':device_string}, 
                                                    #  attn_implementation="flash_attention")
        # accelerator = Accelerator(mixed_precision="bf16")
        # model = accelerator.prepare_model(model)

        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            args=sft_config,
            # tokenizer=None, # The trainer will create a tokenizer from model_name_or_path
        )
        trainer.train()

if __name__ == "__main__":
    main()