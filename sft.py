import os
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

datasets = [["google/IFEval"], ["lukaemon/bbh"], ["DigitalLearningGmbH/MATH-lighteval"], 
            ["google/IFEval", "lukaemon/bbh"], ["google/IFEval", "lukaemon/bbh", "DigitalLearningGmbH/MATH-lighteval"]]

input_output_map = {
    "lukaemon/bbh": {"input": "input", "output": "target"},
    "google/IFEval": {"input": "prompt", "output": "response"},
    "DigitalLearningGmbH/MATH-lighteval": {"input": "problem", "output": "solution"}
}

labels = dict(zip(datasets[-1], range(1, len(datasets[-1]) + 1)))

print(labels)
scratch_dir = os.environ["SCRATCH"]

def process_dataset(dataset, dataset_name, ifeval_train):
    if "MATH" in dataset_name:
        dataset = dataset.filter(lambda example: example["level"] == "Level 5")
    elif "IFEval" in dataset_name:
        # add "response" from ifeval_train to the dataset according to prompt matching
        dataset = dataset.map(lambda example: {"response": ifeval_train[example["prompt"]]})
    input_key = input_output_map[dataset_name]["input"]
    output_key = input_output_map[dataset_name]["output"]
    dataset = dataset.map(lambda example: {"prompt": example[input_key], "completion": example[output_key]})
    return dataset

def fetch_dataset(dataset):
    dataset_label = ""
    all_datasets = []
    for each_dataset in dataset:
        if "IFEval" in each_dataset:
            ifeval_train = json.load(open("data/IFEval_train_all_regenerated.json"))
            ifeval_train = {example["prompt"]: example["response"] for example in ifeval_train}
        else:
            ifeval_train = None
        all_configs = get_dataset_config_names(each_dataset)
        print(all_configs)
        
        each_dataset_label = labels[each_dataset]
        dataset_label += f"{each_dataset_label}"
        for each_config in all_configs:
            dataset = load_dataset(each_dataset, each_config)
            available_splits = list(dataset.keys())
            print(available_splits)
            dataset = process_dataset(dataset, each_dataset, ifeval_train)
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
        sft_config = SFTConfig(max_seq_length=512, packing=True, 
                               per_device_train_batch_size=1, per_device_eval_batch_size=2,
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