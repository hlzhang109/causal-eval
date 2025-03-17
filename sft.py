
from transformers import AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer
from datasets import get_dataset_config_names
# z1 - BBH, z2 - IFEval, z3 - MATH.
# IFEval BBH MATH LVl 5 GPQA MUSR MMLU-PRO

datasets = [["lukaemon/bbh"], ["google/IFEval"], ["DigitalLearningGmbH/MATH-lighteval"], 
            ["lukaemon/bbh", "google/IFEval"], ["lukaemon/bbh", "google/IFEval", "DigitalLearningGmbH/MATH-lighteval"]]
datasets = [datasets[-1]] # Debug

input_output_map = {
    "lukaemon/bbh": {"input": "input", "output": "target"},
    "google/IFEval": {"input": "prompt", "output": ""},
    "DigitalLearningGmbH/MATH-lighteval": {"input": "problem", "output": "solution"}
}

labels = dict(zip(datasets[-1], range(1, len(datasets[-1]) + 1)))

print(labels)

def process_dataset(dataset):
    if "MATH" in each_dataset:
        dataset = dataset.filter(lambda example: example["level"] == "Level 5")
    dataset = dataset.map(lambda example: {"text": example["question"] + " " + example["answer"]})
    return dataset

for dataset in datasets:
    dataset_label = ""
    all_datasets = []
    for each_dataset in dataset:
        all_configs = get_dataset_config_names(each_dataset)
        print(all_configs)
        each_dataset_label = labels[each_dataset]
        dataset_label += f"{each_dataset_label}"
        for each_config in all_configs:
            dataset = load_dataset(each_dataset, each_config) # , split="train")
            available_splits = list(dataset.keys())
            print(available_splits)
            dataset = process_dataset(dataset)
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

    model_name_or_path = "meta-llama/Meta-Llama-3-8B" # "google/gemma-2-9b" # "Qwen/Qwen2.5-7B" # "Qwen/Qwen2.5-14B"
    sft_config = SFTConfig(output_dir=f"models/{model_name_or_path}/{dataset_label}/")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    trainer = SFTTrainer(
        model,
        train_dataset=combined_dataset,
        args=sft_config,
    )
    trainer.train()