import openai
import json
from tqdm import tqdm
from datasets import load_dataset
import os
from utils import set_seed, query_gpt, append_to_jsonl, setup_logger

def main(dataset_name_or_path: str="google/IFEval",
         split: str="train", gpt_model: str="gpt-4o", save_frequency: int=100):
    # 1) Load the Anthropic/hh-rlhf dataset
    dataset = load_dataset(dataset_name_or_path, split=split)
    dataset_name = dataset_name_or_path.split("/")[-1]
    output_filename = f"data/{dataset_name}_{split}.jsonl"

    all_results = []
    save_results = []
    logger.info("Processing each sample...")

    for i, sample in tqdm(enumerate(dataset)):
        key = sample["key"]
        prompt = sample["prompt"]

        # 6) Query GPT-4
        response = ""
        retry_count = 0
        while response == "":
            try:
                # repeat until the response is not empty
                gpt4_reply = query_gpt(prompt, gpt_model)
                response = gpt4_reply.split("Response:")[1]
            except Exception as e:
                logger.error(f"OpenAI API error on sample {i}: {e}")
                retry_count += 1
                if retry_count > 5:
                    response = gpt4_reply
                    break
                continue


        # 8) Log the result
        record = {
            "id": i,
            "key": key,
            "prompt": prompt,
            "response": response,
        }
        all_results.append(record)
        save_results.append(record)
        
        logger.info(f"[bold green]Processed sample {i}[/bold green]")
        logger.info(f"[yellow]Prompt: {prompt}[/yellow]")
        logger.info(f"[bold red]Response: {response}[/bold red]")
        logger.info("")

        if (i+1) % save_frequency == 0:
            append_to_jsonl(output_filename, save_results)
            logger.info(f"Saved GPT-4 results to '{output_filename}'.")
            save_results = []

    # 11) Save all results to JSON
    output_filename = f"data/{dataset_name}_{split}_all.json"
    with open(output_filename, "w", encoding="utf-8") as f: 
        json.dump(all_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    set_seed(42)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logger = setup_logger()
    main(dataset_name_or_path="google/IFEval", split="train", gpt_model="gpt-4o", save_frequency=100)