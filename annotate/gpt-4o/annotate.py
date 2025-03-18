import openai
import json
import time
from tqdm import tqdm
from datasets import load_dataset
import os
import sys
from utils import set_seed, query_gpt, append_to_jsonl, setup_logger

def main(dataset_name_or_path: str="google/IFEval",
         split: str="train", 
         gpt_model: str="gpt-4o", 
         save_frequency: int=10,
         output_dir: str="data"):
    
    # Setup logging
    logger = setup_logger()
    logger.info(f"Starting evaluation with model: {gpt_model}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the IFEval dataset
    try:
        dataset = load_dataset(dataset_name_or_path, split=split)
        logger.info(f"Loaded dataset {dataset_name_or_path}, split: {split}, with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)
    
    dataset_name = dataset_name_or_path.split("/")[-1]
    output_filename = f"{output_dir}/{dataset_name}_{split}.jsonl"
    
    all_results = []
    save_results = []
    
    # Process each sample in the dataset
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        key = sample["key"]
        prompt = sample["prompt"]
        
        # Query GPT model
        response = ""
        retry_count = 0
        max_retries = 5
        
        while response == "" and retry_count <= max_retries:
            try:
                gpt_reply = query_gpt(prompt, gpt_model)
                
                # Extract the actual response content following "Response:"
                if "Response:" in gpt_reply:
                    response = gpt_reply.split("Response:")[1].strip()
                else:
                    # If format isn't followed, use the entire response
                    response = gpt_reply
                    logger.warning(f"Response format not followed for sample {i}")
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"Error on sample {i} (attempt {retry_count}/{max_retries}): {e}")
                
                # Add exponential backoff
                if retry_count <= max_retries:
                    sleep_time = 2 ** retry_count
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to process sample {i} after {max_retries} attempts")
                    response = f"ERROR: Failed after {max_retries} attempts"
                continue
        
        # Log the result
        record = {
            "id": i,
            "key": key,
            "prompt": prompt,
            "response": response,
        }
        
        all_results.append(record)
        save_results.append(record)
        
        logger.info(f"[bold green]Processed sample {i}[/bold green]")
        logger.info(f"[yellow]Prompt: {prompt[:100]}...[/yellow]")
        logger.info(f"[bold red]Response: {response}[/bold red]")
        
        # Save results periodically
        if (i+1) % save_frequency == 0:
            append_to_jsonl(output_filename, save_results)
            logger.info(f"Saved batch results to '{output_filename}' ({len(save_results)} samples)")
            save_results = []
        
        # Add a small delay to avoid rate limits
        time.sleep(0.5)
    
    # Save any remaining results
    if save_results:
        append_to_jsonl(output_filename, save_results)
        logger.info(f"Saved final batch to '{output_filename}' ({len(save_results)} samples)")
    
    # Save complete results as JSON
    final_output = f"{output_dir}/{dataset_name}_{split}_all.json"
    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation complete! Full results saved to '{final_output}'")
    logger.info(f"Processed {len(all_results)} samples in total")

if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    # Set OpenAI API key from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)
    
    # Run the main function
    main(
        dataset_name_or_path="google/IFEval", 
        split="train", 
        gpt_model="gpt-4o", 
        save_frequency=10
    )
