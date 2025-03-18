import random
import numpy as np
import torch
import openai
import json
import logging
import os
import rich
from rich.logging import RichHandler
from rich.console import Console

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console = Console(soft_wrap=True, force_terminal=True, width=rich.get_console().width-5)
    handler = RichHandler(console=console, markup=True, show_time=False, 
                          show_level=False, show_path=True, rich_tracebacks=True)
    logger.addHandler(handler)
    return logger

system_prompt = "You are a helpful assistant evaluating instruction-following ability. For each prompt, provide ONLY a direct response to the specific instruction, prefixed with 'Response: '. Keep your response concise, clear, and strictly follow the instruction without adding explanations or unnecessary information. Your response (excluding the 'Response: ' prefix) should strictly satisfy the length requirement."

def query_gpt(text: str, gpt_model: str="gpt-4o") -> str:
    """
    Query GPT model with a system prompt and user message.
    Returns the model's response.
    """
    try:
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        
        gpt_reply = response['choices'][0]['message']['content'].strip()
        return gpt_reply
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

def append_to_jsonl(output_filename, new_results):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    with open(output_filename, "a", encoding="utf-8") as f:
        for entry in new_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def set_seed(seed: int=42):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
