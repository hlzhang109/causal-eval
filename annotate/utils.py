import random
import numpy as np
import torch
import openai
import json
import logging
import rich
from rich.logging import RichHandler
from rich.console import Console

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console = Console(soft_wrap=True, force_terminal=True, width=rich.get_console().width-5) # , highlight=False)
    handler = RichHandler(console=console, markup=True, show_time=False, \
                          show_level=False, show_path=True, rich_tracebacks=True)
    logger.addHandler(handler)
    return logger

system_prompt = "You are a helpful assistant that can help me come up with very short responses that follow the given prompt instruction: \
                  \"Response: <response>\". \
                  Your must output the keyword \"Response\" in the beginning of your response. \
                  The response should be concise and to the point, and should not be longer than 10 words."

def query_gpt(text: str, gpt_model: str="gpt-4o-mini") -> str:
    """
    Query GPT-4 with a shared context and two possible final responses.
    GPT-4 should pick '1' or '2'.
    """
    
    response = openai.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
    )

    gpt4_reply = response.choices[0].message.content.strip()
    return gpt4_reply

def append_to_jsonl(output_filename, new_results):
    with open(output_filename, "a", encoding="utf-8") as f:
        for entry in new_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")  # Write each entry as a line

def set_seed(seed: int=42):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    