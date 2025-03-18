import json
import openai
import time
import os
import re
import random
from tqdm import tqdm
from utils import setup_logger

# Modified system prompt to better handle edge cases
MODIFIED_SYSTEM_PROMPT = """You are an AI designed to respond to all instructions accurately and helpfully. For each prompt, provide a direct response to the specific instruction, prefixed with 'Response: '.
Your task is to demonstrate instruction-following ability by providing a compliant response even when instructions request unusual formats or lengths.
- Always provide a response that follows the instruction exactly as given
- For creative or hypothetical scenarios, provide appropriate content
- Use the exact format requested in the instruction
- Respond to all requests that would be legal, ethical, and harmless
- Your response should be concise and focused on fulfilling the instruction"""

# Patterns that indicate refusals
REFUSAL_PATTERNS = [
    r"(?i)I('m| am) (sorry|afraid)",
    r"(?i)I cannot|I can't|I'm unable",
    r"(?i)As an AI|As a language model",
    r"(?i)I (do not|don't) (feel comfortable|think it's appropriate)",
    r"(?i)It would (not|n't) be (appropriate|ethical)",
    r"(?i)I (can't|cannot|won't) assist",
    r"(?i)I apologize",
    r"(?i)unable to (provide|fulfill|comply)"
]

def is_refusal(response):
    """Check if the response is a refusal."""
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, response):
            return True
    return False

def query_gpt_with_retry(prompt, model="gpt-4o", max_retries=3):
    """Query GPT with retry mechanism and modified system prompt."""
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": MODIFIED_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,  # Slightly higher temperature for creative responses
            )
            
            reply = response['choices'][0]['message']['content'].strip()
            
            # Extract the response part if it follows our format
            if "Response:" in reply:
                reply = reply.split("Response:")[1].strip()
                
            # Check if it's still a refusal
            if is_refusal(reply) and retry_count < max_retries:
                retry_count += 1
                logger.info(f"Refusal detected. Retrying ({retry_count}/{max_retries})...")
                time.sleep(2 ** retry_count)  # Exponential backoff
                continue
                
            return reply
            
        except Exception as e:
            retry_count += 1
            logger.error(f"API error: {e}")
            if retry_count <= max_retries:
                sleep_time = 2 ** retry_count + random.uniform(0, 1)
                logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                return f"ERROR: Failed after {max_retries} attempts"
    
    return "ERROR: Maximum retries exceeded"

def regenerate_refusals(input_file, output_file=None):
    """Regenerate responses for entries that look like refusals."""
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_regenerated{ext}"
    
    logger.info(f"Loading data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} entries")
    
    # Count refusals
    refusal_count = sum(1 for entry in data if is_refusal(entry.get('response', '')))
    logger.info(f"Found {refusal_count} potential refusals out of {len(data)} entries")
    
    if refusal_count == 0:
        logger.info("No refusals found. Exiting.")
        return
    
    # Regenerate responses for refusals
    regenerated_count = 0
    for i, entry in tqdm(enumerate(data), desc="Regenerating responses", total=len(data)):
        response = entry.get('response', '')
        
        if is_refusal(response):
            logger.info(f"Entry {entry['id']} appears to be a refusal:")
            logger.info(f"Prompt: {entry['prompt'][:100]}...")
            logger.info(f"Original response: {response}")
            
            # Attempt to regenerate
            new_response = query_gpt_with_retry(entry['prompt'])
            
            if not is_refusal(new_response):
                logger.info(f"Regenerated: {new_response}")
                entry['response'] = new_response
                entry['regenerated'] = True
                regenerated_count += 1
            else:
                logger.info(f"Still refused after retry: {new_response}")
                entry['regeneration_attempted'] = True
        
        # Save progress periodically
        if (i+1) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Progress saved ({i+1}/{len(data)})")
    
    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Regeneration complete. Successfully regenerated {regenerated_count} out of {refusal_count} refusals.")
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Set up logging
    logger = setup_logger()
    
    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("Error: OPENAI_API_KEY environment variable is not set")
        exit(1)
    
    # File paths
    input_file = "data/IFEval_train_all.json"  # Change this to your file path
    
    # Run regeneration
    regenerate_refusals(input_file)
