import os
import json
import glob
import re

def extract_results_from_paths(base_dir="outs_math"):
    """
    Recursively scan 'base_dir' for JSON results and extract relevant metadata.
    Returns a list of dictionaries containing the path, metadata, and loaded JSON.
    """

    # Pattern to capture paths like:
    # outs_math/google/gemma-2-9b/1/checkpoint-501/leaderboard/...
    #   .../results_2025-03-28T22-47-57.241760.json
    #
    # Adjust the glob pattern as needed:
    # - We'll assume "leaderboard" is part of the path
    # - We want any JSON file that starts with "results_"
    pattern = os.path.join(base_dir, "**", "**", "**", "**", "leaderboard", "**", "results_*.json")

    results_data = []

    for file_path in glob.iglob(pattern, recursive=True):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            results_data.append({
                "file_path": file_path,
                "results": data['results']
            })
            print(f"Loaded {file_path}")
        except Exception as e:
            print(f"Could not parse JSON in {file_path}: {e}")

    return results_data

def main():
    # Example usage
    extracted = extract_results_from_paths("outs_math")
    keys = ["leaderboard_bbh", "leaderboard_ifeval", "leaderboard_math_hard", "leaderboard_gpqa_main", "leaderboard_musr", "leaderboard_mmlu_pro"]
    
    # Print or process the extracted data
    for item in extracted:
        results = item["results"]
        path = item["file_path"]
        for key in keys:
            if key in results:
                print(path)
                print(key)
                print(results[key])
                print("-----")
            
                
if __name__ == "__main__":
    main()