import os
import multiprocessing

# 1. These MUST be the absolute first lines, before PyTorch is imported
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_model():
    # 2. Import heavy libraries inside the function
    import torch
    from transformers import pipeline
    
    print("Environment secured. Loading GPT-2...")
    
    # 3. Load the model onto the Mac GPU
    generator = pipeline("text-generation", model="gpt2", device="mps")
    
    print("Model loaded. Generating text...")
    output = generator("Running machine learning on a Mac is", max_length=30)
    
    print("\n--- Result ---")
    print(output)

# 4. This block prevents macOS from duplicating processes incorrectly
if __name__ == '__main__':
    # Force macOS to spawn safe processes instead of forking
    multiprocessing.set_start_method('spawn', force=True)
    run_model()