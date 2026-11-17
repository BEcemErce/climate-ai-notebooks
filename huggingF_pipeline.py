import os

from transformers import pipeline

# Initialize a text generation pipeline and send it to the M2 GPU
generator = pipeline("text-generation", model="gpt2", device="cpu")

# Generate text
output = generator("Running machine learning on a Mac is", max_length=30)
print(output)