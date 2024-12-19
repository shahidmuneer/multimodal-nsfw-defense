from diffusers import StableDiffusionXLPipeline
import torch
import os
import pandas as pd
import csv

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


# Set the path to the CSV file containing prompts
file_path = "bluemoon_train.csv"
df = pd.read_csv(file_path)

# Load the Stable Diffusion XL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo")
pipe.to("cuda")

# Set the output directory
output_dir = "/media/NAS/DATASET/UnsafeDataset/images/unsafe_without_prompt"
os.makedirs(output_dir, exist_ok=True)

# Set the path for the CSV file to save image paths and prompts
csv_output_path = "/media/NAS/DATASET/UnsafeDataset/image_prompts.csv"

# Check if the CSV file exists to determine if headers should be written
file_exists = os.path.isfile(csv_output_path)

# Open the CSV file in append mode
with open(csv_output_path, mode='a', newline='') as csvfile:
    fieldnames = ['path', 'prompt']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write headers only if the file does not exist
    if not file_exists:
        writer.writeheader()
    
    # Loop through the dataset and generate images
    for idx, row in df.iterrows():
        prompt = row['message']  # Extract 'message' column as prompt
        if isinstance(prompt, str):  # Ensure prompt is a string
            # Generate images (list of PIL images)
            images = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images
            
            # Loop through each generated image
            for i, image in enumerate(images):
                # Create a unique image ID
                image_id = f"{idx}_{i}"
                image_filename = f"{image_id}.png"
                image_path = os.path.join(output_dir, image_filename)
                
                # Save the image
                image.save(image_path)
                print(f"Image saved to {image_path}")
                
                # Append the relative path and prompt to the CSV file
                relative_path = f"{image_filename}"  # Path after 'unsafe/'
                writer.writerow({'path': "images/unsafe/"+relative_path, 'prompt': ""})
                
                # Flush the CSV file to ensure the entry is saved
                csvfile.flush()
