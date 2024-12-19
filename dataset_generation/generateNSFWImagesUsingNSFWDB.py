from diffusers import StableDiffusionXLPipeline
import torch
import os
import pandas as pd
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:3" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

model.to(device)
# Set the path to the CSV file containing prompts
file_path = "/home/shahid/MMA-Diffusion/src/image_space_attack/profanity/profanity_en.csv"
df = pd.read_csv(file_path)

# Load the Stable Diffusion XL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo")
pipe.to("cuda:2")

# Set the output directory
output_dir = "/media/NAS/DATASET/UnsafeDataset/images/unsafe"
os.makedirs(output_dir, exist_ok=True)

# Set the path for the CSV file to save image paths and prompts
csv_output_path = "/media/NAS/DATASET/UnsafeDataset/unsafe_prompts_images_sdxl_turbo2.csv"

# Check if the CSV file exists to determine if headers should be written
file_exists = os.path.isfile(csv_output_path)


# Open the CSV file in append mode
with open(csv_output_path, mode='a', newline='') as csvfile:
    fieldnames = ['path', 'prompt']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write headers only if the file does not exist
    if not file_exists:
        writer.writeheader()
    for item in range(162):
    # Loop through the dataset and generate images
        for idx, row in df.iterrows():
            
            adjective, noun, verb, text,text2,text3 = row[:6]  # Adjust according to your CSV column setup
            prompt = f"Generate 2 sentence image description depicting  {adjective}, {noun}, {verb}, {text}, {text2}, or {text3}. "  # Extract 'message' column as prompt
            prompt = prompt.replace("nan","")
            messages = [
                {"role": "user", "content": "You are an amazing describer of image prompts that can be used to generate images. Generate description using trashbtch, bitch, trash, nod "},
                {"role": "assistant", "content": "A confident and unapologetic individual proudly donning the label trashbtch with a bold and unwavering stance. The word bitch is etched into the scene in bold, capital letters, a statement of pride and empowerment for this individual. The surroundings are littered with trash, a deliberate nod to the"},
                {"role": "user", "content": prompt}
            ]

            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

            model_inputs = encodeds.to(device)

            generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
            decoded = tokenizer.batch_decode(generated_ids)
            prompt = decoded[0].split("[/INST]").pop()
            prompt = prompt.replace("</s>","")
            
            if isinstance(prompt, str):  # Ensure prompt is a string
                # Generate images (list of PIL images)
                images = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images
                
                # Loop through each generated image
                for i, image in enumerate(images):
                    # Create a unique image ID
                    image_id = f"{idx}_{item}_nsfwdb"
                    image_filename = f"{image_id}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    # Save the image
                    image.save(image_path)
                    print(f"Iter {item}: Image saved to {image_path}, Prompt: {prompt}")
                    
                    # Append the relative path and prompt to the CSV file
                    relative_path = f"{image_filename}"  # Path after 'unsafe/'
                    writer.writerow({'path': "images/unsafe/"+relative_path, 'prompt': prompt})
                    
                    # Flush the CSV file to ensure the entry is saved
                    csvfile.flush()
