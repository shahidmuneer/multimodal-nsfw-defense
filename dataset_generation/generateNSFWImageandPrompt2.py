import csv
import torch
from transformers import pipeline
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Define the function to remove escape sequences
def remove_escape_sequences(s):
    # Define the escape characters to remove
    escape_chars = ['\n', '\t', '\r', '\b', '\f', '\v', '\a']
    for char in escape_chars:
        s = s.replace(char, '')
    return s

def main():
    # Paths
    input_csv_file = '/home/shahid/MMA-Diffusion/src/image_space_attack/profanity/profanity_en.csv'  # Update this to your input CSV file path
    output_csv_file = '/media/NAS/DATASET/UnsafeDataset/image_prompts_3.csv'  # Update this to your desired output CSV file path
    output_dir = '/media/NAS/DATASET/UnsafeDataset/images/unsafe_3'  # Update this to your desired image output directory

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the text-generation pipeline
    text_model_name = "HuggingFaceH4/zephyr-7b-beta"
    text_pipe = pipeline(
        "text-generation",
        model=text_model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to reduce memory usage
        device_map="auto"
    )

    # Load Stable Diffusion XL pipeline with custom UNet
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use the correct checkpoint for your step setting!

    # Load model
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    image_pipe = StableDiffusionXLPipeline.from_pretrained(
        base,
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    # Ensure sampler uses "trailing" timesteps
    image_pipe.scheduler = EulerDiscreteScheduler.from_config(
        image_pipe.scheduler.config,
        timestep_spacing="trailing"
    )

    # Check if the output CSV file exists to determine if headers should be written
    output_csv_exists = os.path.isfile(output_csv_file)

    # Initialize a global index for unique image IDs
    global_idx = 8636

    # Repeat the processing at least 100 times
    for iteration in range(100):
        print(f"Starting iteration {iteration + 1}")

        # Open the input CSV file
        with open(input_csv_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)

            # Open the output CSV file in append mode
            with open(output_csv_file, mode='a', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=['path', 'prompt'])

                # Write headers only if the file does not exist and it's the first iteration
                if not output_csv_exists and iteration == 0:
                    writer.writeheader()

                for row in reader:
                    try:
                        # Example row format assumed: adjective, noun, verb, text
                        adjective, noun, verb, text,text2,text3 = row[:6]  # Adjust according to your CSV column setup

                        # Generate prompt using the words
                        prompt_text = f"Write a detailed image description using the words '{adjective}', '{noun}', '{verb}', '{text}', {text2}, {text3}, use all words in two or three sentences."

                        # Prepare the messages for the chat template
                        messages = [
                            {
                                "role": "system",
                                "content": "You are an expert AI Image prompt generator.",
                            },
                            {
                                "role": "user",
                                "content": prompt_text,
                            },
                        ]

                        # Apply the chat template to format the prompt
                        prompt = text_pipe.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )

                        # Generate the response
                        outputs = text_pipe(
                            prompt,
                            max_new_tokens=77,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.95
                        )

                        # Extract the generated text
                        generated_prompt = outputs[0]["generated_text"]
                        prompt_content = generated_prompt.split('<|assistant|>')[1]
                        clean_prompt = remove_escape_sequences(prompt_content).strip()

                        # Print the results
                        print(f"Iteration {iteration + 1}, Global Index {global_idx}")
                        print(f"Input Words: {row}")
                        print(f"Generated Prompt: {clean_prompt}")

                        # Generate image using the generated prompt
                        images = image_pipe(
                            clean_prompt,
                            num_inference_steps=4,
                            guidance_scale=0
                        ).images

                        # Save the image(s) and write to CSV
                        for i, image in enumerate(images):
                            # Create a unique image ID
                            image_id = f"{global_idx}"
                            image_filename = f"{image_id}.png"
                            image_path = os.path.join(output_dir, image_filename)

                            # Save the image
                            image.save(image_path)
                            print(f"Image saved to {image_path}")

                            # Append the relative path and prompt to the CSV file
                            relative_path = os.path.relpath(image_path, start=os.path.dirname(output_csv_file))
                            writer.writerow({'path': relative_path, 'prompt': clean_prompt})

                            # Flush the CSV file to ensure the entry is saved
                            outfile.flush()

                            global_idx += 1  # Increment the global index

                    except Exception as e:
                        print(f"Error processing row {row}: {e}")

if __name__ == "__main__":
    main()
