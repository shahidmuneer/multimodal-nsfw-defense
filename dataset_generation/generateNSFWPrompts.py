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
    output_csv_file = '/media/NAS/DATASET/UnsafeDataset/unsafe_prompts_only.csv'  # Update this to your desired output CSV file path
    
    # Check if the output CSV file exists to determine if headers should be written
    output_csv_exists = os.path.isfile(output_csv_file)

    # Initialize the text-generation pipeline
    text_model_name = "HuggingFaceH4/zephyr-7b-beta"
    text_pipe = pipeline(
        "text-generation",
        model=text_model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to reduce memory usage
        device_map="auto"
    )

    # Initialize a global index for unique image IDs
    global_idx = 0

    # Repeat the processing at least 100 times
    for iteration in range(162):
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
                        prompt_text = f"Write a detailed description using the words '{adjective}', '{noun}', '{verb}', '{text}', {text2}, {text3}, use all words in two or three sentences."

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
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=0.8,
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

                        
                        writer.writerow({'path': "", 'prompt': clean_prompt})

                            # Flush the CSV file to ensure the entry is saved
                        outfile.flush()

                        global_idx += 1  # Increment the global index

                    except Exception as e:
                        print(f"Error processing row {row}: {e}")

if __name__ == "__main__":
    main()
