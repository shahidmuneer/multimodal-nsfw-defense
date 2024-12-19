import os
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
import torchvision.transforms as T
import pandas as pd
import clip 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensuring deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Dataset class for text-to-image fine-tuning
class Text2ImgDataset(Dataset):
    def __init__(self, target_image_dir, prompts, tokenizer, image_size=(512, 512)):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.root_dir = target_image_dir
        self.target_image_dir = target_image_dir
        self.target_image_filenames = []
        self.labels = []
        self.prompts = []
        dataset = pd.read_csv(os.path.join(self.root_dir,"train.csv"))
        dataset=dataset.iloc[np.random.permutation(len(dataset))]
        dataset = dataset.head(10000)

        for idx, data in dataset.iterrows():
            file_path = os.path.join(self.root_dir, data[1])
            if  os.path.isfile(file_path) and file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                self.target_image_filenames.append(data[1])
                prompt = str(data[1]) if not pd.isna(data[1]) else "No prompt"
                self.prompts.append(prompt)
                # Assuming label is in data[3] as per original code
                self.labels.append(data[3])

        print(f"Finished loading dataset. Total size: {len(self.target_image_filenames)}")

        assert len(self.target_image_filenames) == len(self.prompts), "Mismatch in dataset lengths: number of prompts and images must match."

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        target_path = os.path.join(self.target_image_dir, self.target_image_filenames[idx])
        target_image = Image.open(target_path).convert("RGB").resize(self.image_size)
        label = self.labels[idx]

        # Transform target image to [-1, 1]
        target_image = (T.ToTensor()(target_image) * 2.0) - 1.0

        prompt = self.prompts[idx]
        tokenized_prompt = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokenized_prompt.input_ids.squeeze(0)

        return {
            'target_image': target_image,
            'input_ids': input_ids,
            'label': torch.tensor(label, dtype=torch.float32)  # Ensure float or long as needed
        }


def train(args):
    # Set seed
    set_seed(args.seed)

    # Load the Stable Diffusion text-to-image pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to("cuda")

    #pipe("sexy woman",strength=0.75, num_inference_steps=10)
    image = pipe("A beautiful girl.").images[0]
    image.save("image.png")
    exit()
    # Switch to training mode
    pipe.unet.train()
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Prepare dataset
    with open(args.prompts_file, 'r') as f:
        prompts = f.read().splitlines()

    dataset = Text2ImgDataset(
        target_image_dir=args.target_image_dir,
        prompts=prompts,
        tokenizer=tokenizer,
        image_size=(512, 512)
    )

    # Split dataset into train and validation (80/20)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        pipe.unet.train()
        train_loss_accum = 0.0

        for batch in tqdm(train_dataloader):
            target_images = batch['target_image'].to('cuda', dtype=torch.float32)
            input_ids = batch['input_ids'].to('cuda')
            labels = batch['label'].to('cuda')  # 0 or 1

            # Encode target images to latents
            with torch.no_grad():
                latents = vae.encode(target_images).latent_dist.sample() * 0.18215
                text_embeddings = text_encoder(input_ids)[0]

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()

            # Add noise to latents (noisy latents is what U-Net tries to denoise)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Predict noise residual
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Compute base loss (MSE between predicted noise and actual noise)
            loss = nn.functional.mse_loss(noise_pred, latents, reduction='none')
            # loss shape: (batch_size, C, H, W)

            # Reduce over spatial dimensions and channels
            loss = loss.mean(dim=(1,2,3))  # Now loss is per-sample

            # Apply penalty based on label:
            # If label == 0, increase loss; if label == 1, decrease loss.
            # Example: multiply loss by 2 if label=0 and by 0.5 if label=1
            weight = torch.where(labels == 0, torch.tensor(2.0, device=loss.device), torch.tensor(0.5, device=loss.device))
            loss = loss * weight

            # Final loss is the mean over the batch
            loss = loss.mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation
        pipe.unet.eval()
        val_loss_accum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                target_images = batch['target_image'].to('cuda', dtype=torch.float32)
                input_ids = batch['input_ids'].to('cuda')
                labels = batch['label'].to('cuda')

                latents = vae.encode(target_images).latent_dist.sample() * 0.18215
                text_embeddings = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()

                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

                val_loss = nn.functional.mse_loss(noise_pred, latents, reduction='none').mean(dim=(1,2,3))
                
                # Apply the same weighting for validation loss (not strictly necessary)
                weight = torch.where(labels == 0, torch.tensor(2.0, device=val_loss.device), torch.tensor(0.5, device=val_loss.device))
                val_loss = val_loss * weight
                val_loss_accum += val_loss.mean().item()

                # Heuristic for accuracy:
                # If loss < threshold, predict label=1 else predict label=0
                # Set a threshold (e.g., 0.05) - you may need to adjust this
                threshold = 0.05
                preds = (val_loss < threshold).float()

                # Compare preds to actual labels
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss_accum / len(val_dataloader)
        accuracy = correct / total if total > 0 else 0.0
        print(f"Validation Loss: {avg_val_loss}, Validation Accuracy: {accuracy}")

    # Save the fine-tuned model
    pipe.save_pretrained(args.save_model_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion Text-to-Image Model with Validation")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--target_image_dir', type=str, required=True, help='Directory with target images')
    parser.add_argument('--prompts_file', type=str, required=True, help='File with prompts, one per line')
    parser.add_argument('--save_model_path', type=str, required=True, help='Path to save the fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    train(args)

    # python unlearning.py --model_path runwayml/stable-diffusion-v1-5 --target_image_dir /media/NAS/DATASET/UnsafeDataset/images --prompts_file /media/NAS/DATASET/UnsafeDataset/train.csv --save_model_path ./ --batch_size 32 --num_epochs 5
