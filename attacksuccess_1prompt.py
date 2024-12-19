import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import clip
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from MMPDiffusion import StableDiffusionInpaintingModel  # Import your model
import os 
import random

# Define the ProfanityDataset class (same as in your training code)
class ProfanityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.prompts = []
        dataset = pd.read_excel(f"./UnsafePromptImageDataset/labels.xlsx")
       

        encode_labels = {"normal":[], "sexual":[], "violent":[], "disturbing":[], "hateful":[], "political": []}
        path = "/home/shahid/MMA-Diffusion/src/image_space_attack/UnsafePromptImageDataset/prompts/COCO_prompts.txt"
        with open(path, 'r') as file:
            for line in file:
                # print(line.strip())  
                encode_labels["normal"].append(line)
        path = "/home/shahid/MMA-Diffusion/src/image_space_attack/UnsafePromptImageDataset/prompts/Lexica_prompts.txt"
        with open(path, 'r') as file:
            for line in file:
                # print(line.strip())  
                encode_labels["violent"].append(line)
        path = "/home/shahid/MMA-Diffusion/src/image_space_attack/UnsafePromptImageDataset/prompts/Template_prompts.txt"
        with open(path, 'r') as file:
            for line in file:
                # print(line.strip())  
                encode_labels["sexual"].append(line)
        path = "/home/shahid/MMA-Diffusion/src/image_space_attack/UnsafePromptImageDataset/prompts/4chan_prompts.txt"
        with open(path, 'r') as file:
            for line in file:
                # print(line.strip())  
                encode_labels["hateful"].append(line)

        tokenizer = clip.tokenize  # Use CLIP's tokenizer to validate token length
        for idx, data in dataset.iterrows():
            file_path = os.path.join(self.root_dir, f"{data['image_index']}.png")
            
            if os.path.isfile(file_path) and file_path.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                self.image_paths.append(file_path)
                prompt = None
                if data["final_label"]==0:
                    choice = random.choice(list(range(len(encode_labels["normal"]))))
                    prompt = encode_labels["normal"][choice]
                
                if data["final_label"]==1:
                    if data["rater_0"]==1:
                        choice = random.choice(list(range(len(encode_labels["sexual"]))))
                        prompt = encode_labels["sexual"][choice]

                        
                if data["final_label"]==1:
                    if data["rater_0"]==2:
                        choice = random.choice(list(range(len(encode_labels["violent"]))))
                        prompt = encode_labels["violent"][choice]
                        
                if data["final_label"]==1:
                    if data["rater_0"]==4:
                        choice = random.choice(list(range(len(encode_labels["hateful"]))))
                        prompt = encode_labels["hateful"][choice]

                        
                if data["final_label"]==1:
                    if data["rater_0"]==5:
                        choice = random.choice(list(range(len(encode_labels["hateful"]))))
                        prompt = encode_labels["hateful"][choice]
                
                if prompt == None:
                    if data["final_label"]==0:
                        prompt = "Image is safe"
                    else:
                        prompt = "Sex, violence, hate, blood, horror, sexy, flush, black man. "
                # print(prompt)
                self.prompts.append(self.truncate_prompt(prompt, tokenizer))
                
                self.labels.append(1 if data["final_label"]==0 else 0)
        print(f"Finished loading dataset. Total size: {len(self.image_paths)}")

    def __len__(self):
        return len(self.labels)

    def truncate_prompt(self, prompt, tokenizer, context_length=77):
        tokens = tokenizer(prompt, context_length=77, truncate=True)[0]
        if len(tokens) > context_length:
            tokens = tokens[: context_length - 2]  # Reserve space for special tokens
        return tokens

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        text = self.prompts[idx]
        return image, text, label

    def load_image(self, path):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            image = Image.new("RGB", (256, 256))  # Placeholder in case of error
        return image


# Define transformations for images
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        # Add any additional transforms (e.g., normalization)
    ]
)

# Set your root directory
root_dir = "/home/shahid/MMA-Diffusion/src/image_space_attack/UnsafePromptImageDataset/images"

# Load the full dataset
full_dataset = ProfanityDataset(root_dir=root_dir, transform=transform)

# # Split the dataset into training and validation sets
# num_samples = len(full_dataset)
# train_size = int(0.7 * num_samples)
# val_size = num_samples - train_size

# Create validation dataset
# _, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loader for the validation set
batch_size = 4
val_loader = DataLoader(
    full_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# Initialize the model
vocab_size = 10000  # Set to your actual vocabulary size
embed_dim = 512     # Set to your model's embedding dimension
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = StableDiffusionInpaintingModel(vocab_size, embed_dim, device=device)

# Load the trained model weights
model.load_state_dict(torch.load("mma-defense-final.pth"))
model.to(device)
model.eval()

# Define loss function (if needed for evaluation)
criterion_classification = nn.CrossEntropyLoss()

# Initialize lists to store labels and predictions
all_labels = []
all_preds = []
all_probs = []

# Validation loop
with torch.no_grad():
    for images, texts, labels in tqdm(val_loader, desc="Validation", unit="batch"):
        images = images.to(device=device)
        labels = labels.to(device=device)
        # model.to(dtype=model.clip_model.dtype)
        outputs = model(images, texts)
        # outputs = outputs.to(torch.float32)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(outputs, dim=1)
        # Collect labels and predictions
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

# Compute evaluation metrics
try:
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovo")
except ValueError as e:
    print(f"Cannot compute AUC-ROC: {e}")
    auc = None

precision = precision_score(all_labels, all_preds, average="weighted")
recall = recall_score(all_labels, all_preds, average="weighted")
f1 = f1_score(all_labels, all_preds, average="weighted")
accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds)

# Print the evaluation results
print(f"AUC-ROC: {auc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Attack Success Rate: {100-float((accuracy*100))}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
