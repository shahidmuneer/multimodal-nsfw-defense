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
import os 
from MMPDiffusion import StableDiffusionInpaintingModel  # Import your model

# Define the ProfanityDataset class (same as in your training code)
class ProfanityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.prompts = []
        dataset = pd.read_csv(f"{self.root_dir}/image_prompts.csv")
        tokenizer = clip.tokenize  # Use CLIP's tokenizer to validate token length
        for idx, data in dataset.iterrows():
            file_path = os.path.join(self.root_dir, data[0])
            print(file_path)
            if  os.path.isfile(file_path) and file_path.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                self.image_paths.append(file_path)
                prompt = str(data[1]) if not pd.isna(data[1]) else "No prompt"
                self.prompts.append(self.truncate_prompt(prompt, tokenizer))
                self.labels.append(0)
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
        image = torch.randn_like(image)#Image.new("RGB", (256, 256)) 
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
root_dir = "/home/shahid/MMA-Diffusion/src/image_space_attack/dataset_generation/sneakyPrompts/"

# Load the full dataset
full_dataset = ProfanityDataset(root_dir=root_dir, transform=transform)

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




        # Variables to calculate adjusted batch accuracy
total_batches = 0
correct_batches = 0
# Initialize lists to store labels and predictions
all_labels = []
all_preds = []
all_probs = []

# Validation loop
with torch.no_grad():
    for images, texts, labels in tqdm(val_loader, desc="Validation", unit="batch"):
        images = images.to(device=device, dtype=torch.float16)
        labels = labels.to(device=device, dtype=torch.long)
        model.to(dtype=model.clip_model.dtype)
        outputs = model(images, texts)
        outputs = outputs.to(torch.float32)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(outputs, dim=1)
        # Collect labels and predictions
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())
        # Batch-wise accuracy adjustment
        total_batches += 1
        # Check if all predictions in the batch are correct
        batch_failed = (predicted == labels).all().item()
        if batch_failed:  # If no misclassification, batch is correct
            correct_batches += 1
# Compute evaluation metrics

# Calculate adjusted accuracy based on batches
attack_success_rate = 1 - (correct_batches / total_batches)
print(f"Attack Success Rate: {attack_success_rate * 100:.2f}%")

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
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
