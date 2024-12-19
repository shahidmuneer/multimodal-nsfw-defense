import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split,Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from MMPDiffusion import StableDiffusionInpaintingModel
import os
from tqdm import tqdm
import clip
from torchvision.utils import save_image
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report

# Import your model definitions
# Assuming the code you provided is in a file named 'models.py'
# from models import StableDiffusionInpaintingModel

# For this example, I'll redefine the model classes as per your code snippet
# (This is necessary because we can't import 'models.py' in this environment)

# ... [Include your model definitions here] ...

# Assuming the models are defined as in your code snippet

# Define a custom dataset
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
class ProfanityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.
        Args:
            root_dir (str): Path to the root directory containing subdirectories (0, 1, ..., 9).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Gather all image paths and labels
        self.image_paths = []
        self.labels = []
        self.prompts = []
        dataset = pd.read_csv(f"{self.root_dir}/train.csv")
        # dataset = dataset[]

        tokenizer = clip.tokenize  # Use CLIP's tokenizer to validate token length
        for idx,data in dataset.iterrows():
            file_path = os.path.join(self.root_dir, data[1])
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(file_path)

                prompt = str(data[2]) if not pd.isna(data[2]) else "No prompt"
                self.prompts.append(self.truncate_prompt(prompt,tokenizer))
                self.labels.append(data[3])
        print(f"Finish loading dataset total size {len(self.image_paths)}")
    def __len__(self):
        return len(self.labels)
    
    def truncate_prompt(self, prompt, tokenizer, context_length=77):
        """
        Truncate the prompt to fit within the token limit.
        Args:
            prompt (str): The input text.
            tokenizer (callable): CLIP tokenizer function.
            context_length (int): Maximum token context length.
        Returns:
            str: Truncated prompt.
        """
        tokens = tokenizer(prompt,context_length=77,truncate=True)[0]
        if len(tokens) > context_length:
            tokens = tokens[:context_length - 2]  # Reserve space for special tokens
            # prompt = clip.tokenizer.decode(truncated_tokens)  # Decode back to string
        return tokens


    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.labels[idx]
        text = self.prompts[idx]
        return image,text, label

    def load_image(self, path):
        # Load an image using PIL
        try:
            image = Image.open(path).convert("RGB")  # Convert to RGB
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            image = Image.new("RGB", (244, 244))  # Placeholder in case of error
        return image


# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
    # Add any additional transforms (e.g., normalization)
])

# Simulate data (replace this with your actual data loading logic)
num_samples = 200
vocab_size = 10000  # Vocabulary size for text encoder 
embed_dim = 512     # Embedding dimension

# Placeholder image paths, text data, and labels
# root_dir = "/home/shahid/MMA-Diffusion/src/image_space_attack/CIFAR-10-images/train"
root_dir = "/media/NAS/DATASET/UnsafeDataset"



# Split data into training and validation sets

# Define the dataset
full_dataset = ProfanityDataset(
    root_dir=root_dir,
    transform=transform
)

# Calculate sizes for train and validation splits
num_samples = len(full_dataset)
train_size = int(0.7 * num_samples)  # 70% for training
val_size = num_samples - train_size  # 30% for validation

# Split the dataset
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# Initialize the model
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = StableDiffusionInpaintingModel(vocab_size, embed_dim,device=device)
model.to(device)

# Define loss functions and optimizer
criterion_classification = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3,momentum=0.2, weight_decay=1e-3)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as progress_bar:
        for idx, data in enumerate(progress_bar):
            images, texts, labels = data
            images = images.to(device) 
            model.to(dtype=model.clip_model.dtype)
            labels = labels.to(device=device, dtype=torch.long)
            outputs  = model(images, texts)
            outputs = outputs.to(torch.float32)
            loss = criterion_classification(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.byte()).sum().item()
    epoch_loss = running_loss / len(train_dataset)
    accuracy = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Initialize lists to store labels and predictions
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, texts, labels in val_loader:
            images = images.to(device=device, dtype=torch.float16)
            labels = labels.to(device=device, dtype=torch.long) 
            outputs = model(images, texts)
            outputs = outputs.to(torch.float32)
            loss = criterion_classification(outputs, labels)
            val_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels.byte()).sum().item()
            # Collect labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    val_epoch_loss = val_loss / len(val_dataset)
    val_accuracy = val_correct / val_total

    # Compute metrics
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
    except ValueError as e:
        print(f"Cannot compute AUC-ROC: {e}")
        auc = None

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    torch.save(model.state_dict(), 'mma-defense.pth')
    print(f"Epoch [{epoch+1}/{num_epochs}]"
          f" Train Loss: {epoch_loss:.4f}"
          f" Train Acc: {accuracy:.4f}"
          f" Val Loss: {val_epoch_loss:.4f}"
          f" Val Acc: {val_accuracy:.4f}")
    print(f"AUC-ROC: {auc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

# Save the trained model
    torch.save(model.state_dict(), 'mma-defense-final.pth')
# Save the trained model
