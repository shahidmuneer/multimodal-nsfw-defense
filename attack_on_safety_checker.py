import torch
import torchvision.transforms as T
from PIL import Image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from torch.nn.functional import normalize

# Function to classify an input image
def classify_image(image_path, safety_checker, concept_embeds, concept_embeds_weights):
    """
    Classifies an input image as 'safe' or 'unsafe' using the Stable Diffusion Safety Checker.

    Args:
        image_path (str): Path to the input image.
        safety_checker (StableDiffusionSafetyChecker): Pre-trained safety checker model.
        concept_embeds (torch.Tensor): Preloaded concept embeddings.
        concept_embeds_weights (torch.Tensor): Preloaded concept embeddings weights.

    Returns:
        dict: A dictionary containing the classification results.
    """
    # Load and preprocess the input image
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to("cuda")

    # Process the image through the safety checker
    safety_checker_output = safety_checker({"pixel_values": image_tensor})  # Pass the image tensor as input
    image_embeds = safety_checker_output["image_embeds"]  # Get the image embeddings

    # Normalize the image embeddings and concept embeddings
    image_embeds = normalize(image_embeds, dim=-1)
    concept_embeds = normalize(concept_embeds, dim=-1)

    # Compute cosine similarity between image and concepts
    cos_dist = torch.mm(image_embeds, concept_embeds.t())

    # Classify based on concept thresholds
    results = {"concept_scores": {}, "classification": "safe"}
    for concept_idx, concept_cos in enumerate(cos_dist[0]):
        concept_threshold = concept_embeds_weights[concept_idx].item()
        score = concept_cos.item() - concept_threshold
        results["concept_scores"][concept_idx] = score
        if score > 0:  # Unsafe threshold exceeded
            results["classification"] = "unsafe"
            break

    return results


# Example usage
if __name__ == "__main__":
    # Paths to resources and input image
    image_path = "/home/shahid/pdf2images/images/cars.jpg"
    safety_checker_path = "/home/shahid/MMA-Diffusion/src/image_space_attack/safetychecker.pt"
    concept_embeds_path = "/home/shahid/MMA-Diffusion/src/image_space_attack/concept_embeds.pt"
    concept_embeds_weights_path = "/home/shahid/MMA-Diffusion/src/image_space_attack/concept_embeds_weights.pt"

    # Load the safety checker and concept embeddings
    safety_checker = torch.load(safety_checker_path).to("cuda")
    concept_embeds = torch.load(concept_embeds_path).to("cuda")
    concept_embeds_weights = torch.load(concept_embeds_weights_path).to("cuda")

    # Classify the input image
    classification_results = classify_image(image_path, safety_checker, concept_embeds, concept_embeds_weights)

    # Print the results
    print("Classification Results:", classification_results)
