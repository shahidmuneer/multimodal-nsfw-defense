import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import clip
import torchvision.transforms as transforms
from src.diffusion.unet import UNet

import math

import einops

from torchvision.transforms import functional as F_tv
class GELU_PyTorch_Tanh(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)))
    
# class VisionEncoder(nn.Module):
#     def __init__(self, device):
#         super(VisionEncoder, self).__init__()
#         # Load the CLIP model
#         self.model, _ = clip.load("ViT-B/32", device=device)
#         # We only need the visual encoder
#         self.visual = self.model.visual
#         self.device = device

#     def forward(self, x):
#         # Ensure correct data type
#         x = x.to(device=self.device,dtype=self.model.dtype)
#         print(x.shape)
#         embeddings = self.visual(x)
#         return embeddings
    
# class TextEncoder(nn.Module):
#     def __init__(self, device):
#         super(TextEncoder, self).__init__()
#         # Load the CLIP model
#         self.device = device
#         self.model, _ = clip.load("ViT-B/32", device=device)
#         # We only need the text encoder
#         self.encode_text = self.model.encode_text

#     def forward(self, text_tokens):
        
#         text_tokens = text_tokens.to(device=self.device,dtype=self.model.dtype)
#         embeddings = self.encode_text(text_tokens)
#         return embeddings
    
# class VisionEncoder(nn.Module):
#     def __init__(self, embed_dim):
#         super(VisionEncoder, self).__init__()
#         # Pretrained ResNet50 model
#         self.backbone = models.resnet50(pretrained=True)
#         # Replace the final layer with an embedding layer
#         self.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)
#         self.backbone.fc = nn.Identity()
    
#     def forward(self, x):
#         features = self.backbone(x)  # Extract features
#         embeddings = self.fc(features)  # Project to embedding dimension
#         return embeddings

# class TextEncoder(nn.Module):
#     def __init__(self, vocab_size, embed_dim):
#         super(TextEncoder, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
#     def forward(self, x):
#         embeddings = self.embedding(x)  # Embed input tokens
#         embeddings = embeddings.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, embed_dim)
#         output = self.transformer(embeddings)
#         output = output.permute(1, 0, 2)  # Back to (batch_size, seq_len, embed_dim)
#         # Pooling to get a single embedding per input
#         output = output.mean(dim=1)
#         return output

class MultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, emb_dim: int) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.emb_dim = emb_dim

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        self.norm = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )

    def forward(self, x, t, text_embeddings):
        # x: Image embeddings from CLIP [batch_size, emb_dim]
        # text_embeddings: Text embeddings from CLIP [batch_size, emb_dim]

        batch_size = x.size(0)

        # Concatenate image and text embeddings along sequence dimension
        # So sequence length is 2
        embeddings = torch.stack([x, text_embeddings], dim=1)  # [batch_size, 2, emb_dim]
        embeddings = self.norm(embeddings)

        # Linear projections
        q = self.q_proj(embeddings)  # [batch_size, 2, emb_dim]
        k = self.k_proj(embeddings)
        v = self.v_proj(embeddings)

        # Reshape for multi-head attention
        q = q.view(batch_size, 2, self.n_heads, self.head_dim)
        k = k.view(batch_size, 2, self.n_heads, self.head_dim)
        v = v.view(batch_size, 2, self.n_heads, self.head_dim)

        # Transpose to get dimensions [batch_size, n_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, n_heads, seq_len, seq_len]
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)  # [batch_size, n_heads, seq_len, head_dim]

        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, n_heads, head_dim]
        attn_output = attn_output.view(batch_size, 2, self.emb_dim)

        # Output projection
        output = self.out_proj(attn_output)  # [batch_size, 2, emb_dim]

        # Residual connection and MLP
        output = output + embeddings  # Residual connection
        output = self.mlp(output) + output  # Feed-forward network

        # Split the outputs back to image and text embeddings
        img_embed_out, text_embed_out = output[:, 0, :], output[:, 1, :]

        return img_embed_out,

class MultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, emb_dim: int) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.emb_dim = emb_dim

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        self.norm = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )

    def forward(self, x,  text_embeddings):
        # x: Image embeddings from CLIP [batch_size, emb_dim]
        # text_embeddings: Text embeddings from CLIP [batch_size, emb_dim]

        batch_size = x.size(0)

        # Concatenate image and text embeddings along sequence dimension
        # So sequence length is 2
        embeddings = torch.stack([x, text_embeddings], dim=1)  # [batch_size, 2, emb_dim]
        embeddings = self.norm(torch.flatten(embeddings,start_dim=1,end_dim=2))

        # Linear projections
        q = self.q_proj(embeddings)  # [batch_size, 2, emb_dim]
        k = self.k_proj(embeddings)
        v = self.v_proj(embeddings)

        # Reshape for multi-head attention
        q = q.view(batch_size, 2, self.n_heads, self.head_dim)
        k = k.view(batch_size, 2, self.n_heads, self.head_dim)
        v = v.view(batch_size, 2, self.n_heads, self.head_dim)

        # Transpose to get dimensions [batch_size, n_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, n_heads, seq_len, seq_len]
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)  # [batch_size, n_heads, seq_len, head_dim]

        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, n_heads, head_dim]
        attn_output = attn_output.view(batch_size, 2, self.emb_dim)

        # Output projection
        output = self.out_proj(attn_output)  # [batch_size, 2, emb_dim]

        # Residual connection and MLP
        output = output + embeddings  # Residual connection
        output = self.mlp(output) + output  # Feed-forward network

        

        return output  # Return the updated image embeddings  # Return the updated image embeddings
    
class UNetCustom(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super(UNetCustom, self).__init__()
        # Define the downsampling path
        self.down1 = self.conv_block(in_channels, 64)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.down4 = self.conv_block(256, 512)
        # Bottleneck layer
        self.bottleneck = self.conv_block(512, 1024)
        
        # Define the upsampling path with corrected input channels
        self.up4 = self.up_conv(1024 + embed_dim, 512)
        self.up3 = self.up_conv(512 + 512 + embed_dim, 256)  # 512 from u4, 512 from d4, embed_dim from embeddings
        self.up2 = self.up_conv(256 + 256 + embed_dim, 128)
        self.up1 = self.up_conv(128 + 128 + embed_dim, 64)
        
        # Final output layer
        self.final = nn.Conv2d(640, out_channels, kernel_size=1)

        
        self.down_a1 = self.conv_block(3, 32)
        self.down_a2 = self.conv_block(32, 32)
        self.down_a3 = self.conv_block(32, 64)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 2048), 
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            GELU_PyTorch_Tanh(),  # Using standard GELU instead of custom; replace if necessary
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1),  # 二分类问题，输出维度为1
            # nn.Sigmoid()
        )
    
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Added padding=1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def up_conv(self, in_channels, out_channels):
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        conv = self.conv_block(out_channels, out_channels)
        return nn.Sequential(up, conv)
    
    def forward(self, x, embeddings):
        # Downsampling
        d1 = self.down1(x)  # Shape: [batch_size, 64, 256, 256]
        d2 = self.down2(F.max_pool2d(d1, 2))  # Shape: [batch_size, 128, 128, 128]
        d3 = self.down3(F.max_pool2d(d2, 2))  # Shape: [batch_size, 256, 64, 64]
        d4 = self.down4(F.max_pool2d(d3, 2))  # Shape: [batch_size, 512, 32, 32]
        
        # Bottleneck
        bn = self.bottleneck(F.max_pool2d(d4, 2))  # Shape: [batch_size, 1024, 16, 16]
        
        # Prepare embeddings for concatenation at bottleneck level
        embeddings_bn = embeddings.unsqueeze(-1).unsqueeze(-1)
        embeddings_bn = embeddings_bn.repeat(1, 1, bn.size(2), bn.size(3))  # Shape: [batch_size, embed_dim, 16, 16]
        
        # Concatenate embeddings with bottleneck output
        bn = torch.cat([bn, embeddings_bn], dim=1)  # New shape: [batch_size, 1024 + embed_dim, 16, 16]
       
        # Upsampling with skip connections and embeddings

        # Up4
        u4 = self.up4(bn)  # Shape: [batch_size, 512, 32, 32]
        # Prepare embeddings for u4
        embeddings_u4 = embeddings.unsqueeze(-1).unsqueeze(-1)
        embeddings_u4 = embeddings_u4.repeat(1, 1, u4.size(2), u4.size(3))  # Shape: [batch_size, embed_dim, 32, 32]
        u4 = torch.cat([u4, d4, embeddings_u4], dim=1)  # New shape: [batch_size, 512 + 512 + embed_dim, 32, 32]

        # Up3
        u3 = self.up3(u4)  # Shape: [batch_size, 256, 64, 64]
        embeddings_u3 = embeddings.unsqueeze(-1).unsqueeze(-1)
        embeddings_u3 = embeddings_u3.repeat(1, 1, u3.size(2), u3.size(3))  # Shape: [batch_size, embed_dim, 64, 64]
        u3 = torch.cat([u3, d3, embeddings_u3], dim=1)  # New shape: [batch_size, 256 + 256 + embed_dim, 64, 64]

        # Up2
        u2 = self.up2(u3)  # Shape: [batch_size, 128, 128, 128]
        embeddings_u2 = embeddings.unsqueeze(-1).unsqueeze(-1)
        embeddings_u2 = embeddings_u2.repeat(1, 1, u2.size(2), u2.size(3))  # Shape: [batch_size, embed_dim, 128, 128]
        u2 = torch.cat([u2, d2, embeddings_u2], dim=1)  # New shape: [batch_size, 128 + 128 + embed_dim, 128, 128]

        # Up1
        u1 = self.up1(u2)  # Shape: [batch_size, 64, 256, 256]
        embeddings_u1 = embeddings.unsqueeze(-1).unsqueeze(-1)
        embeddings_u1 = embeddings_u1.repeat(1, 1, u1.size(2), u1.size(3))  # Shape: [batch_size, embed_dim, 256, 256]
        u1 = torch.cat([u1, d1, embeddings_u1], dim=1)  # New shape: [batch_size, 64 + 64 + embed_dim, 256, 256]
        
        output = self.final(u1)
        
        down_a1 = self.down_a1(F.max_pool2d(output, 3)) 
        down_a2 = self.down_a2(F.max_pool2d(down_a1, 3)) 
        down_a3 = self.down_a3(F.max_pool2d(down_a2, 3)) 
        classification = self.classifier(down_a3)

        # classification = self.linear()
        return classification
    

class StableDiffusionInpaintingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, in_channels=3, out_channels=3, T=1000, steps=[10], num_classes=2, device="cuda:3"):
        super(StableDiffusionInpaintingModel, self).__init__()
        self.device = device
        self.T = T

        # Load CLIP model and convert to float32
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model = self.clip_model.float()  # Convert model to float32
        self.clip_model.eval()

        self.vision_encoder = self.clip_model.visual
        self.text_encoder = self.clip_model.encode_text

        # Reparameterization layers
        self.image_reparam = nn.Linear(embed_dim, embed_dim).to(dtype=torch.float32)
        self.text_reparam = nn.Linear(embed_dim, embed_dim).to(dtype=torch.float32)

        self.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True,
            device=device,
            dtype=torch.float32
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            GELU_PyTorch_Tanh(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            GELU_PyTorch_Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2),
        )

    def preprocess_image(self, image):
        # Ensure image is preprocessed correctly and converted to float32
        image = self.preprocess(image).to(self.device)
        image = image.float()  # Convert image to float32
        return image


    def preprocess_images(self, images):
        # Convert tensor to PIL Image if needed
        processed_images=[]
        for image in images:
            if isinstance(image, torch.Tensor):
                # Assuming image is in the range [0, 1]
                image = F_tv.to_pil_image(image)  # Use the first image in the batch
                processed_images.append(self.preprocess_image(image))
        return  torch.stack(processed_images).to(device=self.device,dtype=self.clip_model.dtype)
    
    def forward(self, image, text):
        image = self.preprocess_images(image)
        image_embed = self.vision_encoder(image)
        image_embed = self.image_reparam(image_embed)

        text_tokens = text.to(self.device)#clip.tokenize(text).to(self.device)
        text_embed = self.text_encoder(text_tokens)
        text_embed = self.text_reparam(text_embed)

        # Prepare embeddings for MultiheadAttention
        image_embed = image_embed.unsqueeze(1)  # Shape: [batch_size, 1, embed_dim]
        text_embed = text_embed.unsqueeze(1)    # Shape: [batch_size, 1, embed_dim]
        # print(image_embed.shape,text_embed.shape)
        # Apply MultiheadAttention
        attn_output, _ = self.MultiheadAttention(
            query=text_embed,
            key=image_embed,
            value=image_embed
        )        
        attn_output = attn_output.squeeze(1)    # Shape: [batch_size, embed_dim]
        image_embed = image_embed.squeeze(1)    # Shape: [batch_size, embed_dim]

        # Concatenate image and attention outputs
        x = torch.cat([image_embed, attn_output], dim=1)  # Shape: [batch_size, embed_dim * 2]

        out = self.classifier(x)

        return out
# Example usage
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 10000
    embed_dim = 512
    device = "cuda"
    # Instantiate the model
    model = StableDiffusionInpaintingModel(vocab_size=vocab_size, embed_dim=embed_dim,device=device)
    model.to(device)

    # Dummy inputs
    batch_size = 4
    image = torch.randn(batch_size, 3, 256, 256)  # Input images with missing regions
    text = torch.randint(0, vocab_size, (batch_size, 20))  # Text descriptions
    text = ["I can see you","this is the second atempt", "this is a fake image", "this is a real image"]
    
    # Forward pass
    output,classification = model(image, text)
    print(output.shape)  # Should output torch.Size([4, 3, 256, 256])
    print(classification.shape)  # Should output torch.Size([4, 3, 256, 256])
