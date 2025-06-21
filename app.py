import streamlit as st
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as T
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
import torch.nn.functional as F

class VGG16Encoder(nn.Module):
    def __init__(self, embedding_dim, pretrained=True):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg16 = models.vgg16(weights=weights)
        self.features = nn.Sequential(*list(vgg16.features.children())[:-1])

        first_conv_layer = self.features[0]
        self.features[0] = nn.Conv2d(1, first_conv_layer.out_channels,
                                        kernel_size=first_conv_layer.kernel_size,
                                        stride=first_conv_layer.stride,
                                        padding=first_conv_layer.padding)

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 128, 128)
            output_features = self.features(dummy_input)
            self.flattened_size = output_features.view(output_features.size(0), -1).shape[1]

        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(self.flattened_size, self.embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class UNetWithFrozenEncoder(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super().__init__()
        self.encoder = encoder.features
        for param in self.encoder.parameters():
            param.requires_grad = False

        vgg_channels = [64, 128, 256, 512] # 4 pooling layers

        self.upconv1 = nn.ConvTranspose2d(vgg_channels[-1], vgg_channels[-2], kernel_size=2, stride=2)
        self.conv_decoder1 = nn.Conv2d(vgg_channels[-2] * 2, vgg_channels[-2], kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(vgg_channels[-2], vgg_channels[-3], kernel_size=2, stride=2)
        self.conv_decoder2 = nn.Conv2d(vgg_channels[-3] * 2, vgg_channels[-3], kernel_size=3, padding=1) # Added kernel_size=3

        self.upconv3 = nn.ConvTranspose2d(vgg_channels[-3], vgg_channels[-4], kernel_size=2, stride=2)
        self.conv_decoder3 = nn.Conv2d(vgg_channels[-4] * 2, vgg_channels[-4], kernel_size=3, padding=1)

        # Additional upsampling to reach 256x256
        self.upconv_final = nn.ConvTranspose2d(vgg_channels[-4], vgg_channels[-4] // 2, kernel_size=2, stride=2)
        self.conv_decoder_final = nn.Conv2d(vgg_channels[-4] // 2, vgg_channels[-4] // 2, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(vgg_channels[-4] // 2, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        pool_outputs = []
        x = x
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                pool_outputs.append(x)

        # Decoder forward pass with skip connections
        d1 = self.upconv1(pool_outputs[-1])
        d1 = torch.cat([d1, pool_outputs[-2]], dim=1)
        d1 = F.relu(self.conv_decoder1(d1))

        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, pool_outputs[-3]], dim=1)
        d2 = F.relu(self.conv_decoder2(d2))

        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, pool_outputs[-4]], dim=1)
        d3 = F.relu(self.conv_decoder3(d3))

        # Final upsampling
        d_final_up = self.upconv_final(d3)
        d_final = F.relu(self.conv_decoder_final(d_final_up))

        output = torch.sigmoid(self.final_conv(d_final))
        return output

# --- Load your trained segmentation model ---
def load_segmentation_model(model_path, device):
    embedding_dim = 512 # Adjust if your encoder's embedding dim was different
    loaded_encoder = VGG16Encoder(embedding_dim=embedding_dim, pretrained=False)
    segmentation_model = UNetWithFrozenEncoder(loaded_encoder, num_classes=1).to(device)
    try:
        segmentation_model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        st.error(f"Error: Model weights not found at {model_path}")
        return None
    except RuntimeError as e:
        st.error(f"Error loading state_dict: {e}")
        return None
    segmentation_model.eval()
    return segmentation_model

# --- Preprocessing function ---
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)

# --- Function to get lesion distances (modified for Streamlit) ---
def get_lesion_info_and_distances(image, model, pixel_spacing=(1.0, 1.0), threshold=0.5):
    original_width, original_height = image.size
    original_image_array = np.array(image.convert('L'))
    image_tensor = preprocess_image(image).to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_mask_resized = (output > threshold).float().squeeze().cpu().numpy()

    predicted_mask_original_size = np.array(Image.fromarray(predicted_mask_resized).resize((original_width, original_height), Image.NEAREST))
    labeled_mask_original_size = label(predicted_mask_original_size)
    regions_original_size = regionprops(labeled_mask_original_size)

    lesion_info = []
    for i, region in enumerate(regions_original_size):
        minr, minc, maxr, maxc = region.bbox
        area_pixels = region.area
        centroid_row, centroid_col = region.centroid
        lesion_info.append({
            'id': i + 1,
            'bbox': (minr, minc, maxr, maxc),
            'area_pixels': area_pixels,
            'centroid': (centroid_row, centroid_col)
        })

    distances = []
    if len(lesion_info) >= 2:
        for i in range(len(lesion_info)):
            for j in range(i + 1, len(lesion_info)):
                r1, c1 = lesion_info[i]['centroid']
                r2, c2 = lesion_info[j]['centroid']
                distance_pixels = np.sqrt((r2 - r1)**2 + (c2 - c1)**2)
                distance_real_world = np.sqrt(((c2 - c1) * pixel_spacing[1])**2 + ((r2 - r1) * pixel_spacing[0])**2)
                distances.append((lesion_info[i]['id'], lesion_info[j]['id'], distance_real_world))

    return original_image_array, predicted_mask_original_size, lesion_info, distances

# --- Main Streamlit App ---
def main():
    st.title("Necrotic Lung Lesion Distance Measurement")

    uploaded_file = st.file_uploader("Upload a CT Image...", type=["png", "jpg", "jpeg"])
    pixel_spacing_x = st.number_input("Pixel Spacing (X)", value=1.0)
    pixel_spacing_y = st.number_input("Pixel Spacing (Y)", value=1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'segmentation_unet.pth'
    model = load_segmentation_model(model_path, device)

    if model is not None and uploaded_file is not None:
        image = Image.open(uploaded_file)
        original_image, predicted_mask, lesion_info, distances = get_lesion_info_and_distances(
            image, model, pixel_spacing=(pixel_spacing_y, pixel_spacing_x)
        )

        st.subheader("Original CT Image")
        st.image(original_image, use_container_width=True)

        st.subheader("Detected Lesions")
        fig, ax = plt.subplots()
        ax.imshow(original_image, cmap='gray')
        ax.imshow(predicted_mask, cmap='viridis', alpha=0.5)
        for lesion in lesion_info:
            bbox = lesion['bbox']
            area = lesion['area_pixels']
            minr, minc, maxr, maxc = bbox
            rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, linewidth=1, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(minc, minr - 5, f"ID: {lesion['id']}, Area: {area}", color='lime', fontsize=8, ha='left', va='top')
            centroid_row, centroid_col = lesion['centroid']
            ax.plot(centroid_col, centroid_row, 'w+', markersize=5)
        st.pyplot(fig)

        st.subheader("Distances Between Lesions (mm)")
        if distances:
            for dist in distances:
                st.write(f"Lesion {dist[0]} and Lesion {dist[1]}: {dist[2]:.2f} mm")
        else:
            st.write("Less than two lesions detected.")

if __name__ == "__main__":
    main()
