from final_layer import UNet
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

model  = UNet()
# Load the image
image_path = "./png/test/22828930_15.png"
try:
    testImg = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format

    # Define the transformation to convert the image to a tensor
    transform = transforms.Compose([  # Resize to match your model's input size
        transforms.ToTensor() # Convert to tensor
    ])

    # Apply the transformation
    testImg_tensor = transform(testImg)  # Shape will be (C, H, W)

    # Add a batch dimension (1, C, H, W)
    testImg_tensor = testImg_tensor.unsqueeze(0)

    # Now you can pass the tensor to your model
    output = model(testImg_tensor)
    
    # Optionally, process output as needed
except Exception as e:
    print("Error processing the image:", e)

output = output.squeeze(0)  # Remove batch dimension, shape becomes (C, H, W)

# Use argmax to get the class index for each pixel
segmentation_map = torch.argmax(output, dim=0)  # Shape will be (H, W)

# Convert to numpy array
segmentation_map = segmentation_map.cpu().numpy()  # Move to CPU if on GPU

# Map class indices to colors (if needed, create a color map)
# Assuming 0 is background (black) and 1 is object (white)
color_map = np.array([[0, 0, 0],  # Class 0: black
                      [255, 255, 255]])  # Class 1: white

# Create an RGB image from the segmentation map
rgb_image = color_map[segmentation_map]

# Convert to uint8 and create a PIL image
rgb_image = rgb_image.astype(np.uint8)
pil_image = Image.fromarray(rgb_image)

# Save or display the image
pil_image.show()

