import os
import torch
from torchvision import transforms
from torchvision.io import read_image
from models.resnet_simclr import load_model
from tqdm import tqdm

# Path to the pre-trained model and the directory containing the images
model_path = "/media/hdd3/neo/MODELS/2024-05-20 Heme SimCLR/May17_08-34-51_path-lambda1/checkpoint_0100.pth.tar"
data_dir = "/media/hdd3/neo/pooled_deepheme_data"
save_dir = "/media/hdd3/neo/pooled_deepheme_data_SimCLR"

# Load the model
model = load_model(model_path)
model = model.to("cuda")
model.eval()

# Define image transformation for 96x96 images
transform = transforms.Compose(
    [
        transforms.Resize((96, 96)),  # Resize to the expected input size of the model
    ]
)

def process_directory(directory, save_directory):
    """Recursively process all images in the directory using the model."""
    # Prepare a list of all files to process
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
            ):
                image_files.append((os.path.join(root, file), save_directory))

    # Process each file with a progress bar
    for image_path, save_dir in tqdm(image_files, desc="Processing Images"):
        process_image(image_path, save_dir)

def process_image(image_path, output_dir):
    """Process an image and save the output tensor."""
    try:
        # Load and transform the image
        image = read_image(image_path).to("cuda")
        
        # Convert image from ByteTensor to FloatTensor and scale it
        image = image.float() / 255  # Normalize to range [0, 1]

        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Forward pass through the model
        with torch.no_grad():
            output = model(image)

        # Save the output tensor
        output_path = os.path.join(
            output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".pt"
        )
        torch.save(output, output_path)
        print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Failed to process {image_path}: {str(e)}")

# Start processing
process_directory(data_dir, save_dir)
