import os
import shutil
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
    """Recursively process all files in the directory using the model."""
    for root, dirs, files in tqdm(os.walk(directory), desc="Processing"):
        for file in tqdm(files, desc=root):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, directory)
            save_path = os.path.join(save_directory, relative_path)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
            ):
                process_image(file_path, save_path)
            else:
                # Copy non-image files to the new directory
                shutil.copy(file_path, save_path)


def process_image(image_path, output_dir):
    """Process an image and save the output tensor."""
    try:
        # Load and transform the image
        image = read_image(image_path).to("cuda")
        image = image.float() / 255  # Normalize to range [0, 1]
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Forward pass through the model
        with torch.no_grad():
            output = model(image)
            # squeeze the tensor to remove all dimensions of size 1
            output = output.squeeze()

        # Save the output tensor
        output_path = os.path.join(
            output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".pt"
        )
        torch.save(output, output_path)
        print(f"Processed and saved: {output_path}")
        print(f"Output shape: {output.shape}")

    except Exception as e:
        print(f"Failed to process {image_path}: {str(e)}")


# Start processing
process_directory(data_dir, save_dir)
