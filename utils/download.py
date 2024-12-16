import os

from torchvision.models import resnet34

# Set the custom directory for TORCH_HOME
os.environ["TORCH_HOME"] = "/data/training_code/Pein/DETR/my_detr/pretrained"

# List of ResNet architectures
resnet_archs = {
    "resnet34": resnet34,
}

# Initialize and trigger download of pretrained weights
for arch_name, model_func in resnet_archs.items():
    print(f"Downloading pretrained weights for {arch_name}...")
    model = model_func(pretrained=True)
    print(f"{arch_name} weights downloaded.")
