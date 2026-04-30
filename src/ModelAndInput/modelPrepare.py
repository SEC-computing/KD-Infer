# this file converts the .pth file to .onnx file for given model
import torch
import os
import torchvision.models as models

# Step 1: Load the ResNet model and state dictionary
model_path = "./checkpoints/resnet50-0676ba61.pth"  # Path to your .pth file

if os.path.exists(model_path):
    # resnet = models.resnet50(pretrained=False)
    # resnet = models.resnet34(pretrained=False)
    # resnet = models.vgg11(pretrained=False)
    resnet = models.alexnet(pretrained=False)
    resnet.load_state_dict(torch.load(model_path))
    print(f"Loaded model weights from {model_path}")
else:
    print(f"{model_path} not found. Downloading pretrained model model from torchvision...")
    # resnet = models.resnet50(pretrained=True)
    # resnet = models.resnet34(pretrained=True)
    # resnet = models.vgg11(pretrained=True)
    resnet = models.alexnet(pretrained=True)

# Step 2: Set the model to evaluation mode
resnet.eval()

# Step 3: Define a dummy input with the expected input shape
dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels (RGB), 224x224 image size

# Step 4: Export the model to ONNX
# onnx_path = "./checkpoints/resnet50_model_simple.onnx"  # Desired ONNX file path
# onnx_path = "./checkpoints/resnet34_model_simple.onnx"  # Desired ONNX file path
onnx_path = "./checkpoints/alexnet_model_simple.onnx"  # Desired ONNX file path
torch.onnx.export(
    resnet, 
    dummy_input, 
    onnx_path, 
    export_params=True,  # Store the trained parameters in the model file
    opset_version=11,    # ONNX version to export to (11 is widely supported)
    do_constant_folding=True,  # Optimize constant expressions or not
    input_names=['input'],  # Name of the input layer
    output_names=['output'],  # Name of the output layer
    dynamic_axes={'input': {0: 'batch_size'},  # Allow variable batch size
                  'output': {0: 'batch_size'}}
)

print(f"ONNX model has been saved to {onnx_path}")