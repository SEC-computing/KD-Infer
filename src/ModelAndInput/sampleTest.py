# this script tests an ONNX plaintext model for image classification using a sample image.
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Step 1: Load the ONNX model
onnx_model_path = "./checkpoints/alexnet_model_simple.onnx"  # Replace with your ONNX file path
session = ort.InferenceSession(onnx_model_path)

# Step 2: Prepare input preprocessing
def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Define image transformations (resize, crop, normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),          # Convert image to PyTorch tensor
        transforms.Normalize(           # Normalize image using ImageNet mean and std
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.numpy()

# Step 3: Preprocess the input image
image_path = "resnet50_input_image_n02109961_36.jpeg" # Replace with your test image path, the true image label is 248
input_data = preprocess_image(image_path)

# Step 4: Run inference
input_name = session.get_inputs()[0].name  # Get input layer name
output_name = session.get_outputs()[0].name  # Get output layer name
outputs = session.run([output_name], {input_name: input_data})[0]

# Step 5: Post-process the output
# Apply softmax to get probabilities
probabilities = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
top_class = np.argmax(probabilities, axis=1)

print(f"Predicted class index: {top_class[0]}")