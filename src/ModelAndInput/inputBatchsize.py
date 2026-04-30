# This file gets the input and output sizes of each function

import onnxruntime as ort
import numpy as np
import csv
from PIL import Image
from torchvision import transforms
import onnx
from onnx import helper
import os
import matplotlib.pyplot as plt

# Step 1: Load the ONNX model
onnx_model_path = "./checkpoints/alexnet_model_simple.onnx"  # Path to your ONNX model


model = onnx.load(onnx_model_path)

# Add all intermediate nodes as outputs
for node in model.graph.node:
    for output in node.output:
        if output not in [o.name for o in model.graph.output]:
            print(f"Adding intermediate output: {output}")
            intermediate_output = helper.ValueInfoProto()
            intermediate_output.name = output
            model.graph.output.append(intermediate_output)

# Save the modified model
modified_model_path = "./checkpoints/alexnet_with_intermediates.onnx"
onnx.save(model, modified_model_path)

onnx_model_path = modified_model_path

session = ort.InferenceSession(onnx_model_path)

# Step 2: Get the list of all layer/node names
all_layers = [output.name for output in session.get_outputs()]
# print(f"All layers in the model: {all_layers}")

# Step 3: Define preprocessing for the input image
def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),         # Convert image to PyTorch tensor
        transforms.Normalize(          # Normalize using ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.numpy()

# Step 4: Preprocess the input
image_path = "resnet50_input_image_n02109961_36.jpeg"  # Replace with your test image path
input_data = preprocess_image(image_path)

# Step 5: Run inference and fetch outputs for each layer
input_name = session.get_inputs()[0].name  # Input layer name

# CSV file to save outputs
csv_file_path = "./checkpoints/alexnet_onnxlayer_outputs.csv"

numbers = []

# Open the CSV file for writing
with open(csv_file_path, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the header row
    csvwriter.writerow(["Layer Name", "Output Shape", "Output Data"])
    
    # Iterate through all layers to get their outputs
    for layer_name in all_layers:
        output = session.run([layer_name], {input_name: input_data})
        output_array = np.array(output)  # Convert to NumPy array
        
        shape = output_array.shape
        if len(shape) == 5:
            number = 8192 / (shape[3] * shape[4])
        else:
            number = 8192
        numbers.append(number)
        
        # Save layer name, output shape, and output data (truncated for readability)
        csvwriter.writerow([layer_name, output_array.shape, output_array.flatten()[:3]])  # Save first 3 elements

# Plot the calculated numbers
plt.figure()
plt.plot(numbers, marker='o')
plt.axhline(y=8192, color='r', linestyle='--', label='8192')
plt.xticks(range(0, len(numbers), 2), fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
# plt.xlabel('Function Index')
# plt.ylabel('Input Batch Size Needed at Offline')
# plt.title('Calculated Numbers per Layer')
# plt.legend()

plt.show()

# plt.savefig('./checkpoints/layer_numbers_plot.png')
# print("Plot saved to ./checkpoints/layer_numbers_plot.png")

# Delete the modified ONNX model file
if os.path.exists(modified_model_path):
    os.remove(modified_model_path)
    print(f"Deleted file: {modified_model_path}")
else:
    print(f"File not found: {modified_model_path}")


print(f"Layer outputs have been saved to {csv_file_path}")