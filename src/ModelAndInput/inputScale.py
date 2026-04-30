# this file convert the raw image to the scaled value with binary format
import numpy as np
from PIL import Image
from torchvision import transforms

# Step 1: Prepare input preprocessing
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

# Step 2: Preprocess the input image
image_path = "resnet50_input_image_n02109961_36.jpeg" # Replace with your test image path, the true image label is 248
input_data = preprocess_image(image_path)


##########checking the bit length of the scaled values##########
# Step 1: Scale the values
scaled_data = np.floor(input_data * (2 ** 12))


# Step 2: Convert the result to unsigned 64-bit integers
converted_data = scaled_data.astype(np.uint64)

# Shape: [N, channel, height, width]
print(f'the input shape is {converted_data.shape}')

exceed_bool = False

# Step 3: Check if the bit length exceeds 41
def check_bit_length(data, max_bits=41):
    for value in data.flatten():  # Flatten the array to iterate over all elements
        if int(value).bit_length() > max_bits:
            exceed_bool = True
            print(f'there is input bit length exceeding 41!')
            break
    return exceed_bool

# Check for values with bit lengths exceeding 41
exceeding_flag = check_bit_length(converted_data)


# Assuming converted_data is a NumPy array of dtype uint64
def save_to_binary_file(data, filename):
    # Save the NumPy array to a binary file
    data.tofile(filename)


# Save the converted_data to a file
filename = "uint64_inp.bin"
save_to_binary_file(converted_data, filename)
print(f"Data saved to {filename}")

