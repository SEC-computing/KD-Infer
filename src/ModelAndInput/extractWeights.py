# this script extracts weights and biases from an ONNX model, scales them, and saves them to binary files.
import os
import onnx
import numpy as np

def scale_array(array):
    """
    Scale the array by 2^12, floor, mod with a prime number (optional), and convert to uint64.

    Parameters:
        array (numpy.ndarray): Input array.
        prime_num (int): A prime number for the modulo operation.

    Returns:
        numpy.ndarray: Processed array.
    """
    # Scale the array
    scaled_array = array * (2 ** 12)

    # Floor the result
    floored_array = np.floor(scaled_array)

    max_bits = np.max(floored_array)
    min_bits = np.min(floored_array)
    

    """
    if int(max_bits).bit_length() > 12:
        print(f"bit length of max exceeding 12!")
    
    if int(min_bits).bit_length() > 12:
        print(f"bit length of min exceeding 12!")
    """

    # Convert to uint64
    return floored_array.astype(np.uint64)

def extract_and_process_weights(onnx_model_path, output_dir):
    """
    Extract weight and bias matrices from an ONNX model, process them, and save.

    Parameters:
        onnx_model_path (str): Path to the ONNX model file.
        output_dir (str): Directory to save the processed weights and biases.
        prime_num (int): A prime number for the modulo operation (optional).
    """
    # Load the ONNX model
    model = onnx.load(onnx_model_path)

    # Get the initializers (weights and biases)
    initializers = model.graph.initializer

    for initializer in initializers:
        # Convert initializer to a NumPy array
        array = onnx.numpy_helper.to_array(initializer)
        processed_array = scale_array(array)

        if 1:
            # Save the processed array to a .bin file
            output_file = os.path.join(output_dir, f"{initializer.name}.bin")
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'wb') as bin_file:
                bin_file.write(processed_array.tobytes())
    print(f"Save model parameters to {output_dir} successfully.")


# Example Usage
if __name__ == "__main__":
    # Path to the ONNX model file
    # onnx_model_path = "./checkpoints/resnet50_model_simple.onnx" #for resnet50
    # onnx_model_path = "./checkpoints/resnet34_model_simple.onnx" #for resnet34
    # onnx_model_path = "./checkpoints/vgg11_model_simple.onnx" #for vgg11
    onnx_model_path = "./checkpoints/alexnet_model_simple.onnx" #for alexnet

    # Directory to save the processed weights and biases
    # output_dir = "processed_weights_and_biases" #for resnet50
    # output_dir = "processed_weights_and_biases_resnet34" #for resnet34
    # output_dir = "processed_weights_and_biases_vgg11" #for vgg11
    output_dir = "processed_weights_and_biases_alexnet" #for alexnet

    # Define a prime number for the modulo operation
    # prime_num = 2199023190017 # for 41 bits

    # Extract, process, and save weights and biases
    extract_and_process_weights(onnx_model_path, output_dir)