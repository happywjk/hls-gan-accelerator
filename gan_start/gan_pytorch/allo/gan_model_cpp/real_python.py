import torch
import os
import numpy as np

def extract_and_save_parameters_by_type(pth_file, output_dir):
    """
    Extracts weights, biases, and BatchNorm parameters from a .pth file and saves
    them into individual .data files, grouped by parameter type.

    Args:
        pth_file (str): The path to the .pth file.
        output_dir (str): The directory to save the .data files.
    """
    try:
        checkpoint = torch.load(pth_file)
        model_state_dict = checkpoint

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_idx = 1
        weights = []
        biases = []
        bn_weights = []
        bn_biases = []
        bn_means = []
        bn_vars = []
        
        # First, collect all parameter information
        conv_weights = {}  # To determine dimensions for zero biases
        
        # Identify BatchNorm layer prefixes first by finding running_mean or running_var
        bn_layer_prefixes = set()
        for name in model_state_dict.keys():
            if 'running_mean' in name or 'running_var' in name:
                # Extract the prefix (e.g., 'net.0.1' from 'net.0.1.running_mean')
                prefix = name.rsplit('.', 1)[0]
                bn_layer_prefixes.add(prefix)
        
        # Now categorize parameters
        for name, param in model_state_dict.items():
            # Check if this parameter belongs to a BatchNorm layer
            is_bn_param = False
            for prefix in bn_layer_prefixes:
                if name.startswith(prefix + '.'):
                    is_bn_param = True
                    break
            
            if is_bn_param:
                if 'weight' in name:
                    bn_weights.append((name, param.cpu().numpy()))
                elif 'bias' in name:
                    bn_biases.append((name, param.cpu().numpy()))
                elif 'running_mean' in name:
                    bn_means.append((name, param.cpu().numpy()))
                elif 'running_var' in name:
                    bn_vars.append((name, param.cpu().numpy()))
            else:
                if 'weight' in name:
                    weights.append((name, param.cpu().numpy()))
                    
                    # Store conv weights for dimension calculation
                    if 'net.0.0' in name or 'net.1.0' in name or 'net.2.0' in name or 'net.3.0' in name or 'net.4.0' in name:
                        conv_weights[name] = param.cpu().numpy()
                        
                elif 'bias' in name:
                    biases.append((name, param.cpu().numpy()))
        
        # Create zero biases for conv layers
        for name, weight_array in conv_weights.items():
            # For ConvTranspose2d, output channels is the first dimension
            out_channels = weight_array.shape[0]
            
            # Create layer name for the bias (replace '0.weight' with '0.bias')
            bias_name = name.replace('weight', 'bias')
            
            # Create zero bias with correct dimension
            zero_bias = np.zeros(out_channels, dtype=np.float32)
            
            # Add to biases list
            biases.append((bias_name, zero_bias))
        
        # Sort biases to maintain order
        biases.sort(key=lambda x: x[0])
            
        # Write weights
        for name, weight in weights:
            file_path = os.path.join(output_dir, f"input{file_idx}.data")
            with open(file_path, "w") as f:
                for val in weight.flatten():
                    f.write(f"{val}\n")
            print(f"Weights for '{name}' saved to '{file_path}'")
            file_idx += 1

        # Write biases
        for name, bias in biases:
            file_path = os.path.join(output_dir, f"input{file_idx}.data")
            with open(file_path, "w") as f:
                for val in bias.flatten():
                    f.write(f"{val}\n")
            print(f"Biases for '{name}' saved to '{file_path}'")
            file_idx += 1

        # Write BatchNorm weights (gamma)
        for name, bn_weight in bn_weights:
            file_path = os.path.join(output_dir, f"input{file_idx}.data")
            with open(file_path, "w") as f:
                for val in bn_weight.flatten():
                    f.write(f"{val}\n")
            print(f"BatchNorm weights (gamma) for '{name}' saved to '{file_path}'")
            file_idx += 1

        # Write BatchNorm biases (beta)
        for name, bn_bias in bn_biases:
            file_path = os.path.join(output_dir, f"input{file_idx}.data")
            with open(file_path, "w") as f:
                for val in bn_bias.flatten():
                    f.write(f"{val}\n")
            print(f"BatchNorm biases (beta) for '{name}' saved to '{file_path}'")
            file_idx += 1

        # Write BatchNorm running means
        for name, bn_mean in bn_means:
            file_path = os.path.join(output_dir, f"input{file_idx}.data")
            with open(file_path, "w") as f:
                for val in bn_mean.flatten():
                    f.write(f"{val}\n")
            print(f"BatchNorm running mean for '{name}' saved to '{file_path}'")
            file_idx += 1

        # Write BatchNorm running variances
        for name, bn_var in bn_vars:
            file_path = os.path.join(output_dir, f"input{file_idx}.data")
            with open(file_path, "w") as f:
                for val in bn_var.flatten():
                    f.write(f"{val}\n")
            print(f"BatchNorm running variance for '{name}' saved to '{file_path}'")
            file_idx += 1

        print(f"All parameters successfully extracted and saved to '{output_dir}'.")

    except FileNotFoundError:
        print(f"Error: File '{pth_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    pth_file_path = 'gen.pth'  # Replace with your .pth file path
    output_directory = './'  # Directory to save the .data files
    extract_and_save_parameters_by_type(pth_file_path, output_directory)