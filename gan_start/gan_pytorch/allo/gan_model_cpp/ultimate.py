import os
import subprocess
import time
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from shutil import copyfile

def run_gan_workflow():
    """
    Automates the GAN workflow from model extraction to image generation
    """
    print("==== Starting GAN Hardware Accelerator Workflow ====")
    
    # Create output directories if they don't exist
    os.makedirs("pytorch_images", exist_ok=True)
    os.makedirs("hardware_images", exist_ok=True)
    
    # Step 1: Extract weights and biases from the pretrained model
    print("\n[1/5] Extracting weights and biases from pretrained model...")
    extract_weights_from_model()
    
    # Step 2: Run the PyTorch model to generate output
    print("\n[2/5] Running PyTorch model inference...")
    run_pytorch_model()
    
    # Step 3: Build and run hardware emulation
    print("\n[3/5] Building and running hardware emulation...")
    run_hardware_emulation()
    
    # Step 4: Convert hardware output to images
    print("\n[4/5] Converting hardware output to images...")
    hardware_output = convert_hardware_output()
    
    # Step 5: Convert PyTorch output to images (for comparison)
    print("\n[5/5] Converting PyTorch output to images (for comparison)...")
    pytorch_output = convert_pytorch_output()
    
    # Step 6: Display and save comparison images
    compare_outputs(pytorch_output, hardware_output)
    
    print("\n==== GAN Hardware Accelerator Workflow Completed ====")

def extract_weights_from_model():
    """
    Extract weights and biases from the pretrained model
    Uses the extract_and_save_parameters function from paste.txt
    """
    try:
        # Check if pretrained model exists
        if not os.path.exists('gen.pth'):
            print("Error: Pretrained model 'gen.pth' not found.")
            return False
        
        # Import the extraction function from extract_weights.py
        # Assuming the code from paste.txt is saved as extract_weights.py
        from real_python import extract_and_save_parameters_by_type
        
        # Extract parameters
        extract_and_save_parameters_by_type('gen.pth', './')
        print("Successfully extracted model parameters.")
        return True
    except Exception as e:
        print(f"Error extracting model parameters: {e}")
        return False

def run_pytorch_model():
    """
    Run the PyTorch GAN model to generate output
    Uses the test_generator function from paste-3.txt
    """
    try:
        # Import the test_generator function from test_generator.py
        # Assuming the code from paste-3.txt is saved as test_generator.py
        from real_gan import test_generator
        
        # Run the generator test
        test_generator()
        print("Successfully ran PyTorch model inference.")
        
        # Copy expected output to a timestamped file for reference
        timestr = time.strftime("%Y%m%d-%H%M%S")
        copyfile("expected_output.data", f"expected_output_{timestr}.data")
        print(f"Saved PyTorch output as expected_output_{timestr}.data")
        return True
    except Exception as e:
        print(f"Error running PyTorch model: {e}")
        return False

def run_hardware_emulation():
    """
    Run hardware emulation with the correct command
    """
    try:
        # First, check if Makefile exists
        if not os.path.exists('Makefile'):
            print("Error: Makefile not found.")
            return False
        
        # Get the XDEVICE environment variable
        xdevice = os.environ.get('XDEVICE')
        if not xdevice:
            print("Error: XDEVICE environment variable is not set.")
            return False
        
        # Print key information
        print(f"Using platform: {xdevice}")
        
        # Run the make command
        print("Building and running hardware emulation...")
        make_process = subprocess.run(['make', 'run', 'TARGET=hw', f'PLATFORM={xdevice}'], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, 
                                     text=True)
        
        # Always print stdout to see execution time and other important output
        print(make_process.stdout)
        
        if make_process.returncode != 0:
            print(f"Error during hardware emulation. Return code: {make_process.returncode}")
            print("==== ERROR OUTPUT ====")
            print(make_process.stderr)
            return False
        else:
            # Print stderr anyway as some tools output important info to stderr even on success
            if make_process.stderr:
                print("==== ADDITIONAL OUTPUT ====")
                print(make_process.stderr)
            
        print("Hardware emulation completed successfully.")
        
        # Copy the output to a timestamped file
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if os.path.exists("output.data"):
            copyfile("output.data", f"output_{timestr}.data")
            print(f"Saved hardware output as output_{timestr}.data")
        else:
            print("Warning: output.data file not found after emulation.")
        
        return True
    except Exception as e:
        print(f"Error during hardware emulation: {e}")
        return False

def convert_hardware_output():
    """
    Convert hardware output to images
    """
    try:
        if not os.path.exists("output.data"):
            print("Error: Hardware output file 'output.data' not found.")
            return None
        
        # Read output data
        images = []
        with open("output.data", "r") as f:
            for line in f:
                value = float(line.strip())
                images.append(value)
        
        # Convert to tensor and reshape
        images = torch.tensor(images)
        images = images.reshape(8, 3, 128, 128)
        images = (images + 1) / 2  # Normalize to [0, 1]
        
        # Save individual images
        timestr = time.strftime("%Y%m%d-%H%M%S")
        for i in range(images.shape[0]):
            img = images[i].permute(1, 2, 0).numpy()  # Convert to (H, W, C)
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Hardware Output {i+1}")
            plt.tight_layout()
            plt.savefig(f"hardware_images/hw_image_{timestr}_{i+1}.png")
            plt.close()
        
        # Create and save grid visualization
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            image = images[i].permute(1, 2, 0)  # Convert (C, H, W) -> (H, W, C)
            ax.imshow(image)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"hardware_images/hardware_grid_{timestr}.png")
        plt.close()
        
        print(f"Hardware images saved to hardware_images/hardware_grid_{timestr}.png")
        return images
        
    except Exception as e:
        print(f"Error converting hardware output: {e}")
        return None

def convert_pytorch_output():
    """
    Convert PyTorch expected output to images
    """
    try:
        if not os.path.exists("expected_output.data"):
            print("Error: PyTorch output file 'expected_output.data' not found.")
            return None
        
        # Read expected output data
        images = []
        with open("expected_output.data", "r") as f:
            for line in f:
                value = float(line.strip())
                images.append(value)
        
        # Convert to tensor and reshape
        images = torch.tensor(images)
        images = images.reshape(8, 3, 128, 128)
        images = (images + 1) / 2  # Normalize to [0, 1]
        
        # Save individual images
        timestr = time.strftime("%Y%m%d-%H%M%S")
        for i in range(images.shape[0]):
            img = images[i].permute(1, 2, 0).numpy()  # Convert to (H, W, C)
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"PyTorch Output {i+1}")
            plt.tight_layout()
            plt.savefig(f"pytorch_images/pt_image_{timestr}_{i+1}.png")
            plt.close()
        
        # Create and save grid visualization
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            image = images[i].permute(1, 2, 0)  # Convert (C, H, W) -> (H, W, C)
            ax.imshow(image)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"pytorch_images/pytorch_grid_{timestr}.png")
        plt.close()
        
        print(f"PyTorch images saved to pytorch_images/pytorch_grid_{timestr}.png")
        return images
        
    except Exception as e:
        print(f"Error converting PyTorch output: {e}")
        return None

def compare_outputs(pytorch_output, hardware_output):
    """
    Compare and visualize the difference between PyTorch and hardware outputs
    """
    if pytorch_output is None or hardware_output is None:
        print("Cannot compare outputs: One or both outputs are missing.")
        return
    
    # Create comparison visualization
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    
    # First row: PyTorch outputs
    for i in range(8):
        axes[0, i].imshow(pytorch_output[i].permute(1, 2, 0))
        axes[0, i].set_title(f"PyTorch {i+1}")
        axes[0, i].axis("off")
    
    # Second row: Hardware outputs
    for i in range(8):
        axes[1, i].imshow(hardware_output[i].permute(1, 2, 0))
        axes[1, i].set_title(f"Hardware {i+1}")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"comparison_{timestr}.png")
    plt.close()
    
    # Calculate and display error metrics
    mse = torch.mean((pytorch_output - hardware_output) ** 2)
    mae = torch.mean(torch.abs(pytorch_output - hardware_output))
    
    print("\n==== Comparison Results ====")
    print(f"Mean Squared Error: {mse.item():.6f}")
    print(f"Mean Absolute Error: {mae.item():.6f}")
    print(f"Comparison image saved to comparison_{timestr}.png")

# Execute the workflow when the script is run
if __name__ == "__main__":
    # Create required Python files from the pastes

    
    # Run the workflow
    run_gan_workflow()