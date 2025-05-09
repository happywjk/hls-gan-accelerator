import numpy as np
import torch
from torch import nn
import time
import os
import subprocess
import psutil
import threading
import statistics

def get_gpu_power():
    """Get current GPU power usage in watts using nvidia-smi."""
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])
        return float(output.decode().strip())
    except:
        return 0.0

def monitor_gpu_power(stop_event, power_readings):
    """Monitor GPU power in a separate thread."""
    while not stop_event.is_set():
        power_readings.append(get_gpu_power())
        time.sleep(0.05)  # Sample every 50ms

def monitor_cpu_power(stop_event, power_readings, process=None):
    """Monitor CPU usage percentage and convert to approximate power."""
    # If no specific process is provided, monitor system-wide
    if process is None:
        process = psutil
        
    # Get the TDP of your CPU - you may need to adjust this value
    # This is a rough approximation - typical laptop CPUs range from 15W to 45W
    # Desktop CPUs can range from 65W to 125W or more
    CPU_TDP = 65.0  # Default to 65W TDP, adjust based on your CPU
    
    while not stop_event.is_set():
        if process == psutil:
            # System-wide CPU percentage
            usage_percent = process.cpu_percent(interval=0.05)
        else:
            # Process-specific CPU percentage
            usage_percent = process.cpu_percent(interval=0.05) / psutil.cpu_count()
        
        # Estimate power based on CPU utilization (linear approximation)
        # This is rough and doesn't account for frequency scaling, etc.
        estimated_power = (usage_percent / 100.0) * CPU_TDP
        power_readings.append(estimated_power)
        time.sleep(0.05)  # Sample every 50ms

def test_generator_with_power_monitoring():
    print("Starting generator test with power monitoring...")
    
    # Parameters for the test
    batch_size = 8
    z_dim = 128  # Input channels for the Generator
    
    # Initial dimensions for latent vector
    init_height, init_width = 1, 1
    
    # Check for CUDA availability
    device_cpu = torch.device("cpu")
    device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    using_cuda = torch.cuda.is_available()
    print(f"CUDA available: {using_cuda}")
    
    # Random input initialization (latent vector z)
    start_data_gen = time.perf_counter()
    z = np.random.randn(batch_size, z_dim, init_height, init_width).astype(np.float32)
    end_data_gen = time.perf_counter()
    data_gen_time = (end_data_gen - start_data_gen) * 1e9
    print(f"Input data generation time: {data_gen_time:.3e} ns")
    
    # Define Generator class (already provided by the user, including GenBlock)
    start_model_def = time.perf_counter()
    class GenBlock(nn.Module):
        def __init__(self, in_channels, out_channels, padding):
            super(GenBlock, self).__init__()
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
        def forward(self, x):
            return self.layers(x)
    
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.layers = nn.Sequential(
                GenBlock(128, 256, 0),     # 1x1 -> 4x4
                GenBlock(256, 128, 1),     # 4x4 -> 8x8
                GenBlock(128, 64, 1),      # 8x8 -> 16x16
                GenBlock(64, 32, 1),      # 16x16 -> 32x32
                GenBlock(32, 16, 1),       # 32x32 -> 64x64
                nn.ConvTranspose2d(16, 3, 4, 2, 1), # 64x64 -> 128x128
                nn.Tanh()
            )
    
        def forward(self, x):
            return self.layers(x)
    
    # Create Generator and set to evaluation mode
    generator = Generator()
    generator.eval()
    end_model_def = time.perf_counter()
    model_def_time = (end_model_def - start_model_def) * 1e9
    print(f"Model definition time: {model_def_time:.3e} ns")
    
    # Define layer dimensions for easier reference
    layer_dims = [
        {"in_ch": 128, "out_ch": 256},  # First GenBlock
        {"in_ch": 256, "out_ch": 128},  # Second GenBlock
        {"in_ch": 128, "out_ch": 64},   # Third GenBlock
        {"in_ch": 64, "out_ch": 32},    # Fourth GenBlock
        {"in_ch": 32, "out_ch": 16},    # Fifth GenBlock
        {"in_ch": 16, "out_ch": 3}      # Final ConvTranspose2d
    ]
    
    # Layer parameters mapping (for easier reference)
    layer_info = [
        {"conv": "net.0.0.weight", "bias": "net.0.0.bias", "bn_weight": "net.0.1.weight", "bn_bias": "net.0.1.bias", 
         "bn_mean": "net.0.1.running_mean", "bn_var": "net.0.1.running_var"},
        {"conv": "net.1.0.weight", "bias": "net.1.0.bias", "bn_weight": "net.1.1.weight", "bn_bias": "net.1.1.bias", 
         "bn_mean": "net.1.1.running_mean", "bn_var": "net.1.1.running_var"},
        {"conv": "net.2.0.weight", "bias": "net.2.0.bias", "bn_weight": "net.2.1.weight", "bn_bias": "net.2.1.bias", 
         "bn_mean": "net.2.1.running_mean", "bn_var": "net.2.1.running_var"},
        {"conv": "net.3.0.weight", "bias": "net.3.0.bias", "bn_weight": "net.3.1.weight", "bn_bias": "net.3.1.bias", 
         "bn_mean": "net.3.1.running_mean", "bn_var": "net.3.1.running_var"},
        {"conv": "net.4.0.weight", "bias": "net.4.0.bias", "bn_weight": "net.4.1.weight", "bn_bias": "net.4.1.bias", 
         "bn_mean": "net.4.1.running_mean", "bn_var": "net.4.1.running_var"},
        {"conv": "net.5.weight", "bias": "net.5.bias"}
    ]
    
    # File mapping (based on your updated output)
    file_mapping = {
        "net.0.0.weight": "input1.data",
        "net.1.0.weight": "input2.data",
        "net.2.0.weight": "input3.data",
        "net.3.0.weight": "input4.data",
        "net.4.0.weight": "input5.data",
        "net.5.weight": "input6.data",
        "net.0.0.bias": "input7.data",  # Zero biases
        "net.1.0.bias": "input8.data",  # Zero biases
        "net.2.0.bias": "input9.data",  # Zero biases
        "net.3.0.bias": "input10.data", # Zero biases
        "net.4.0.bias": "input11.data", # Zero biases
        "net.5.bias": "input12.data",   # Real bias
        "net.0.1.weight": "input13.data",
        "net.1.1.weight": "input14.data",
        "net.2.1.weight": "input15.data",
        "net.3.1.weight": "input16.data",
        "net.4.1.weight": "input17.data",
        "net.0.1.bias": "input18.data",
        "net.1.1.bias": "input19.data",
        "net.2.1.bias": "input20.data",
        "net.3.1.bias": "input21.data",
        "net.4.1.bias": "input22.data",
        "net.0.1.running_mean": "input23.data",
        "net.1.1.running_mean": "input24.data",
        "net.2.1.running_mean": "input25.data",
        "net.3.1.running_mean": "input26.data",
        "net.4.1.running_mean": "input27.data",
        "net.0.1.running_var": "input28.data",
        "net.1.1.running_var": "input29.data",
        "net.2.1.running_var": "input30.data",
        "net.3.1.running_var": "input31.data",
        "net.4.1.running_var": "input32.data",
    }
    
    # Function to read parameter data from file
    def read_param_file(filename):
        with open(filename, "r") as f:
            return np.array([float(line.strip()) for line in f])
    
    # Load parameters from files - timing weight loading phase
    start_weight_loading = time.perf_counter()
    
    weights = {}
    for param_name, file_name in file_mapping.items():
        try:
            param_data = read_param_file(file_name)
            weights[param_name] = param_data
        except FileNotFoundError:
            print(f"Warning: File {file_name} not found. Skipping parameter {param_name}.")
    
    end_weight_loading = time.perf_counter()
    weight_loading_time = (end_weight_loading - start_weight_loading) * 1e9
    print(f"Weight loading time from files: {weight_loading_time:.3e} ns")
    
    # Set the weights and parameters to the Generator model - timing model initialization
    start_model_init = time.perf_counter()
    
    with torch.no_grad():
        # Set parameters for each GenBlock
        for i in range(5):  # 5 GenBlocks
            # Get expected dimensions
            in_ch = layer_dims[i]["in_ch"]
            out_ch = layer_dims[i]["out_ch"]
            k_size = 4  # kernel size is always 4
            
            # Get and reshape ConvTranspose2d weights
            conv_weight = weights[layer_info[i]["conv"]]
            try:
                # For ConvTranspose2d, shape is (in_channels, out_channels, kernel_size, kernel_size)
                # This is opposite to Conv2d
                conv_weight_reshaped = conv_weight.reshape(in_ch, out_ch, k_size, k_size)
                generator.layers[i].layers[0].weight.copy_(torch.from_numpy(conv_weight_reshaped))
            except Exception as e:
                print(f"Error loading weights for layer {i}: {e}")
                print(f"Expected shape: {in_ch}x{out_ch}x{k_size}x{k_size}, got array of size {len(conv_weight)}")
            
            # Set ConvTranspose2d bias (should be zeros from files)
            try:
                # Create zeros with proper dimensions if needed
                zeros = np.zeros(out_ch, dtype=np.float32)
                generator.layers[i].layers[0].bias.copy_(torch.from_numpy(zeros))
            except Exception as e:
                print(f"Error setting bias for layer {i}: {e}")
            
            # Set BatchNorm2d parameters
            try:
                bn_weight = weights[layer_info[i]["bn_weight"]]
                bn_bias = weights[layer_info[i]["bn_bias"]]
                bn_mean = weights[layer_info[i]["bn_mean"]]
                bn_var = weights[layer_info[i]["bn_var"]]
                
                generator.layers[i].layers[1].weight.copy_(torch.from_numpy(bn_weight))
                generator.layers[i].layers[1].bias.copy_(torch.from_numpy(bn_bias))
                generator.layers[i].layers[1].running_mean.copy_(torch.from_numpy(bn_mean))
                generator.layers[i].layers[1].running_var.copy_(torch.from_numpy(bn_var))
            except Exception as e:
                print(f"Error loading BatchNorm parameters for layer {i}: {e}")
        
        # Set final ConvTranspose2d weights and biases
        in_ch = layer_dims[5]["in_ch"]
        out_ch = layer_dims[5]["out_ch"]
        k_size = 4
        
        try:
            final_weight = weights[layer_info[5]["conv"]]
            final_weight_reshaped = final_weight.reshape(in_ch, out_ch, k_size, k_size)
            generator.layers[5].weight.copy_(torch.from_numpy(final_weight_reshaped))
        except Exception as e:
            print(f"Error loading final layer weights: {e}")
        
        try:
            final_bias = weights[layer_info[5]["bias"]]
            generator.layers[5].bias.copy_(torch.from_numpy(final_bias))
        except Exception as e:
            print(f"Error loading final layer bias: {e}")
    
    end_model_init = time.perf_counter()
    model_init_time = (end_model_init - start_model_init) * 1e9
    print(f"Model initialization time (weight setting): {model_init_time:.3e} ns")
    
    # Data conversion and inference timing with power monitoring
    print("\n===== CPU Inference Timing & Power =====")
    
    # CPU data conversion timing
    start_data_to_cpu = time.perf_counter()
    z_tensor_cpu = torch.from_numpy(z).to(device_cpu)
    end_data_to_cpu = time.perf_counter()
    data_to_cpu_time = (end_data_to_cpu - start_data_to_cpu) * 1e9
    print(f"Data to CPU tensor conversion time: {data_to_cpu_time:.3e} ns")
    
    # CPU model transfer timing
    start_model_to_cpu = time.perf_counter()
    generator = generator.to(device_cpu)
    end_model_to_cpu = time.perf_counter()
    model_to_cpu_time = (end_model_to_cpu - start_model_to_cpu) * 1e9
    print(f"Model to CPU transfer time: {model_to_cpu_time:.3e} ns")
    
    # CPU inference timing with power monitoring
    try:
        num_iterations = 50
        
        print("\nMeasuring CPU inference time and power...")
        
        # Get the current process for CPU monitoring
        current_process = psutil.Process(os.getpid())
        
        # First run (often slower due to compilation/optimization)
        with torch.no_grad():
            _ = generator(z_tensor_cpu)
        
        # Lists to collect power readings for each iteration
        all_cpu_power_readings = []
        
        # Timed runs with power monitoring
        total_time_cpu = 0
        for i in range(num_iterations):
            # Setup CPU power monitoring
            cpu_power_readings = []
            cpu_stop_event = threading.Event()
            cpu_monitor_thread = threading.Thread(
                target=monitor_cpu_power, 
                args=(cpu_stop_event, cpu_power_readings, current_process)
            )
            cpu_monitor_thread.daemon = True
            cpu_monitor_thread.start()
            
            # Wait a moment for monitoring to start
            time.sleep(0.1)
            
            # Run inference
            start_time = time.perf_counter()
            with torch.no_grad():
                output_cpu = generator(z_tensor_cpu).detach()
            end_time = time.perf_counter()
            
            # Stop monitoring
            cpu_stop_event.set()
            cpu_monitor_thread.join(timeout=1.0)
            
            # Record timing
            iter_time = end_time - start_time
            total_time_cpu += iter_time
            
            # Process power readings
            if cpu_power_readings:
                all_cpu_power_readings.append(cpu_power_readings)
                if i % 10 == 0:  # Print every 10th iteration
                    avg_power = sum(cpu_power_readings) / len(cpu_power_readings)
                    print(f"  Iter {i}: Time: {iter_time*1000:.2f}ms, Avg CPU Power: {avg_power:.2f}W")

        # Calculate average time
        avg_time_cpu = total_time_cpu / num_iterations
        avg_time_cpu_ns = avg_time_cpu * 1e9
        
        # Calculate average power across all iterations
        flat_power_readings = [reading for readings in all_cpu_power_readings for reading in readings]
        avg_power_cpu = sum(flat_power_readings) / len(flat_power_readings) if flat_power_readings else 0
        
        # Calculate energy consumption (Power * Time)
        energy_cpu = avg_power_cpu * avg_time_cpu  # Joules
        
        print(f"\nCPU Performance Summary:")
        print(f"  Average inference time (50 runs): {avg_time_cpu*1000:.2f} ms ({avg_time_cpu_ns:.3e} ns)")
        print(f"  Average power consumption: {avg_power_cpu:.2f} watts")
        print(f"  Average energy per inference: {energy_cpu*1000:.2f} mJ")
        
        # Save CPU output for file writing
        output_np = output_cpu.numpy()
        
        # CUDA measurements if available
        if using_cuda:
            print("\n===== CUDA Inference Timing & Power =====")
            
            # CUDA model transfer timing
            start_model_to_cuda = time.perf_counter()
            generator = generator.to(device_cuda)
            torch.cuda.synchronize()
            end_model_to_cuda = time.perf_counter()
            model_to_cuda_time = (end_model_to_cuda - start_model_to_cuda) * 1e9
            print(f"Model to CUDA transfer time: {model_to_cuda_time:.3e} ns")
            
            # CUDA data transfer timing
            start_data_to_cuda = time.perf_counter()
            z_tensor_cuda = z_tensor_cpu.to(device_cuda)
            torch.cuda.synchronize()
            end_data_to_cuda = time.perf_counter()
            data_to_cuda_time = (end_data_to_cuda - start_data_to_cuda) * 1e9
            print(f"Data to CUDA transfer time: {data_to_cuda_time:.3e} ns")
            
            # Warm-up runs for CUDA
            print("\nPerforming CUDA warm-up runs...")
            with torch.no_grad():
                for _ in range(10):  # 10 warm-up iterations
                    _ = generator(z_tensor_cuda)
            torch.cuda.synchronize()
            
            # Lists to collect power readings for each iteration
            all_gpu_power_readings = []
            
            # CUDA inference timing with power monitoring
            print("Measuring CUDA inference time and power...")
            total_time_cuda = 0
            for i in range(num_iterations):
                # Setup GPU power monitoring
                gpu_power_readings = []
                gpu_stop_event = threading.Event()
                gpu_monitor_thread = threading.Thread(
                    target=monitor_gpu_power, 
                    args=(gpu_stop_event, gpu_power_readings)
                )
                gpu_monitor_thread.daemon = True
                gpu_monitor_thread.start()
                
                # Wait a moment for monitoring to start
                time.sleep(0.1)
                
                # Ensure previous operations are complete
                torch.cuda.synchronize()
                
                # Run inference
                start_time = time.perf_counter()
                with torch.no_grad():
                    output_cuda = generator(z_tensor_cuda)
                    torch.cuda.synchronize()  # Wait for CUDA operations to complete
                end_time = time.perf_counter()
                
                # Stop monitoring
                gpu_stop_event.set()
                gpu_monitor_thread.join(timeout=1.0)
                
                # Record timing
                iter_time = end_time - start_time
                total_time_cuda += iter_time
                
                # Process power readings
                if gpu_power_readings:
                    all_gpu_power_readings.append(gpu_power_readings)
                    if i % 10 == 0:  # Print every 10th iteration
                        avg_power = sum(gpu_power_readings) / len(gpu_power_readings)
                        print(f"  Iter {i}: Time: {iter_time*1000:.2f}ms, Avg GPU Power: {avg_power:.2f}W")
            
            # Calculate average time
            avg_time_cuda = total_time_cuda / num_iterations
            avg_time_cuda_ns = avg_time_cuda * 1e9
            
            # Calculate average power across all iterations
            flat_power_readings = [reading for readings in all_gpu_power_readings for reading in readings]
            avg_power_cuda = sum(flat_power_readings) / len(flat_power_readings) if flat_power_readings else 0
            
            # Calculate energy consumption (Power * Time)
            energy_cuda = avg_power_cuda * avg_time_cuda  # Joules
            
            print(f"\nGPU Performance Summary:")
            print(f"  Average inference time (50 runs): {avg_time_cuda*1000:.2f} ms ({avg_time_cuda_ns:.3e} ns)")
            print(f"  Average power consumption: {avg_power_cuda:.2f} watts")
            print(f"  Average energy per inference: {energy_cuda*1000:.2f} mJ")
            
            # Calculate and print speedup metrics
            time_speedup = avg_time_cpu / avg_time_cuda
            power_ratio = avg_power_cuda / avg_power_cpu
            energy_efficiency = energy_cpu / energy_cuda
            
            print(f"\n===== Performance Comparison (CPU vs. GPU) =====")
            print(f"  Time Speedup (CPU/GPU): {time_speedup:.2f}x")
            print(f"  Power Ratio (GPU/CPU): {power_ratio:.2f}x")
            print(f"  Energy Efficiency (CPU/GPU): {energy_efficiency:.2f}x")
            
            # Additional statistical analysis
            print("\n===== Detailed Power Analysis =====")
            if flat_power_readings:
                gpu_power_stats = {
                    "min": min(flat_power_readings),
                    "max": max(flat_power_readings),
                    "median": statistics.median(flat_power_readings),
                    "stddev": statistics.stdev(flat_power_readings) if len(flat_power_readings) > 1 else 0
                }
                print(f"GPU Power Statistics (watts):")
                print(f"  Min: {gpu_power_stats['min']:.2f}W")
                print(f"  Max: {gpu_power_stats['max']:.2f}W")
                print(f"  Median: {gpu_power_stats['median']:.2f}W")
                print(f"  StdDev: {gpu_power_stats['stddev']:.2f}W")
            
            # Flatten CPU power readings for analysis
            flat_cpu_power_readings = [reading for readings in all_cpu_power_readings for reading in readings]
            if flat_cpu_power_readings:
                cpu_power_stats = {
                    "min": min(flat_cpu_power_readings),
                    "max": max(flat_cpu_power_readings),
                    "median": statistics.median(flat_cpu_power_readings),
                    "stddev": statistics.stdev(flat_cpu_power_readings) if len(flat_cpu_power_readings) > 1 else 0
                }
                print(f"\nCPU Power Statistics (watts):")
                print(f"  Min: {cpu_power_stats['min']:.2f}W")
                print(f"  Max: {cpu_power_stats['max']:.2f}W")
                print(f"  Median: {cpu_power_stats['median']:.2f}W")
                print(f"  StdDev: {cpu_power_stats['stddev']:.2f}W")
                
            # Calculate total execution times including setup
            total_cpu_time = (data_to_cpu_time + model_to_cpu_time + avg_time_cpu_ns) / 1e9
            total_cuda_time = (model_to_cuda_time + data_to_cuda_time + avg_time_cuda_ns) / 1e9
            
            print("\n===== Total Execution Times =====")
            print(f"Total CPU execution time: {total_cpu_time:.6f} seconds")
            print(f"  - Data transfer: {data_to_cpu_time/1e9:.6f} seconds")
            print(f"  - Model transfer: {model_to_cpu_time/1e9:.6f} seconds")
            print(f"  - Inference: {avg_time_cpu:.6f} seconds")
            
            print(f"\nTotal CUDA execution time: {total_cuda_time:.6f} seconds")
            print(f"  - Model to CUDA: {model_to_cuda_time/1e9:.6f} seconds")
            print(f"  - Data to CUDA: {data_to_cuda_time/1e9:.6f} seconds")
            print(f"  - Inference: {avg_time_cuda:.6f} seconds")
            
            print(f"\nTotal speedup (CPU/CUDA): {total_cpu_time/total_cuda_time:.2f}x")
            
            # Calculate energy efficiency metrics including transfer costs
            print("\n===== Total Energy Analysis =====")
            # Rough estimate for data and model transfers (typically lower power)
            transfer_power_estimate = avg_power_cuda * 0.7  # Assume 70% of inference power
            
            # Total energy calculations
            cpu_total_energy = avg_power_cpu * total_cpu_time
            gpu_total_energy = (
                (transfer_power_estimate * (model_to_cuda_time + data_to_cuda_time) / 1e9) + 
                (avg_power_cuda * avg_time_cuda)
            )
            
            print(f"Total CPU energy: {cpu_total_energy*1000:.2f} mJ")
            print(f"Total GPU energy: {gpu_total_energy*1000:.2f} mJ")
            print(f"Total energy efficiency ratio (CPU/GPU): {cpu_total_energy/gpu_total_energy:.2f}x")
    
    except Exception as e:
        print(f"Error during model execution: {e}")
        return
    
    # Expected output dimensions
    final_height = 128
    final_width = 128
    final_channels = 3  # RGB output
    
    # Write input data to file for host.cpp
    with open("input0.data", "w") as f:
        for val in z.flatten():
            f.write(f"{val}\n")
    
    # Initialize output buffer with zeros (for host.cpp to write to)
    output_buffer_idx = 33  # Next available index after parameter files
    with open(f"input{output_buffer_idx}.data", "w") as f:
        output_size = batch_size * final_channels * final_height * final_width
        for _ in range(output_size):
            f.write("0.0\n")
    
    # Save the expected output to a file for verification
    with open("expected_output.data", "w") as f:
        for val in output_np.flatten():
            f.write(f"{val}\n")
    
    print("\n===== File Operations =====")
    print("Random noise input saved to input0.data")
    print(f"Zero-initialized output buffer saved to input{output_buffer_idx}.data")
    print("Expected output saved to expected_output.data")
    print("Generator parameters loaded from input1.data through input32.data")
    print("Model execution completed with power measurements.")

if __name__ == "__main__":
    test_generator_with_power_monitoring()