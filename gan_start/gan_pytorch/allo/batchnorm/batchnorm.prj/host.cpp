//=============================================================================
// Host code for BatchNorm2d testing
//=============================================================================

#include "xcl2.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include <iomanip>
#include <fstream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Program program;
    cl::Kernel krnl_top;

    // Define constants
    const int BATCH_SIZE = 4;
    const int NUM_CHANNELS = 16;
    const int HEIGHT = 64;
    const int WIDTH = 64;
    
    const int input_size = BATCH_SIZE * NUM_CHANNELS * HEIGHT * WIDTH;
    const int param_size = NUM_CHANNELS;
    
    // Read input data
    std::vector<float, aligned_allocator<float>> input_data(input_size);
    std::ifstream ifile0("input0.data");
    if (!ifile0.is_open()) {
        std::cerr << "Error: Could not open input0.data file.\n";
        return 1;
    }
    for (int i = 0; i < input_size; i++) {
        ifile0 >> input_data[i];
    }
    
    // Read BatchNorm parameters
    std::vector<float, aligned_allocator<float>> gamma(param_size);
    std::ifstream ifile1("input1.data");
    if (!ifile1.is_open()) {
        std::cerr << "Error: Could not open input1.data file.\n";
        return 1;
    }
    for (int i = 0; i < param_size; i++) {
        ifile1 >> gamma[i];
    }
    
    std::vector<float, aligned_allocator<float>> beta(param_size);
    std::ifstream ifile2("input2.data");
    if (!ifile2.is_open()) {
        std::cerr << "Error: Could not open input2.data file.\n";
        return 1;
    }
    for (int i = 0; i < param_size; i++) {
        ifile2 >> beta[i];
    }
    
    std::vector<float, aligned_allocator<float>> running_mean(param_size);
    std::ifstream ifile3("input3.data");
    if (!ifile3.is_open()) {
        std::cerr << "Error: Could not open input3.data file.\n";
        return 1;
    }
    for (int i = 0; i < param_size; i++) {
        ifile3 >> running_mean[i];
    }
    
    std::vector<float, aligned_allocator<float>> running_var(param_size);
    std::ifstream ifile4("input4.data");
    if (!ifile4.is_open()) {
        std::cerr << "Error: Could not open input4.data file.\n";
        return 1;
    }
    for (int i = 0; i < param_size; i++) {
        ifile4 >> running_var[i];
    }
    
    // Allocate output buffer
    std::vector<float, aligned_allocator<float>> output_data(input_size, 0.0f);

    // OPENCL HOST CODE AREA START
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_top = cl::Kernel(program, "top", &err));
            valid_device = true;
            break;
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    
    // Create buffers and allocate memory
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                          sizeof(float) * input_size, input_data.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_gamma(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                          sizeof(float) * param_size, gamma.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_beta(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                          sizeof(float) * param_size, beta.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_mean(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                          sizeof(float) * param_size, running_mean.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_var(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                          sizeof(float) * param_size, running_var.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
                          sizeof(float) * input_size, output_data.data(), &err));

    // Set kernel arguments
    OCL_CHECK(err, err = krnl_top.setArg(0, buffer_input));
    OCL_CHECK(err, err = krnl_top.setArg(1, buffer_gamma));
    OCL_CHECK(err, err = krnl_top.setArg(2, buffer_beta));
    OCL_CHECK(err, err = krnl_top.setArg(3, buffer_mean));
    OCL_CHECK(err, err = krnl_top.setArg(4, buffer_var));
    OCL_CHECK(err, err = krnl_top.setArg(5, buffer_output));
    
    // Copy input data to device
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input, buffer_gamma, buffer_beta,
                                                   buffer_mean, buffer_var}, 0));

    // Launch the kernel
    cl::Event event;
    OCL_CHECK(err, err = q.enqueueTask(krnl_top, nullptr, &event));
    
    // Copy results back
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    
    // Get execution time
    uint64_t nstimestart, nstimeend;
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    auto exe_time = nstimeend - nstimestart;

    std::cout << "Kernel execution time: " << exe_time << " ns\n";
    
    // Write output to file
    std::ofstream ofile("output.data");
    if (!ofile.is_open()) {
        std::cerr << "Failed to open output file!" << std::endl;
        return EXIT_FAILURE;
    }
    for (int i = 0; i < input_size; i++) {
        ofile << output_data[i] << std::endl;
    }
    ofile.close();
    
    std::cout << "Output written to output.data\n";
    return EXIT_SUCCESS;
}