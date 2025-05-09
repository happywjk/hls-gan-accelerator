//=============================================================================
// Host code for GAN model with two GenBlocks
//=============================================================================

// OpenCL utility layer include
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
    const int BATCH_SIZE = 8;
    const int KERNEL_SIZE = 4;
    
    // Block 0: 128 -> 1024, 1x1 -> 4x4
    const int IN_CHANNELS_0 = 128;
    const int OUT_CHANNELS_0 = 256;
    const int INPUT_HEIGHT_0 = 1;
    const int INPUT_WIDTH_0 = 1;
    
    // Block 1: 1024 -> 512, 4x4 -> 8x8
    const int IN_CHANNELS_1 = 256;
    const int OUT_CHANNELS_1 = 128;
    
    // Block 2: 512 -> 256, 8x8 -> 16x16
    const int IN_CHANNELS_2 = 128;
    const int OUT_CHANNELS_2 = 64;
    
    // Block 3: 256 -> 128, 16x16 -> 32x32
    const int IN_CHANNELS_3 = 64;
    const int OUT_CHANNELS_3 = 32;
    
    // Block 4: 128 -> 64, 32x32 -> 64x64
    const int IN_CHANNELS_4 = 32;
    const int OUT_CHANNELS_4 = 16;
    
    // Final layer: 64 -> 3, 64x64 -> 128x128
    const int IN_CHANNELS_5 = 16;
    const int OUT_CHANNELS_5 = 3;
    const int OUTPUT_HEIGHT = 8;
    const int OUTPUT_WIDTH = 8;
    
    // Calculate sizes for buffers
    const int input_size = BATCH_SIZE * IN_CHANNELS_0 * INPUT_HEIGHT_0 * INPUT_WIDTH_0;
    const int weight_size_0 = IN_CHANNELS_0 * OUT_CHANNELS_0 * KERNEL_SIZE * KERNEL_SIZE;
    const int bias_size_0 = OUT_CHANNELS_0;
    const int weight_size_1 = IN_CHANNELS_1 * OUT_CHANNELS_1 * KERNEL_SIZE * KERNEL_SIZE;
    const int bias_size_1 = OUT_CHANNELS_1;
    const int output_size = BATCH_SIZE * OUT_CHANNELS_1 * OUTPUT_HEIGHT * OUTPUT_WIDTH;


    // Calculate sizes for intermediate buffers
    const int block0_output_size = BATCH_SIZE * OUT_CHANNELS_0 * 4 * 4;

    // Allocate Memory in Host Memory
    // Read input noise vector
    std::ifstream ifile0("input0.data");
    if (!ifile0.is_open()) {
      std::cerr << "Error: Could not open input0.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_input(input_size);
    for (unsigned i = 0; i < input_size; i++) {
      ifile0 >> source_input[i];
    }
    ifile0.close();
    
    // Read block 0 parameters
    // Weights for block 0
    std::ifstream ifile1("input1.data");
    if (!ifile1.is_open()) {
      std::cerr << "Error: Could not open input1.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_weights_0(weight_size_0);
    for (unsigned i = 0; i < weight_size_0; i++) {
      ifile1 >> source_weights_0[i];
    }
    ifile1.close();
    
    // Bias for block 0
    std::ifstream ifile2("input2.data");
    if (!ifile2.is_open()) {
      std::cerr << "Error: Could not open input2.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_bias_0(bias_size_0);
    for (unsigned i = 0; i < bias_size_0; i++) {
      ifile2 >> source_bias_0[i];
    }
    ifile2.close();
    
    // BatchNorm parameters for block 0
    // Gamma
    std::ifstream ifile3("input3.data");
    if (!ifile3.is_open()) {
      std::cerr << "Error: Could not open input3.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_gamma_0(bias_size_0);
    for (unsigned i = 0; i < bias_size_0; i++) {
      ifile3 >> source_gamma_0[i];
    }
    ifile3.close();
    
    // Beta
    std::ifstream ifile4("input4.data");
    if (!ifile4.is_open()) {
      std::cerr << "Error: Could not open input4.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_beta_0(bias_size_0);
    for (unsigned i = 0; i < bias_size_0; i++) {
      ifile4 >> source_beta_0[i];
    }
    ifile4.close();
    
    // Running mean
    std::ifstream ifile5("input5.data");
    if (!ifile5.is_open()) {
      std::cerr << "Error: Could not open input5.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_mean_0(bias_size_0);
    for (unsigned i = 0; i < bias_size_0; i++) {
      ifile5 >> source_mean_0[i];
    }
    ifile5.close();
    
    // Running var
    std::ifstream ifile6("input6.data");
    if (!ifile6.is_open()) {
      std::cerr << "Error: Could not open input6.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_var_0(bias_size_0);
    for (unsigned i = 0; i < bias_size_0; i++) {
      ifile6 >> source_var_0[i];
    }
    ifile6.close();
    
    // Read block 1 parameters
    // Weights for block 1
    std::ifstream ifile7("input7.data");
    if (!ifile7.is_open()) {
      std::cerr << "Error: Could not open input7.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_weights_1(weight_size_1);
    for (unsigned i = 0; i < weight_size_1; i++) {
      ifile7 >> source_weights_1[i];
    }
    ifile7.close();
    
    // Bias for block 1
    std::ifstream ifile8("input8.data");
    if (!ifile8.is_open()) {
      std::cerr << "Error: Could not open input8.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_bias_1(bias_size_1);
    for (unsigned i = 0; i < bias_size_1; i++) {
      ifile8 >> source_bias_1[i];
    }
    ifile8.close();
    
    // BatchNorm parameters for block 1
    // Gamma
    std::ifstream ifile9("input9.data");
    if (!ifile9.is_open()) {
      std::cerr << "Error: Could not open input9.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_gamma_1(bias_size_1);
    for (unsigned i = 0; i < bias_size_1; i++) {
      ifile9 >> source_gamma_1[i];
    }
    ifile9.close();
    
    // Beta
    std::ifstream ifile10("input10.data");
    if (!ifile10.is_open()) {
      std::cerr << "Error: Could not open input10.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_beta_1(bias_size_1);
    for (unsigned i = 0; i < bias_size_1; i++) {
      ifile10 >> source_beta_1[i];
    }
    ifile10.close();
    
    // Running mean
    std::ifstream ifile11("input11.data");
    if (!ifile11.is_open()) {
      std::cerr << "Error: Could not open input11.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_mean_1(bias_size_1);
    for (unsigned i = 0; i < bias_size_1; i++) {
      ifile11 >> source_mean_1[i];
    }
    ifile11.close();
    
    // Running var
    std::ifstream ifile12("input12.data");
    if (!ifile12.is_open()) {
      std::cerr << "Error: Could not open input12.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_var_1(bias_size_1);
    for (unsigned i = 0; i < bias_size_1; i++) {
      ifile12 >> source_var_1[i];
    }
    ifile12.close();
    
    // Output buffer initialization
    std::ifstream ifile13("input13.data");
    if (!ifile13.is_open()) {
      std::cerr << "Error: Could not open input13.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_output(output_size);
    for (unsigned i = 0; i < output_size; i++) {
      ifile13 >> source_output[i];
    }
    ifile13.close();

    // OPENCL HOST CODE AREA START
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    
    std::vector<float, aligned_allocator<float>> block0_output(block0_output_size, 0);
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
    
    // Allocate Buffers in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * input_size, source_input.data(), &err));
    
    OCL_CHECK(err, cl::Buffer buffer_weights_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                             sizeof(float) * weight_size_0, source_weights_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_bias_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_0, source_bias_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_gamma_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                           sizeof(float) * bias_size_0, source_gamma_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_beta_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_0, source_beta_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_mean_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_0, source_mean_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_var_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                         sizeof(float) * bias_size_0, source_var_0.data(), &err));
    
    OCL_CHECK(err, cl::Buffer buffer_weights_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                             sizeof(float) * weight_size_1, source_weights_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_bias_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_1, source_bias_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_gamma_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                           sizeof(float) * bias_size_1, source_gamma_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_beta_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_1, source_beta_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_mean_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_1, source_mean_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_var_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                         sizeof(float) * bias_size_1, source_var_1.data(), &err));

    OCL_CHECK(err, cl::Buffer buffer_block0_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
    sizeof(float) * block0_output_size, block0_output.data(), &err));

    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
                                           sizeof(float) * output_size, source_output.data(), &err));

    // Set kernel arguments
    int arg_idx = 0;
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_input));  // Input noise vector
    
    // Block 0 parameters
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_weights_0));
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_bias_0));
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_gamma_0));
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_beta_0));
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_mean_0));
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_var_0));
    
    // Block 1 parameters
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_weights_1));
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_bias_1));
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_gamma_1));
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_beta_1));
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_mean_1));
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_var_1));
    
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_block0_output));
    // Output
    OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_output));
    
    // Copy input data to device global memory
    std::vector<cl::Memory> inBufVec = {
        buffer_input,
        buffer_weights_0, buffer_bias_0, buffer_gamma_0, buffer_beta_0, buffer_mean_0, buffer_var_0,
        buffer_weights_1, buffer_bias_1, buffer_gamma_1, buffer_beta_1, buffer_mean_1, buffer_var_1,
        buffer_block0_output,
        buffer_output
    };
    
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects(inBufVec, 0 /* 0 means from host*/));

    cl::Event event;
    uint64_t nstimestart, nstimeend;
    std::cout << "|-------------------------+-------------------------|\n"
              << "| Kernel                  |    Wall-Clock Time (ns) |\n"
              << "|-------------------------+-------------------------|\n";

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_top, nullptr, &event));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    // Get the execution time
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    auto exe_time = nstimeend - nstimestart;

    std::cout << "| " << std::left << std::setw(24) << "top"
              << "|" << std::right << std::setw(24) << exe_time << " |\n";
    std::cout << "|-------------------------+-------------------------|\n";
    std::cout << "Note: Wall Clock Time is meaningful for real hardware execution "
              << "only, not for emulation.\n";
    std::cout << "Please refer to profile summary for kernel execution time for "
              << "hardware emulation.\n";
    std::cout << "Finished execution!\n\n";

    // Write the output data to file
    std::ofstream ofile;
    ofile.open("output.data");
    if (!ofile) {
        std::cerr << "Failed to open output file!" << std::endl;
        return EXIT_FAILURE;
    }
    for (unsigned i = 0; i < output_size; i++) {
        ofile << source_output[i] << std::endl;
    }
    ofile.close();
    
    std::cout << "Output size: " << output_size << " elements written to output.data" << std::endl;
    return EXIT_SUCCESS;
}